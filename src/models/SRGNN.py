import math
import numpy as np
import torch
from torch import nn
from .layers import GNN
from .abstract import CAFM

class SRGNN(CAFM):
    def __init__(self, args):
        super(SRGNN, self).__init__(args)

        self.hidden_size = args.hidden_size
        self.step = args.num_layers
        self.device = args.device
        self.item_embeddings = nn.Embedding(args.item_size, self.hidden_size, padding_idx=0)
        self.gnn = GNN(self.hidden_size, self.step)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_fct = nn.CrossEntropyLoss()
        self.args = args
        self._reset_parameters()
        self.load_feat_embeddings()

    def _reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def _get_slice(self, item_seq):
        mask = item_seq.gt(0)
        items, n_node, A, alias_inputs = [], [], [], []
        max_n_node = item_seq.size(1)
        item_seq = item_seq.cpu().numpy()
        for u_input in item_seq:
            node = np.unique(u_input)
            if 0 in node:
                node = np.append(node[node != 0], 0)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))

            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break

                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1

            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)

            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        alias_inputs = torch.LongTensor(alias_inputs).to(self.device)
        A = torch.FloatTensor(np.array(A)).to(self.device)
        items = torch.LongTensor(items).to(self.device)

        return alias_inputs, A, items, mask

    def forward(self, item_seq, item_seq_len):
        alias_inputs, A, items, mask = self._get_slice(item_seq)
        seq_emb = self.item_embeddings(items)
        if self.args.use_fuse:
            item_emb = self.add_position_embedding(items, seq_emb)
            fusion_emb = self.get_fuse_emb(items, item_emb)
            hidden = self.merge_embeddings(seq_emb, fusion_emb, item_seq_len)
        else:
            hidden = seq_emb
        hidden = self.gnn(A, hidden)
        alias_inputs = alias_inputs.view(-1, alias_inputs.size(1), 1).expand(
            -1, -1, self.hidden_size
        )
        seq_hidden = torch.gather(hidden, dim=1, index=alias_inputs)
        ht = self.gather_indexes(seq_hidden, item_seq_len - 1) #B, hidden_size
        q1 = self.linear_one(ht).view(ht.size(0), 1, ht.size(1))
        q2 = self.linear_two(seq_hidden)

        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * seq_hidden * mask.view(mask.size(0), -1, 1).float(), 1)
        seq_output = self.linear_transform(torch.cat([a, ht], dim=1))
        return seq_output

    # Replace the encoding of the fill position with the original
    def merge_embeddings(self, item_emb, fusion_emb, seq_len):
        B, N, C = item_emb.size()
        range_tensor = torch.arange(N).unsqueeze(0).unsqueeze(-1)
        seq_len_expanded = seq_len.unsqueeze(1).expand(-1, N, -1)
        mask = range_tensor < seq_len_expanded
        mask = mask.expand(-1, -1, C)
        merged = torch.where(mask, fusion_emb, item_emb)
        return merged

    def calculate_loss(self, batch):
        user_id, item_seq, item_seq_len, answer = batch
        seq_out = self.forward(item_seq, item_seq_len)
        labels = answer.squeeze(1)
        logits = torch.matmul(seq_out, self.item_embeddings.weight.transpose(0, 1))
        ce_loss = self.loss_fct(logits, labels)
        loss = ce_loss + self.args.alpha * self.crm_loss
        return loss

    def predict_full(self, batch):
        user_id, item_seq, item_seq_len, answer = batch
        seq_out = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embeddings.weight
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred
