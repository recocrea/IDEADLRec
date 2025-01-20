import math
import copy
import torch
import random
import numpy as np
import torch.nn as nn
from .layers import SelfAttentionLayer, LayerNorm
from .abstract import IDEALRec


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        layer = SelfAttentionLayer(args)
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(args.num_layers)]
        )

    def forward(self, hidden_states, attention_mask):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states


class CL4SRec(IDEALRec):

    def __init__(self, args):
        super(CL4SRec, self).__init__(args)

        self.model = Encoder(args)
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)

        self.batch_size = args.batch_size
        self.max_len = args.max_seq_length
        self.lmd = args.lmd
        self.tau = args.tau
        self.sim = args.sim

        self.mask_default = self.mask_correlated_samples(batch_size=self.batch_size)
        self.nce_fct = nn.CrossEntropyLoss()

        self.apply(self._init_weights)
        self.load_feat_embeddings()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


    def augment(self, item_seq, item_seq_len):
        aug_seq1 = []
        aug_seq2 = []
        for seq, length in zip(item_seq, item_seq_len):
            if length > 1:
                switch = random.sample(range(3), k=2)
            else:
                switch = [3, 3]
                aug_seq = seq

            if switch[0] == 0:
                aug_seq = self.item_crop(seq, length)
            elif switch[0] == 1:
                aug_seq = self.item_mask(seq, length)
            elif switch[0] == 2:
                aug_seq = self.item_reorder(seq, length)

            aug_seq1.append(aug_seq)

            if switch[1] == 0:
                aug_seq = self.item_crop(seq, length)
            elif switch[1] == 1:
                aug_seq = self.item_mask(seq, length)
            elif switch[1] == 2:
                aug_seq = self.item_reorder(seq, length)
            aug_seq2.append(aug_seq)

        return torch.stack(aug_seq1), torch.stack(aug_seq2)

    def item_crop(self, item_seq, item_seq_len, eta=0.6):
        num_left = math.floor(item_seq_len * eta)
        crop_begin = random.randint(self.max_len - item_seq_len, self.max_len - num_left)
        croped_item_seq = np.zeros(item_seq.shape[0])
        if crop_begin + num_left < item_seq.shape[0]:
            croped_item_seq[self.max_len-num_left:] = item_seq.cpu().detach().numpy()[crop_begin:crop_begin + num_left]
        else:
            croped_item_seq[self.max_len-num_left:] = item_seq.cpu().detach().numpy()[crop_begin:]
        return torch.tensor(croped_item_seq, dtype=torch.long, device=item_seq.device)

    def item_mask(self, item_seq, item_seq_len, gamma=0.3):
        num_mask = math.floor(item_seq_len * gamma)
        mask_index = random.sample(range(self.max_len-item_seq_len, self.max_len), k=num_mask)
        masked_item_seq = item_seq.cpu().detach().numpy().copy()
        masked_item_seq[mask_index] = 0
        return torch.tensor(masked_item_seq, dtype=torch.long, device=item_seq.device)

    def item_reorder(self, item_seq, item_seq_len, beta=0.6):
        num_reorder = math.floor(item_seq_len * beta)
        reorder_begin = random.randint(self.max_len - item_seq_len, self.max_len - num_reorder)
        reordered_item_seq = item_seq.cpu().detach().numpy().copy()
        shuffle_index = list(range(reorder_begin, reorder_begin + num_reorder))
        random.shuffle(shuffle_index)
        reordered_item_seq[reorder_begin:reorder_begin + num_reorder] = reordered_item_seq[shuffle_index]
        return torch.tensor(reordered_item_seq, dtype=torch.long, device=item_seq.device)

    def forward(self, item_seq):
        seq_emb = self.item_embeddings(item_seq)
        item_emb = self.add_position_embedding(item_seq, seq_emb)
        mask = self.get_attention_mask(item_seq)
        if self.args.use_fuse:
            hidden_emb = self.get_fuse_emb(item_seq, item_emb)
        else:
            hidden_emb = item_emb

        seq_out = self.model(hidden_emb, mask)  # B, N, hidden_size
        return seq_out

    def calculate_bpr_loss(self, seq_out, pos, neg):
        seq_emb = seq_out.view(-1, self.args.hidden_size)
        pos_emb = self.item_embeddings(pos).view(-1, self.args.hidden_size)
        neg_emb = self.item_embeddings(neg).view(-1, self.args.hidden_size)

        pos_logits = torch.sum(pos_emb * seq_emb, -1)
        neg_logits = torch.sum(neg_emb * seq_emb, -1)
        istarget = (pos > 0).view(pos.size(0) * self.args.max_seq_length).float()

        bpr_loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return bpr_loss


    def calculate_loss(self, batch):
        user_id, item_seq, item_seq_len, pos, neg, answer = batch
        B, _ = item_seq.size()
        seq_out = self.forward(item_seq)
        bpr_loss = self.calculate_bpr_loss(seq_out, pos, neg)
        aug_item_seq1, aug_item_seq2 = self.augment(item_seq, item_seq_len)
        seq_output1 = self.forward(aug_item_seq1).view(B, -1)
        seq_output2 = self.forward(aug_item_seq2).view(B, -1)

        nce_logits, nce_labels = self.info_nce(
            seq_output1,
            seq_output2,
            temp=self.tau,
            batch_size=B,
            sim=self.sim
        )

        nce_loss = self.nce_fct(nce_logits, nce_labels)

        return bpr_loss + self.lmd * nce_loss + 0.7 * self.con_loss

    def decompose(self, z_i, z_j, origin_z, batch_size):

        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.cdist(z, z, p=2)

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        alignment = positive_samples.mean()

        sim = torch.cdist(origin_z, origin_z, p=2)
        mask = torch.ones((batch_size, batch_size), dtype=bool)
        mask = mask.fill_diagonal_(0)
        negative_samples = sim[mask].reshape(batch_size, -1)
        uniformity = torch.log(torch.exp(-2 * negative_samples).mean())

        return alignment, uniformity

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot'):
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        if sim == 'cos':
            sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.T) / temp

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

    def predict_full(self, batch):
        user_id, item_seq, item_seq_len, pos, neg, answer = batch
        seq_out = self.forward(item_seq)[:, -1, :]
        test_item_emb = self.item_embeddings.weight
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred