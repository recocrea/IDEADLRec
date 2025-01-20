import copy
import torch
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

class SASRecModel(IDEALRec):
    def __init__(self, args):
        super(SASRecModel, self).__init__(args)
        self.model = Encoder(args)
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.args = args
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

    def forward(self, item_seq):
        seq_emb = self.item_embeddings(item_seq)
        item_emb = self.add_position_embedding(item_seq, seq_emb)
        mask = self.get_attention_mask(item_seq)
        if self.args.use_fuse:
            hidden_emb = self.get_fuse_emb(item_seq, item_emb)
        else:
            hidden_emb = item_emb

        seq_out = self.model(hidden_emb, mask) #B, N, hidden_size
        return seq_out

    def calculate_loss(self, batch):
        user_id, item_seq, item_seq_len, pos, neg, answer = batch
        seq_out = self.forward(item_seq)

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

        final_loss = bpr_loss + self.args.alpha * self.crm_loss
        return final_loss

    def predict_full(self, batch):
        user_id, item_seq, item_seq_len, pos, neg, answer = batch
        seq_out = self.forward(item_seq)[:, -1, :]
        test_item_emb = self.item_embeddings.weight
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred
