import torch
from torch import nn
from torch.nn.init import xavier_uniform_, xavier_normal_
from .abstract import IDEALRec

class GRU4Rec(IDEALRec):
    def __init__(self, args):
        super(GRU4Rec, self).__init__(args)
        self.num_layers = args.num_layers
        self.hidden_size = args.hidden_size
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.emb_dropout = nn.Dropout(args.dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size*2,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.args = args
        self.apply(self._init_weights)
        self.load_feat_embeddings()

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)

    def forward(self, item_seq):
        seq_emb = self.item_embeddings(item_seq)
        if self.args.use_fuse:
            item_emb = self.add_position_embedding(item_seq, seq_emb)
            hidden_emb = self.get_fuse_emb(item_seq, item_emb)
        else:
            hidden_emb = self.emb_dropout(seq_emb)
        gru_output, _ = self.gru_layers(hidden_emb)
        seq_out = self.dense(gru_output) #B, N, hidden_size
        return seq_out

    def calculate_loss(self, batch):
        user_id, item_seq, pos, neg, answer = batch
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

        final_loss = bpr_loss + 0.7 * self.con_loss
        return final_loss

    def predict_full(self, batch):
        user_id, item_seq, item_seq_len, pos, neg, answer = batch
        seq_out = self.forward(item_seq)[:, -1, :]
        test_item_emb = self.item_embeddings.weight
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred
