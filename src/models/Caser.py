import torch
import torch.nn as nn
import torch.nn.functional as F
from .abstract import IDEALRec
from .loss import RegLoss
from torch.nn.init import normal_, xavier_normal_, constant_

class Caser(IDEALRec):
    def __init__(self, args):
        super(Caser, self).__init__(args)

        self.hidden_size = args.hidden_size
        self.n_h = args.nh
        self.n_v = args.nv
        self.reg_weight = args.reg_weight
        self.n_users = args.user_size

        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.user_embeddings = nn.Embedding(self.n_users, self.hidden_size)
        self.conv_v = nn.Conv2d(
            in_channels=1, out_channels=self.n_v, kernel_size=(args.max_seq_length, 1)
        )

        lengths = [i + 1 for i in range(args.max_seq_length)]
        self.conv_h = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=self.n_h,
                    kernel_size=(i, self.hidden_size),
                )
                for i in lengths
            ]
        )

        self.fc1_dim_v = self.n_v * self.hidden_size
        self.fc1_dim_h = self.n_h * len(lengths)
        fc1_dim_in = self.fc1_dim_v + self.fc1_dim_h
        self.fc1 = nn.Linear(fc1_dim_in, self.hidden_size)
        self.fc2 = nn.Linear(
            self.hidden_size + self.hidden_size, self.hidden_size
        )

        self.dropout = nn.Dropout(args.dropout_prob)
        self.ac_conv = nn.ReLU()
        self.ac_fc = nn.ReLU()
        self.reg_loss = RegLoss()
        self.loss_fct = nn.CrossEntropyLoss()

        self.apply(self._init_weights)
        self.load_feat_embeddings()

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, 0, 1.0 / module.embedding_dim)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, user_ids, item_seq, item_seq_len):
        seq_emb = self.item_embeddings(item_seq)
        if self.args.use_fuse:
            item_emb = self.add_position_embedding(item_seq, seq_emb)
            fusion_emb = self.get_fuse_emb(item_seq, item_emb)
            item_emb = self.merge_embeddings(seq_emb, fusion_emb, item_seq_len).unsqueeze(1)
        else:
            item_emb = seq_emb.unsqueeze(1)

        user_emb = self.user_embeddings(user_ids).squeeze(1)

        out, out_h, out_v = None, None, None
        if self.n_v:
            out_v = self.conv_v(item_emb)
            out_v = out_v.view(-1, self.fc1_dim_v)

        # horizontal conv layer
        out_hs = list()
        if self.n_h:
            for conv in self.conv_h:
                conv_out = self.ac_conv(conv(item_emb).squeeze(3))
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect

        out = torch.cat([out_v, out_h], 1)
        out = self.dropout(out)
        z = self.ac_fc(self.fc1(out))
        x = torch.cat([z, user_emb], 1)
        seq_output = self.ac_fc(self.fc2(x))
        return seq_output

    def merge_embeddings(self, item_emb, fusion_emb, seq_len):
        B, N, C = item_emb.size()
        range_tensor = torch.arange(N).unsqueeze(0).unsqueeze(-1)
        seq_len_expanded = seq_len.unsqueeze(1).expand(-1, N, -1)
        mask = range_tensor < seq_len_expanded
        mask = mask.expand(-1, -1, C)
        merged = torch.where(mask, fusion_emb, item_emb)
        return merged

    def reg_loss_conv_h(self):
        loss_conv_h = 0
        for name, parm in self.conv_h.named_parameters():
            if name.endswith("weight"):
                loss_conv_h = loss_conv_h + parm.norm(2)
        return self.reg_weight * loss_conv_h

    def calculate_loss(self, batch):
        user_id, item_seq, item_seq_len, answer = batch
        seq_out = self.forward(user_id, item_seq, item_seq_len)
        labels = answer.squeeze(1)
        logits = torch.matmul(seq_out, self.item_embeddings.weight.transpose(0, 1))
        ce_loss = self.loss_fct(logits, labels)

        reg_loss = self.reg_loss(
            [
                self.user_embeddings.weight,
                self.item_embeddings.weight,
                self.conv_v.weight,
                self.fc1.weight,
                self.fc2.weight,
            ]
        )
        loss = ce_loss + self.reg_weight * reg_loss + self.reg_loss_conv_h() + 0.7*self.con_loss
        return loss

    def predict_full(self, batch):
        user_id, item_seq, item_seq_len, answer = batch
        seq_out = self.forward(user_id, item_seq, item_seq_len)
        test_item_emb = self.item_embeddings.weight
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred


