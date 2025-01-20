import torch
import torch.nn as nn
from .layers import LayerNorm, ACT2FN, CrossAttentionLayer

def load_tensor(pt_path):
    data = torch.load(pt_path, weights_only=True)
    return data

class Project(nn.Module):
    def __init__(self, args):
        super(Project, self).__init__()
        self.layernorm = LayerNorm(args.visual_size, eps=1e-12)
        self.dense_1 = nn.Linear(args.visual_size, args.hidden_size * 2)
        self.dense_2 = nn.Linear(args.hidden_size * 2, args.hidden_size)
        self.act = ACT2FN[args.hidden_act]

    def forward(self, input_tensor):
        hidden_states = self.layernorm(input_tensor)
        hidden_states = self.dense_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        return hidden_states

class IDEALRec(nn.Module):
    def __init__(self, args):
        super(IDEALRec, self).__init__()
        self.feat_embeddings = nn.Embedding(args.item_size, args.visual_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.project = Project(args)
        self.fuse_model = CrossAttentionLayer(args)
        self.dropout = nn.Dropout(args.dropout_prob)
        self.layernorm = LayerNorm(args.hidden_size, eps=args.eps)
        self.args = args
        self.crm_loss = 0

    def load_feat_embeddings(self):
        self.feat_embeddings.weight.data = load_tensor(self.args.visual_file)
        self.feat_embeddings.weight.requires_grad = False

    def get_attention_mask(self, item_seq):
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(attention_mask.device)
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def add_position_embedding(self, item_seq, seq_emb):
        B, N = item_seq.size()
        position_ids = torch.arange(N, dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embeddings = self.position_embeddings(position_ids)
        emb = self.layernorm(seq_emb + position_embeddings)
        emb = self.dropout(emb)
        return emb

    def get_fuse_emb(self, item_seq, item_emb):
        mask = self.get_attention_mask(item_seq)
        feat_emb = self.project(self.feat_embeddings(item_seq))
        if self.args.use_contrastive:
            self.crm_loss = self.CRM_loss(item_seq, feat_emb)
        fusion_emb = self.fuse_model(item_emb, feat_emb, mask)
        return fusion_emb

    def CRM_loss(self, input_ids, emb):
        B, N, D = emb.shape
        mask = input_ids != 0
        mask_expanded = mask.unsqueeze(-1).float()

        mean_embeddings = (emb * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        mean_embeddings_expanded = mean_embeddings.unsqueeze(1).expand(-1, N, -1)

        mse_loss = ((emb - mean_embeddings_expanded) ** 2) * mask_expanded
        mse_loss = mse_loss.sum(dim=(1, 2)) / mask_expanded.sum(dim=(1, 2))
        loss = (mse_loss + 0.0001 / (mse_loss + 1e-4)).mean()

        return loss

    def gather_indexes(self, output, gather_index):
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    @classmethod
    def load_model(cls, args):
        path = args.checkpoint_path
        state_dict = torch.load(path, map_location='cpu', weights_only=True)
        model = cls(args)
        model.load_state_dict(state_dict, strict=True)
        return model
