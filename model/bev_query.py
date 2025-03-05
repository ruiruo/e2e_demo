import torch
from torch import nn
from timm.models.layers import trunc_normal_
from utils.config import Configuration


class BevQuery(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()
        self.cfg = cfg

        tf_layer = nn.TransformerDecoderLayer(d_model=self.cfg.query_en_dim, nhead=self.cfg.query_en_heads,
                                              batch_first=True, dropout=self.cfg.query_en_dropout)
        self.tf_query = nn.TransformerDecoder(tf_layer, num_layers=self.cfg.query_en_layers)

        self.pos_embed = nn.Parameter(torch.randn(1, self.cfg.query_en_bev_length, self.cfg.query_en_dim) * .02)

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if 'pos_embed' in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        trunc_normal_(self.pos_embed, std=.02)

    def forward(self, tgt_feature, img_feature):
        assert tgt_feature.shape == img_feature.shape
        batch_size, channel, h, w = tgt_feature.shape

        tgt_feature = tgt_feature.view(batch_size, channel, -1)
        img_feature = img_feature.view(batch_size, channel, -1)
        tgt_feature = tgt_feature.permute(0, 2, 1)  # [batch_size, seq_len, embed_dim]
        img_feature = img_feature.permute(0, 2, 1)  # [batch_size, seq_len, embed_dim]

        tgt_feature = tgt_feature + self.pos_embed
        img_feature = img_feature + self.pos_embed

        bev_feature = self.tf_query(tgt_feature, memory=img_feature)
        bev_feature = bev_feature.permute(0, 2, 1)

        bev_feature = bev_feature.view(batch_size, channel, h, w)
        return bev_feature
