import select

import torch
from torch import nn
from model.encoders import SelfStateEncoder
from utils.config import Configuration


class TrajectoryGenerator(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(self.cfg.token_nums + 3, self.cfg.embedding_dim)
        self.self_state_encoder = SelfStateEncoder(
            feature_dim=3,
            embed_dim=self.cfg.embedding_dim,
            num_heads=2
        )

    def encoder(self, data):
        self_state = self.self_state_encoder(data['ego_feature'].to(self.cfg.device, non_blocking=True))
        goal_state = self.self_state_encoder(data['goal_feature'].to(self.cfg.device, non_blocking=True))
        # TODO: add cross attention


    def forward(self, data):
        token_embedding = self.token_embedding(data['ego_feature'].to(self.cfg.device, non_blocking=True))