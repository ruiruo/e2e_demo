import select

import torch
from torch import nn
from model.encoders import SelfStateEncoder
from utils.config import Configuration


class TrajectoryGenerator(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(self.cfg.token_nums + 2, self.cfg.embedding_dim)
        self.self_state_encoder = SelfStateEncoder(self.cfg.embedding_dim, 3)

    def encoder(self, data):
        bz, sl = data['input_ids'].shape
        input_ids = data['input_ids'].reshape(bz * sl, -1)
        ego_info = data['ego_info'].reshape(bz * sl, -1)
        self_state = self.self_state_encoder(data['ego_info'].to(self.cfg.device, non_blocking=True))


        # TODO: add cross attention
        return self_state

    def forward(self, data):
        for i, j in data.items():
            print(i, j.shape)
        self_state = self.encoder(data)

        print(self_state.shape)

        raise NotImplementedError
        token_embedding = self.token_embedding(data['ego_feature'].to(self.cfg.device, non_blocking=True))
