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

    def encode_self_state(self, data):
        bz, sl = data['input_ids'].shape
        input_ids = data['input_ids'].reshape([bz * sl, ]).to(self.cfg.device, non_blocking=True)
        input_embedding = self.token_embedding(input_ids)
        ego_info = data['ego_info'].repeat(sl, 1)
        self_state = self.self_state_encoder(
            input_embedding,
            ego_info.to(self.cfg.device, non_blocking=True)
        )
        return self_state.reshape(bz, sl, -1)

    def encode_agent_topology(self, data):
        # (bz, agent, [token, heading, v, acc, length, width, abs_dis, hit_dis])
        bz, sl, _ = data['agent_info'].shape
        agent_token = data['agent_info'][:, :, 0].reshape([bz * sl, ]).to(torch.int).to(self.cfg.device,
                                                                                        non_blocking=True)
        agent_embedding = self.token_embedding(agent_token).reshape(bz, sl, -1)
        agent_features = data['agent_info'][:, :, 1:].to(self.cfg.device, non_blocking=True)

    def forward(self, data):
        # Q
        self_state = self.encode_self_state(data)
        # K
        env_state = self.encode_self_state(data)

        raise NotImplementedError
        token_embedding = self.token_embedding(data['ego_feature'].to(self.cfg.device, non_blocking=True))
