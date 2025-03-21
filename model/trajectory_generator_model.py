import select

import torch
from torch import nn
from model.encoders import StateEncoder, BackgroundEncoder
from utils.config import Configuration


class TrajectoryGenerator(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(self.cfg.token_nums + 3, self.cfg.embedding_dim)
        self.self_state_encoder = StateEncoder(self.cfg.embedding_dim, 3)
        self.background_encoder = BackgroundEncoder(self.cfg.embedding_dim, self.cfg.pad_token,
                                                    7, 5)

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

        # process agent
        bz, t, sl, c = data['agent_info'].shape
        agent_token = data['agent_info'][:, :, :, 0]  # (bz, t, sl)
        agent_mask = torch.tensor((agent_token != -1))
        agent_token = agent_token.clone()
        agent_token[~agent_mask] = self.cfg.pad_token
        agent_token = agent_token.reshape(bz * t * sl).long().to(self.cfg.device, non_blocking=True)
        agent_embedding = self.token_embedding(agent_token)  # => (bz*t*sl, embed_dim)
        agent_embedding = agent_embedding.reshape(bz, t, sl, -1)
        agent_features = data['agent_info'][:, :, :, 1:].to(self.cfg.device, non_blocking=True)  # => (bz, t, sl, c-1)

        # process goal
        goal_token = data["goal"].reshape(bz).long().to(self.cfg.device, non_blocking=True)
        goal_embedding = self.token_embedding(goal_token)  # => (bz, embed_dim)
        background_state = self.background_encoder(agent_embedding, agent_features, goal_embedding, agent_mask)
        return background_state

    def forward(self, data):
        # Q
        self_state = self.encode_self_state(data)
        # K, V
        env_state = self.encode_agent_topology(data)

        raise NotImplementedError
        token_embedding = self.token_embedding(data['ego_feature'].to(self.cfg.device, non_blocking=True))
