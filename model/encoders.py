import torch
import torch.nn as nn


class DynTanhNorm(nn.Module):
    """A toy Dynamic Tanh 'norm' that learns gamma,beta from input features."""

    def __init__(self, embed_dim):
        super(DynTanhNorm, self).__init__()
        self.param_gen = nn.Linear(embed_dim, 2 * embed_dim)

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        # generate gamma, beta: (batch, seq_len, embed_dim) each
        gb = self.param_gen(x)
        gamma, beta = gb.chunk(2, dim=-1)
        # apply tanh-based "normalization"
        out = torch.tanh(gamma * x + beta)
        return out


class StateEncoder(nn.Module):
    """
    - pos_dim: The embedding dimension for positional information (e.g., 256)
    - feat_dim: The dimension of the external features (e.g., 3 for heading, speed, acc)

    Overall idea:
      1. Use a single linear layer: (B, feat_dim) -> (B, 2*pos_dim)
         This projects the external features into 2*pos_dim to generate gamma and beta.
      2. Apply dynamic modulation to pos_emb: tanh( gamma * pos_emb + beta )
         Both gamma and beta have the shape (B, pos_dim).
      3. In this way, the few external features act as control signals to flexibly scale and
       shift the positional information.

    # Note: if we require better non-linear, add L-RELU-L later
    """

    def __init__(self, pos_dim=256, feat_dim=3):
        super(StateEncoder, self).__init__()
        # Generate (gamma, beta) where the input has feat_dim and the output is 2 * pos_dim
        self.param_gen = nn.Linear(feat_dim, 2 * pos_dim)

    def forward(self, pos_emb, features):
        """
        Parameters:
          pos_emb: (B, pos_dim)   -- Positional embedding (already projected to high dimensions)
          features: (B, feat_dim) -- Dynamic features (e.g., heading, speed, acc)

        Output:
          (B, pos_dim) -- The position representation after dynamic modulation
        """
        # Generate gamma and beta using the external features
        gamma_beta = self.param_gen(features)  # shape: (B, 2*pos_dim)
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # each of shape (B, pos_dim)

        # Apply dynamic activation using tanh
        out = torch.tanh(gamma * pos_emb + beta)  # (B, pos_dim)
        return out


class BackgroundEncoder(nn.Module):
    """
      1) add features into positional information
      2) encode agent topology by BERT ( add context into embedding)
      3) output it as  K, V
    """

    def __init__(self, pos_embed_dim=256, pad_token=0, feat_dim=7, abs_dis_local=5):
        super(BackgroundEncoder, self).__init__()
        # Feature = (heading, v, acc, length, width, abs_dis, hit_dis)
        # heading, v, acc -> speed
        # length, width -> Box size
        # abs_dis -> abstract distance to self
        # abs_dis -> abstract distance to self(hit box level)
        # Question: would it bring noise?
        self.agent_state_encoder = StateEncoder(pos_embed_dim, feat_dim=feat_dim)
        self.pad_token = pad_token
        self.abs_dis_local = abs_dis_local

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=pos_embed_dim,
            nhead=4,
            dim_feedforward=pos_embed_dim,
            dropout=0.1,
            activation="relu",
            norm_first=True,
        )

        self.topology_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
        )

    def forward(self, agent_emb, agent_feature, goal_emb, agent_mask=None):
        """
        - agent_emb: (batch, time, agent_num, pos_dim)
        - features:  (batch, time, agent_num, feat_dim)
        - goal_emb: (batch, pos_dim)
        - agent_mask: (batch, time, agent_num), 0 => not valid，
          out: (batch, time, agent_num, bert_hidden)
        """
        bz, t, sl, d_pos = agent_emb.shape
        # ============ (1) Flatten (bz*t*sl, d_pos) ============
        agent_emb_flat = agent_emb.reshape(bz * t * sl, d_pos)  # (batch_size', d_pos)
        agent_feature_flat = agent_feature.reshape(bz * t * sl, -1)  # (batch_size', feat_dim)

        # ============ (2) DyT ============
        #  -> (bz*t*sl, d_pos)
        shifted_agent_emb = self.agent_state_encoder(agent_emb_flat, agent_feature_flat)
        shifted_agent_emb = shifted_agent_emb.reshape(bz, t, sl, -1)

        # ============ (3) goal as a extra agent ============
        goal = goal_emb.unsqueeze(1).repeat(1, t, 1)
        goal = goal.unsqueeze(2)
        # (bz, t, 1, d_pos)

        # ============ (4) add Mask ============
        if agent_mask is not None:
            goal_mask = torch.ones(bz, t, 1, dtype=torch.bool, device=agent_mask.device)
            agent_mask_cat = torch.cat([agent_mask, goal_mask], dim=2)
            agent_mask_flat = agent_mask_cat.reshape(bz * t, sl + 1)
        else:
            agent_mask_flat = None
        topology = torch.cat([shifted_agent_emb, goal], dim=2).reshape(bz * t, sl + 1, d_pos)
        # (bz * t, agent +1, pos_emb)
        topology_emb = self.topology_encoder(topology.transpose(0, 1), src_key_padding_mask=~agent_mask_flat)
        topology_emb = topology_emb.transpose(0, 1)
        topology_emb = topology_emb.reshape(bz, t, sl + 1, d_pos)
        return topology_emb
