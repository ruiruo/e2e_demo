import select

import torch
from torch import nn
from model.encoders import StateEncoder, BackgroundEncoder
from utils.config import Configuration


class TrajectoryGenerator(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()
        self.cfg = cfg
        # Embedding for tokens (inputs, outputs, etc.)
        self.token_embedding = nn.Embedding(
            num_embeddings=self.cfg.token_nums + 3,
            embedding_dim=self.cfg.embedding_dim
        )
        # Encoder
        # Self-state encoder: uses StateEncoder to fuse the vehicle's own info
        # (heading, speed, acc) into the token embedding
        self.self_state_encoder = StateEncoder(
            pos_dim=self.cfg.embedding_dim,
            feat_dim=3
        )
        # Background encoder: uses BackgroundEncoder to handle multiple agents + final goal
        self.background_encoder = BackgroundEncoder(
            pos_embed_dim=self.cfg.embedding_dim,
            pad_token=self.cfg.pad_token,
            feat_dim=7,  # features: (heading, v, acc, length, width, abs_dis, hit_dis)
            abs_dis_local=5  # example hyperparameter
        )
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.cfg.embedding_dim,
            nhead=self.cfg.tf_de_heads,
            dim_feedforward=self.cfg.tf_de_dim,
            dropout=self.cfg.tf_de_dropout,
            activation='relu'
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.cfg.tf_de_layers)
        # Projection from decoder hidden states to location vocabulary
        self.output_projection = nn.Linear(
            self.cfg.embedding_dim,
            self.cfg.token_nums + 3
        )

    # Encoder parts
    def encoder(self, data):
        """
          1) self_state (Q): (B, L, embed_dim)
          2) env_state  (K, V): (B, T, S+1, embed_dim)
        """
        self_state = self.encode_self_state(data)
        env_state = self.encode_agent_topology(data)
        return self_state, env_state

    def encode_self_state(self, data):
        """
        Encode the vehicle's own state (as Q).

        data['input_ids']: (B, L)
          - token IDs for each time step
        data['ego_info']: (B, 3)
          - dynamic features (e.g., heading, speed, acc)

        1) Flatten the tokens, embed them, fuse with ego_info via StateEncoder.
        2) Return shape: (B, L, embed_dim)
        """
        B, L = data['input_ids'].shape

        # (B, L) -> flatten -> (B*L,)
        input_ids = data['input_ids'].reshape(-1).to(self.cfg.device, non_blocking=True)
        input_embedding = self.token_embedding(input_ids)  # (B*L, embed_dim)

        # ego_info: (B, 3) -> replicate for L steps => (B, L, 3) -> (B*L, 3)
        ego_info = data['ego_info'].to(self.cfg.device, non_blocking=True).unsqueeze(1)
        ego_info = ego_info.repeat(1, L, 1).reshape(B * L, -1)

        # StateEncoder (DyT style): fuse input_embedding & ego_info
        self_state_flat = self.self_state_encoder(input_embedding, ego_info)  # (B*L, embed_dim)

        # reshape back to (B, L, embed_dim)
        self_state = self_state_flat.view(B, L, -1)
        return self_state

    def encode_agent_topology(self, data):
        """
        Encode agent topology (K, V).

        data['agent_info']: (B, T, S, c) where c includes:
           token, heading, v, acc, length, width, abs_dis, hit_dis
        data['goal']: (B,) representing goal token IDs

        1) Extract agent tokens, replace -1 with pad_token, embed them.
        2) Extract agent features, then pass them to BackgroundEncoder
           (which internally uses StateEncoder).
        3) The goal token is embedded and treated as an extra agent in the topology.
        4) Output shape: (B, T, S+1, embed_dim)
        """
        B, T, S, c = data['agent_info'].shape

        # agent_token: (B, T, S)
        agent_token = data['agent_info'][:, :, :, 0]  # tokens
        agent_mask = (agent_token != -1)  # True => valid
        agent_token = agent_token.clone()
        agent_token[~agent_mask] = self.cfg.pad_token  # fill pad_token for invalid

        # Flatten for embedding: (B*T*S,)
        agent_token = agent_token.view(-1).long().to(self.cfg.device, non_blocking=True)
        agent_embedding = self.token_embedding(agent_token)  # (B*T*S, embed_dim)
        agent_embedding = agent_embedding.view(B, T, S, -1)  # (B, T, S, embed_dim)

        # agent_features: (B, T, S, c-1)
        agent_features = data['agent_info'][:, :, :, 1:].to(self.cfg.device, non_blocking=True)

        # goal: (B,) => embedded => (B, embed_dim)
        goal_token = data["goal"].view(B).long().to(self.cfg.device, non_blocking=True)
        goal_embedding = self.token_embedding(goal_token)  # (B, embed_dim)

        # BackgroundEncoder => (B, T, S+1, embed_dim)
        background_state = self.background_encoder(
            agent_emb=agent_embedding,
            agent_feature=agent_features,
            goal_emb=goal_embedding,
            agent_mask=agent_mask
        )
        return background_state

    # Decoder parts
    def create_mask(self, seq_len):
        """
        Create a causal (auto-regressive) mask of shape (seq_len, seq_len).
        Upper-triangular entries are True => future positions are masked.
        """
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=self.cfg.device),
            diagonal=1
        ).bool()
        return mask

    def decoder(self, tgt_emb, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Transformer decoder forward pass:
          1) self-attention on tgt_emb
          2) cross-attention on memory
        Return logits of vocab => (tgt_seq_len, batch_size, vocab_size)
        """
        dec_out = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )  # => (tgt_seq_len, batch_size, embed_dim)

        # Project to vocabulary space
        pred_traj_points = self.output_projection(dec_out)  # (tgt_seq_len, batch_size, vocab_size)
        return pred_traj_points

    def forward(self, data):
        # data = {"input_ids", "labels", "agent_info", "ego_info", "goal"}
        # input_ids   ; (bz, sl)                   ; [bz, (BOS,...... TN) token]
        # labels      ; (bz, sl)                   ; [bz, (T0,...... EOS) token]
        # agent_info  ; (bz, sl, n_agent, f_agent) ; [bz, sl, n_agent, (features)]
        # ego_info    ; (bz, f_ego)                   ; [bz, (heading, v, acc)]
        # goal        ; (bz, 1)
        # Q, KV
        self_state, env_state = self.encoder(data)
        import pdb
        pdb.set_trace()
        raise NotImplementedError

    def predict(self, data, predict_token_num):
        # Encoder
        self_state, env_state = self.encoder(data)

        # Auto Regressive Decoder
        raise NotImplementedError
