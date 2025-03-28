from model.encoders import StateEncoder, BackgroundEncoder
from torch import nn
from utils.config import Configuration
import torch


class TrajectoryGenerator(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()
        self.cfg = cfg
        # Embedding for tokens (inputs, outputs, etc.)
        self.token_embedding = nn.Embedding(
            num_embeddings=self.cfg.token_nums,
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
            feat_dim=5,  # features: (heading, v, acc, length, width, abs_dis, hit_dis)
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
        self.trajectory_gen = nn.TransformerDecoder(decoder_layer, num_layers=self.cfg.tf_de_layers)
        # Projection from decoder hidden states to location vocabulary
        self.output_projection = nn.Linear(
            self.cfg.embedding_dim,
            self.cfg.token_nums
        )

    # Encoder parts
    def encoder(self, data):
        """
          1) self_state (Q): (B, L, embed)
          2) env_state  (K, V): (B, T, S+1, embed)
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
        2) Return shape: (B, L, embed)
        """
        b, l = data['input_ids'].shape

        # (B, L) -> flatten -> (B*L,)
        input_ids = data['input_ids'].reshape(-1).to(self.cfg.device, non_blocking=True)
        input_embedding = self.token_embedding(input_ids)  # (B*L, embed)

        # ego_info: (B, 3) -> replicate for L steps => (B, L, 3) -> (B*L, 3)
        ego_info = data['ego_info'].to(self.cfg.device, non_blocking=True).unsqueeze(1)
        ego_info = ego_info.repeat(1, l, 1).reshape(b * l, -1)

        # StateEncoder (DyT style): fuse input_embedding & ego_info
        self_state_flat = self.self_state_encoder(input_embedding, ego_info)  # (B*L, embed)

        # reshape back to (B, L, embed)
        self_state = self_state_flat.view(b, l, -1)
        return self_state

    def encode_agent_topology(self, data):
        """
        Encode agent topology (K, V).

        data['agent_info']: (B, T, S, c) where c includes:
           token, heading, v, acc, length, width, abs_dis, hit_dis
        data['goal']: (B) representing goal token IDs

        1) Extract agent tokens, replace -1 with pad_token, embed them.
        2) Extract agent features, then pass them to BackgroundEncoder
           (which internally uses StateEncoder).
        3) The goal token is embedded and treated as an extra agent in the topology.
        4) Output shape: (B, T, S+1, embed)
        """
        bz, timer, num_agent, c = data['agent_info'].shape

        # agent_token: (B, T, S)
        agent_token = data['agent_info'][:, :, :, 0]  # tokens
        agent_mask = (agent_token != -1)  # True => valid
        agent_token = agent_token.clone()
        agent_token[~agent_mask] = self.cfg.pad_token  # fill pad_token for invalid

        # Flatten for embedding: (B*T*S,)
        agent_token = agent_token.view(-1).long().to(self.cfg.device, non_blocking=True)
        agent_embedding = self.token_embedding(agent_token)  # (B*T*S, embed)
        agent_embedding = agent_embedding.view(bz, timer, num_agent, -1)  # (B, T, S, embed)

        # agent_features: (B, T, S, c-1)
        agent_features = data['agent_info'][:, :, :, 1:6].to(self.cfg.device, non_blocking=True)
        # goal: (B,) => embedded => (B, embed)
        goal_token = data["goal"].view(bz).long().to(self.cfg.device, non_blocking=True)
        goal_embedding = self.token_embedding(goal_token)  # (B, embed)

        # BackgroundEncoder => (B, T, S+1, embed)
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
        dec_out = self.trajectory_gen(
            tgt=tgt_emb,
            memory=memory,  # shape => (some_mem_len, B, D)
            tgt_mask=tgt_mask
        )
        # => (tgt_seq_len, batch_size, embed)
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
        bz, length = data["input_ids"].shape
        # Q, KV
        self_state, env_state = self.encoder(data)
        bz, t, num_agents, dim_agents = env_state.shape
        env_state = env_state.permute(1, 0, 2, 3)
        mem = [self.get_env_window_around_t(env_state, i) for i in range(t)]
        mem = torch.concatenate([torch.unsqueeze(i, 0) for i in mem], dim=0)
        # mem = [t, agent * 3, bz, dim] example: torch.Size([10, 11*3, 4, 256])
        mem = mem.reshape(t * num_agents * 3, bz, dim_agents)
        tgt = self_state.transpose(0, 1)
        # tgt = [t, bz, dim]
        causal_mask = self.create_mask(length)
        # cross attention
        try:
            pred_logits = self.decoder(
                tgt_emb=tgt,
                memory=mem,
                tgt_mask=causal_mask
            )
        except:
            import pdb
            pdb.set_trace()
        # (bz, length, vocab)
        pred_logits = pred_logits.transpose(0, 1)
        return pred_logits, self_state, env_state

    def predict(self, data, predict_token_num=10):
        """
        Autoregressive generation loop with the velocity embedding always at position 0.

        Assumptions:
          - self_state has shape (B, 2, D) after encoding:
             * self_state[:, 0, :] => velocity embedding
             * self_state[:, 1, :] => some 'initial' token or other state
          - We'll generate 'predict_token_num' new tokens,
            each appended after the initial token(s).
        """
        bz, length = data["input_ids"].shape
        # Suppose data["input_ids"] has length=2 => [BOS, something], or you just rely on self_state

        # 1) Encode to get self_state, env_state
        self_state, env_state = self.encoder(data)
        env_state = env_state.permute(1, 0, 2, 3)
        # 2) Extract velocity embedding => shape (B,1,D)
        start_emb = self_state[:, 0, :].unsqueeze(1)

        # (B,0) => empty
        generated_tokens = torch.empty((bz, 0), dtype=torch.long, device=self.cfg.device)

        # 3) Build the partial embedding:
        for step_i in range(predict_token_num):
            #    [ start_emb ] + embeddings of all previously generated tokens
            # a) embed the previously generated tokens => (B, #sofar, D)
            if generated_tokens.size(1) > 0:
                token_emb = self.token_embedding(generated_tokens)  # => (B, #sofar, D)
                # cat along seq_len => (B, 1 + #sofar, D)
                partial_emb = torch.cat([start_emb, token_emb], dim=1)
            else:
                # if no tokens generated yet, partial_emb = velocity alone
                partial_emb = start_emb

            # b) transpose => (seq_len, B, D)
            partial_emb_tbd = partial_emb.transpose(0, 1)  # shape => (#seq, B, D)

            # 4) gather environment window => shape (3*A, B, D) or single slice, etc.
            mem = self.get_env_window_around_t(env_state, step_i)

            # 5) causal mask => (#seq, #seq)
            causal_mask = self.create_mask(partial_emb_tbd.size(0))

            dec_out = self.trajectory_gen(
                tgt=partial_emb_tbd,
                memory=mem,
                tgt_mask=causal_mask
            )

            # 7) project => (#seq, B, vocab_size)
            logits = self.output_projection(dec_out)

            # 8) take the last position => (B, vocab_size)
            next_token_logits = logits[-1, :, :]

            # 9) greedy => (B,1)
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # 10) append => (B, step_i+1)
            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

        return generated_tokens

    @staticmethod
    def get_env_window_around_t(rep_env, t):
        """
        rep_env: (T, B, A, D)
          - T: total time steps
          - B: batch size
          - A: agent_count (e.g., agent_num+1)
          - D: embedding dimension

        Returns a memory tensor of shape (some_len, B, D),
        where some_len = (#frames_in_window * A).

        Slices frames [t-1, t, t+1] with boundary checks,
        then flattens agent dimension and transposes to
        (some_len, B, D).
        """
        timestamp, bz, agent, dim = rep_env.shape

        # List of frames (B, A, D)
        frames_list = []
        for offset in [-1, 0, 1]:
            # clamp
            i_clamped = max(0, min(timestamp - 1, t + offset))
            # shape => (B, A, D)
            frames_list.append(rep_env[i_clamped])

        # Concatenate along agent/time dimension => shape (B, #frames*A, D)
        frames_cat = torch.cat(frames_list, dim=1)

        # Transpose to ( #frames*A, B, D ), the standard shape for cross-attention
        frames_cat = frames_cat.transpose(0, 1)
        return frames_cat
