from model.encoders import StateEncoder, BackgroundEncoder
from torch import nn
from utils.config import Configuration
import torch


class TrajectoryGenerator(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()
        self.cfg = cfg
        # ─────────────────────── Embeddings ────────────────────────
        self.token_embedding = nn.Embedding(
            num_embeddings=cfg.token_nums,
            embedding_dim=cfg.embedding_dim,
            padding_idx=cfg.pad_token,
        )

        # Encoder
        # ───────────────────── State & Context Encoders ─────────────
        self.self_state_encoder = StateEncoder(
            pos_dim=cfg.embedding_dim,
            feat_dim=3,  # (heading, speed, acc)
        )

        # agent  features: heading, v, acc, length, width (+ optional dists) → feat_dim = 5
        self.background_encoder = BackgroundEncoder(
            pos_embed_dim=cfg.embedding_dim,
            pad_token=cfg.pad_token,
            feat_dim=5,
            abs_dis_local=5,
            dropout=cfg.dropout,
            num_layers=cfg.num_topy_layers,
        )

        # ───────────────────────── Decoder ──────────────────────────
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.embedding_dim,
            nhead=cfg.tf_de_heads,
            dim_feedforward=cfg.tf_de_dim,
            dropout=cfg.dropout,
            activation="relu",
        )
        self.trajectory_gen = nn.TransformerDecoder(
            decoder_layer, num_layers=cfg.tf_de_layers
        )
        self.output_projection = nn.Linear(cfg.embedding_dim, cfg.token_nums)

        # ─────────────── Normalisation helpers ─────────────────────
        self.input_norm = nn.LayerNorm(cfg.embedding_dim)
        self.memory_norm = nn.LayerNorm(cfg.embedding_dim)
        self.output_norm = nn.LayerNorm(cfg.embedding_dim)

    # ===============================================================
    #                         Encoder
    # ===============================================================
    def encoder(self, data):
        """Return self‑state (B,L,D) and env‑state (B,T,S+1,D)."""
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
        ego = data["ego_info"].to(self.cfg.device, non_blocking=True)
        ego = ego.unsqueeze(1).repeat(1, l, 1).reshape(b * l, -1)
        fused = self.self_state_encoder(input_embedding, ego)  # (B*L,D)
        return self.input_norm(fused.view(b, l, -1))  # (B,L,D)

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
        agent_info = data['agent_info']

        bz, timer, num_agent, _ = agent_info.shape

        # agent_token: (B, T, S)
        agent_token = agent_info[:, :, :, 0]
        agent_mask = (agent_token != -1)
        agent_token = agent_token.clone()
        agent_token[~agent_mask] = self.cfg.pad_token

        # Flatten for embedding: (B*T*S,)
        agent_token = agent_token.view(-1).long().to(self.cfg.device, non_blocking=True)
        agent_embedding = self.token_embedding(agent_token)
        agent_embedding = agent_embedding.view(bz, timer, num_agent, -1)

        # agent_features: (B, T, S, c-1)
        agent_features = agent_info[:, :, :, 1:6].to(self.cfg.device, non_blocking=True)

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

        return self.memory_norm(background_state)

    # Decoder parts
    def causal_mask(self, seq_len):
        """
        Create a causal (auto-regressive) mask of shape (seq_len, seq_len).
        Upper-triangular entries are True => future positions are masked.
        """
        return torch.triu(torch.ones(seq_len, seq_len, device=self.cfg.device), diagonal=1).bool()

    def decoder(self, tgt_emb, memory, tgt_mask=None, tgt_kp_mask=None):
        """
        Transformer decoder forward pass:
          1) self-attention on tgt_emb
          2) cross-attention on memory
        Return logits of vocab => (tgt seq len, batch size, vocab size)
        """
        dec = self.trajectory_gen(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_kp_mask,
        )
        return self.output_projection(self.output_norm(dec))

    # ===============================================================
    #                             Forward (Teacher‑Forcing)
    # ===============================================================
    def forward(self, data):
        """Teacher‑forcing forward pass *without* future leakage."""
        bz, T = data["input_ids"].shape
        kp_full = (data["input_ids"] == self.cfg.pad_token).to(self.cfg.device)

        self_state, env_state = self.encoder(data)  # self: (B,L,D) env: (B,T,S+1,D)
        rep_env = env_state.permute(1, 0, 2, 3)  # (T,B,A,D) for window extraction

        logits_step = []
        for t in range(1, T):  # last label is EOS → no need to predict
            tgt_len = t + 1
            tgt_emb = self_state[:, :tgt_len, :].transpose(0, 1)  # (tgt_len, B, D)
            tgt_kpm = kp_full[:, :tgt_len]  # (B, tgt_len)
            mem = self.get_env_window_around_t(rep_env, t)  # (3A,B,D)
            dec_out = self.decoder(
                tgt_emb=tgt_emb,
                memory=mem,
                tgt_mask=self.causal_mask(t + 1),
                tgt_kp_mask=tgt_kpm,
            )
            logits_step.append(dec_out[-1])  # only newest position

        pred_logits = torch.stack(logits_step, dim=1)  # (B,T-1,V)
        return pred_logits, self_state, rep_env

    # ===============================================================
    #                      Autoregressive Predict
    # ===============================================================
    def predict(self, data, predict_token_num: int = 10, with_logits=False):
        """
        Autoregressive generate tokens for 'predict_token_num' steps,
        BUT ALSO store the per-step logits (before argmax).

        Returns:
            generated_tokens: (B, predict_token_num) discrete tokens (argmax)
            all_step_logits:  (B, predict_token_num, vocab_size) raw logits at each step
        """
        bz, _ = data["input_ids"].shape
        # 1) Encode to get self_state, env_state
        self_state, env_state = self.encoder(data)

        rep_env = env_state.permute(1, 0, 2, 3)

        # 2) Extract velocity embedding => shape (B,1,D)
        start_emb = self_state[:, 0, :].unsqueeze(1)  # (B,1,D)
        generated  = torch.empty(bz, 0, dtype=torch.long, device=self.cfg.device)
        all_logits = torch.zeros(bz, predict_token_num, self.cfg.token_nums, device=self.cfg.device)

        # 3) Loop for each decoding step
        for step in range(predict_token_num):
            # a) Build partial_emb = [start_emb] + embeddings of previously generated tokens
            if generated.size(1):
                past_emb = self.token_embedding(generated)
                # cat along seq_len => (B, 1 + #sofar, D)
                partial_emb = torch.cat([start_emb, past_emb], dim=1)
            else:
                # if no tokens generated yet, partial_emb = velocity alone
                partial_emb = start_emb
            # b) transpose => (seq_len, B, D)
            partial_emb_tbd = partial_emb.transpose(0, 1)

            # c) Get environment "memory" (3 frames around t = step_i)
            mem = self.get_env_window_around_t(rep_env, step)  # (3A,B,D)

            # d) Causal mask
            causal_mask = self.causal_mask(partial_emb_tbd.size(0))

            # e) Decode
            dec_out = self.trajectory_gen(
                tgt=partial_emb_tbd,
                memory=mem,
                tgt_mask=causal_mask,
            )

            # f) Project to vocab => (seq_len, B, vocab)
            logits = self.output_projection(dec_out)

            # g) take the last position => (B, vocab size)
            next_logits = logits[-1]

            # h) Greedy argmax => (B, 1)
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

            # i) Append to outputs
            generated = torch.cat([generated, next_token], dim=1)

            all_logits[:, step, :] = next_logits

        return (generated, all_logits) if with_logits else generated


    # ---------------------------------------------------------------
    @staticmethod
    def get_env_window_around_t(rep_env: torch.Tensor, t: int):
        """
        rep_env: (T, B, A, D)
          - T: total time steps
          - B: batch size
          - A: agent_count (e.g., agent_num+1)
          - D: embedding dimension

        Returns a memory tensor of shape (some_len, B, D),
        where some_len = (#frames_in_window * A).

        Slices frames [t-2, t-1, t] with boundary checks,
        then flattens agent dimension and transposes to
        (some_len, B, D).
        """
        T, B, A, D = rep_env.shape
        ids = [max(0, min(T - 1, t + off)) for off in (-2, -1, 0)]
        frames = [rep_env[i] for i in ids]                # each (B,A,D)
        cat = torch.cat(frames, dim=1).transpose(0, 1)     # (3A,B,D)
        return cat


    @staticmethod
    def _get_mem_mask_around_t(mask_base, t):
        """
        mask_base: (B, T, 3*(S+1))
        returns  : (B, 3*(S+1))   –  matches memory_key_padding_mask shape
        """
        # just pick frame t (already repeated 3×)
        return mask_base[:, t]  # (B, source_len)
