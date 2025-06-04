from model.trajectory_generator_model import TrajectoryGenerator
from torch.distributions import Categorical
from torch import nn
from utils.config import Configuration
import torch


class TrajectoryPolicy(nn.Module):
    """
    PPO-ready wrapper around your original TrajectoryGenerator.

    act()  -> sample 1 token, return (token, logp, value)
    evaluate_actions() -> batched logp, entropy, value for PPO loss
    """

    def __init__(self, cfg: Configuration):
        super().__init__()
        self.cfg = cfg

        # ---------- original generator ----------
        self.generator = TrajectoryGenerator(cfg)
        # ---------- value head ----------
        self.value_head = nn.Sequential(
            nn.Linear(cfg.embedding_dim, cfg.embedding_dim),
            nn.ReLU(),
            nn.Linear(cfg.embedding_dim, 1),
        )

    def _encode(self, data):
        self_state, env_state = self.generator.encoder(data)
        rep_env = env_state.permute(1, 0, 2, 3)
        return self_state, rep_env

    def _step_logits(self, self_state, rep_env, seq_len):
        """
        seq_len  = # tokens already generated  (≥1, includes start token)
        Returns logits for the NEXT token (B, vocab)
        """
        tgt_emb = self_state[:, :seq_len, :].transpose(0, 1)
        mem = self.generator.get_env_window_around_t(rep_env, seq_len - 1)
        logits = self.generator.decoder(
            tgt_emb,
            memory=mem,
            tgt_mask=self.generator.causal_mask(seq_len),
        )[-1]  # latest position
        return logits  # (B, vocab)

    @torch.no_grad()
    def act(self, data, seq_generated):
        """
        data: full observation dict (see your original forward)
        seq_generated: (B, t) tokens already generated (t ≥ 1)
          – typically the velocity/start token plus any past tokens
        """
        self_state, rep_env = self._encode(data)

        # replace part of self_state with embeddings for tokens already generated
        if seq_generated.size(1) > 1:
            past_emb = self.generator.token_embedding(seq_generated[:, 1:])
            self_state[:, 1:seq_generated.size(1), :] = past_emb

        logits = self._step_logits(self_state, rep_env, seq_generated.size(1))
        dist = Categorical(logits=logits)
        action = dist.sample()  # (B,)
        logp = dist.log_prob(action)  # (B,)
        value = self.value_head(self_state[:, 0, :]).squeeze(-1)  # critic on first token

        return action, logp, value

    def evaluate_actions(self, data, seq_generated, actions):
        """
        data, seq_generated as in `act`;  actions = tokens chosen earlier
        Returns logp, entropy, value  (all (B,))
        """
        self_state, rep_env = self._encode(data)

        if seq_generated.size(1) > 1:
            past_emb = self.generator.token_embedding(seq_generated[:, 1:])
            self_state[:, 1:seq_generated.size(1), :] = past_emb

        logits = self._step_logits(self_state, rep_env, seq_generated.size(1))
        dist = Categorical(logits=logits)

        logp = dist.log_prob(actions)
        entropy = dist.entropy()
        value = self.value_head(self_state[:, 0, :]).squeeze(-1)

        return logp, entropy, value
