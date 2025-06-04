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


class DyTHead(nn.Module):
    """
    DyT module for use in cross-attention of Multi-Head Attention (MHA).

    In the cross-attention scenario:
      - query comes from the decoder (shape: (batch, q_len, hidden_dim)),
      - key comes from the encoder (shape: (batch, k_len, hidden_dim)).

    The module first aggregates the key (e.g., by taking the mean) to obtain context,
    then concatenates this context with the query tokens to compute dynamic parameters,
    and finally applies the tanh activation: tanh(γ * query + β).
    """

    def __init__(self, hidden_dim):
        super(DyTHead, self).__init__()
        # After concatenation, the input dimension is 2 * hidden_dim
        self.gamma = nn.Linear(2 * hidden_dim, hidden_dim)
        self.beta = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, query, key):
        # Aggregate key by taking the mean over the sequence dimension to obtain context information
        key_context = key.mean(dim=1, keepdim=True)  # shape: (batch, 1, hidden_dim)
        # Expand the context to match the query length
        key_context = key_context.expand(-1, query.size(1), -1)
        # Concatenate the query with the context information
        combined = torch.cat([query, key_context], dim=-1)  # shape: (batch, q_len, 2 * hidden_dim)

        # Compute dynamic parameters based on the combined features
        gamma = self.gamma(combined)
        beta = self.beta(combined)
        # Apply the dynamic tanh activation on the query
        return torch.tanh(gamma * query + beta)
