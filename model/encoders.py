import torch
import torch.nn as nn


class SelfStateEncoder(nn.Module):
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
        super(SelfStateEncoder, self).__init__()
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
