import torch
import torch.nn as nn


class SelfStateEncoder(nn.Module):
    def __init__(self, feature_dim=3, embed_dim=128, num_heads=2):
        """
        Args:
            feature_dim: Dimensionality of the input features (heading, speed, acc)
            embed_dim: Dimensionality of the projected embedding (e.g., 128)
            num_heads: Number of attention heads in the self-attention module
        """
        super(SelfStateEncoder, self).__init__()
        # Linear projection to map the 3 input features to embed_dim
        self.linear = nn.Linear(feature_dim, embed_dim)
        # Layer normalization for stable training
        self.norm = nn.LayerNorm(embed_dim)
        # Self-attention module, although for a single token it is trivial
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, feature_dim) representing [heading, speed, acc].
        Returns:
            out: Tensor of shape (batch_size, embed_dim) used as the initial Q token.
        """
        # Expand input to add a sequence dimension (batch_size, 1, feature_dim)
        x = x.unsqueeze(1)
        # Project the input features to the embedding space
        token = torch.relu(self.linear(x))  # shape: (batch_size, 1, embed_dim)
        # Apply self-attention; with a single token, the attention result remains the same
        attn_output, _ = self.attn(token, token, token)
        # Add residual connection and layer normalization
        token = self.norm(token + attn_output)
        # Squeeze the sequence dimension and return the final embedding (batch_size, embed_dim)
        return token.squeeze(1)
