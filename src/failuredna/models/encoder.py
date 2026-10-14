"""FailureDNA Temporal Transformer Encoder and Projection Head.

Implements multi-layer Transformer encoder with positional encoding,
masked temporal pooling, non-linear projection head, and supervised RUL heads.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from failuredna.utils.logging import get_logger

logger = get_logger(__name__)


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for temporal sequence modeling."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encodings to token embeddings: x is (B, W, d_model)."""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MaskedTemporalPooling(nn.Module):
    """Explicitly pools sequence representations while discounting left-padded cycles."""

    def __init__(self, method: str = "masked_mean", d_model: int = 128):
        super().__init__()
        self.method = method.lower()
        if self.method not in ("masked_mean", "last", "attention"):
            raise ValueError(f"Unsupported pooling method: {method}. Choose from 'masked_mean', 'last', 'attention'.")

        if self.method == "attention":
            self.att_linear = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Pool sequence tensor (B, W, D) into a fixed vector (B, D).

        Args:
            x: Sequence features of shape (B, W, D).
            mask: Optional binary validity mask of shape (B, W) where 1=observed, 0=padded.

        Returns:
            Pooled feature vector of shape (B, D).
        """
        B, W, D = x.shape

        if mask is None:
            mask = torch.ones((B, W), dtype=torch.float32, device=x.device)
        else:
            mask = mask.to(x.device).float()

        if self.method == "masked_mean":
            # Expand mask to (B, W, 1)
            mask_expanded = mask.unsqueeze(-1)
            sum_features = torch.sum(x * mask_expanded, dim=1)
            mask_sum = torch.clamp(torch.sum(mask_expanded, dim=1), min=1.0)
            return sum_features / mask_sum

        elif self.method == "last":
            # Extract the last observed time step per batch item
            last_indices = torch.clamp(torch.sum(mask, dim=1).long() - 1, min=0, max=W - 1)
            batch_idx = torch.arange(B, device=x.device)
            return x[batch_idx, last_indices]

        elif self.method == "attention":
            # Score each time step with learnable query
            scores = self.att_linear(x).squeeze(-1)  # (B, W)
            # Mask out padded positions
            scores = scores.masked_fill(mask == 0, -1e9)
            weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # (B, W, 1)
            return torch.sum(x * weights, dim=1)

        raise RuntimeError("Unreachable pooling code path.")


class TemporalTransformerEncoder(nn.Module):
    """Primary FailureDNA Temporal Transformer Encoder."""

    def __init__(
        self,
        input_dim: int = 24,
        hidden_dim: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        pooling: str = "masked_mean",
        max_window_size: int = 200,
    ):
        """Initialize TemporalTransformerEncoder.

        Args:
            input_dim: Number of input telemetry features (settings + sensors).
            hidden_dim: Transformer embedding dimension d_model (default: 128).
            n_heads: Number of attention heads (default: 8).
            n_layers: Number of Transformer encoder layers (default: 4).
            dim_feedforward: Hidden dimension of FFN (default: 256).
            dropout: Dropout probability (default: 0.1).
            pooling: Method to pool token sequence into fixed embedding (default: 'masked_mean').
            max_window_size: Maximum sequence length for positional encoding.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers

        # 1. Linear input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 2. Positional Encoding
        self.pos_encoder = SinusoidalPositionalEncoding(hidden_dim, max_len=max_window_size, dropout=dropout)

        # 3. Transformer Encoder Layers with Pre-LayerNorm architecture
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 4. Masked temporal pooling
        self.pooling = MaskedTemporalPooling(method=pooling, d_model=hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode temporal sensor window into a fixed representation vector.

        Args:
            x: Input tensor of shape (B, W, D).
            mask: Optional binary validity mask of shape (B, W) where 1=valid, 0=padded.

        Returns:
            Representation embedding vector h of shape (B, hidden_dim).
        """
        B, W, D = x.shape
        if D != self.input_dim:
            raise ValueError(f"Expected input feature dim {self.input_dim}, got {D}.")

        # 1. Project input to hidden dimension
        h = self.input_proj(x)  # (B, W, hidden_dim)

        # 2. Add Positional Encoding
        h = self.pos_encoder(h)

        # 3. Create attention mask for padding positions (True = ignore in PyTorch)
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = (mask == 0).to(x.device)

        # 4. Pass through Transformer Encoder
        h = self.transformer(h, src_key_padding_mask=key_padding_mask)

        # 5. Masked temporal pooling
        pooled = self.pooling(h, mask=mask)
        pooled = self.output_norm(pooled)

        return pooled


class ProjectionHead(nn.Module):
    """Non-linear projection head mapping representation h -> z for contrastive InfoNCE."""

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        output_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Project representation vector h of shape (B, input_dim) to z of shape (B, output_dim)."""
        return self.net(h)


class SupervisedRULHead(nn.Module):
    """Supervised regression head mapping representation h -> scalar RUL."""

    def __init__(self, input_dim: int = 128, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)


class FailureDNAContrastiveModel(nn.Module):
    """Full self-supervised FailureDNA model integrating Encoder and Projection Head."""

    def __init__(
        self,
        encoder: TemporalTransformerEncoder,
        projection_head: ProjectionHead,
    ):
        super().__init__()
        self.encoder = encoder
        self.projection_head = projection_head

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both representation embedding h and projected latent z."""
        h = self.encoder(x, mask=mask)
        z = self.projection_head(h)
        return h, z

    def get_embedding(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract frozen representation embedding h."""
        with torch.no_grad():
            return self.encoder(x, mask=mask)


class SupervisedTransformerModel(nn.Module):
    """Fully supervised end-to-end benchmark using identical Transformer backbone."""

    def __init__(
        self,
        encoder: TemporalTransformerEncoder,
        head: SupervisedRULHead,
    ):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.encoder(x, mask=mask)
        return self.head(h)
