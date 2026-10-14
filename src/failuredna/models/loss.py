"""Normalized Temperature-scaled Cross Entropy (NT-Xent / InfoNCE) Loss Module.

Implements symmetric contrastive loss between paired augmented views.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """NT-Xent (InfoNCE) loss with temperature scaling for contrastive learning.

    Given a batch of B pairs of projected embeddings (z_1, z_2), computes the
    symmetric contrastive loss treating (z_{1, i}, z_{2, i}) as positive pairs
    and all other 2(B - 1) embeddings as negative examples.
    """

    def __init__(self, temperature: float = 0.1, eps: float = 1e-8):
        """Initialize NTXentLoss.

        Args:
            temperature: Softmax temperature parameter tau > 0 (default: 0.1).
            eps: Small epsilon for numerical stability.
        """
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}.")
        self.temperature = temperature
        self.eps = eps

    def forward(self, z_1: torch.Tensor, z_2: torch.Tensor) -> torch.Tensor:
        """Compute InfoNCE contrastive loss across a batch.

        Args:
            z_1: Projected embeddings from view 1 of shape (B, D_proj).
            z_2: Projected embeddings from view 2 of shape (B, D_proj).

        Returns:
            Scalar contrastive loss tensor with finite gradients.
        """
        if z_1.shape != z_2.shape:
            raise ValueError(f"Shape mismatch: z_1 {z_1.shape} vs z_2 {z_2.shape}.")

        batch_size = z_1.shape[0]
        if batch_size < 2:
            raise ValueError(f"Batch size must be >= 2 for contrastive negatives; got {batch_size}.")

        # 1. L2 Normalize embeddings onto unit hypersphere
        z_1_norm = F.normalize(z_1, p=2, dim=1, eps=self.eps)
        z_2_norm = F.normalize(z_2, p=2, dim=1, eps=self.eps)

        # 2. Concatenate views to form [2B, D_proj]
        z_all = torch.cat([z_1_norm, z_2_norm], dim=0)  # Shape (2B, D)

        # 3. Compute pairwise cosine similarity matrix scaled by temperature
        # S_ij = (z_i . z_j) / tau
        sim_matrix = torch.matmul(z_all, z_all.T) / self.temperature  # Shape (2B, 2B)

        # 4. Mask out self-similarity (diagonal entries)
        diag_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_all.device)
        sim_matrix.masked_fill_(diag_mask, -1e9)

        # 5. Define positive targets:
        # For index i in [0..B-1], positive is i + B
        # For index i in [B..2B-1], positive is i - B
        targets = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z_all.device),
            torch.arange(0, batch_size, device=z_all.device),
        ])

        # 6. Cross Entropy Loss over the 2B rows
        loss = F.cross_entropy(sim_matrix, targets)

        return loss
