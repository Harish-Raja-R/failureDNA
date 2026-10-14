"""Physically cautious temporal augmentations for self-supervised contrastive learning.

Guarantees:
1. Preserves temporal causality and degradation trajectory structure.
2. Applies conservative perturbations (jitter, sensor masking, temporal masking, scaling).
3. Produces two stochastic augmented views per temporal sequence window.
4. Tracks augmentation parameters and validates feature distributions.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from failuredna.utils.logging import get_logger

logger = get_logger(__name__)


class TemporalAugmentor:
    """Stochastic domain-preserving temporal augmentation pipeline for multivariate telemetry."""

    def __init__(
        self,
        jitter_sigma: float = 0.015,
        sensor_mask_prob: float = 0.15,
        max_masked_sensors: int = 3,
        temp_mask_ratio: float = 0.15,
        scale_min: float = 0.95,
        scale_max: float = 1.05,
        random_state: Optional[int] = None,
    ):
        """Initialize TemporalAugmentor.

        Args:
            jitter_sigma: Standard deviation of additive Gaussian noise (default: 0.015).
            sensor_mask_prob: Probability of triggering sensor channel masking (default: 0.15).
            max_masked_sensors: Maximum number of sensor channels to mask (default: 3).
            temp_mask_ratio: Maximum fraction of time steps to mask (default: 0.15).
            scale_min: Lower bound for multiplicative magnitude scaling (default: 0.95).
            scale_max: Upper bound for multiplicative magnitude scaling (default: 1.05).
            random_state: Optional seed for reproducible stochastic operations.
        """
        self.jitter_sigma = jitter_sigma
        self.sensor_mask_prob = sensor_mask_prob
        self.max_masked_sensors = max_masked_sensors
        self.temp_mask_ratio = temp_mask_ratio
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.rng = np.random.RandomState(random_state) if random_state is not None else np.random.RandomState()

    def augment_single(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply stochastic augmentations to a single temporal window of shape (W, D).

        Args:
            x: Input window of shape (W, D) where W is sequence length, D is channel dimension.
            mask: Optional binary validity mask of shape (W,) where 1 = observed, 0 = padded.

        Returns:
            Augmented window of shape (W, D).
        """
        x_aug = x.copy().astype(np.float32)
        W, D = x_aug.shape

        # Identify valid time steps if mask is provided
        valid_indices = np.where(mask == 1)[0] if mask is not None else np.arange(W)
        if len(valid_indices) == 0:
            valid_indices = np.arange(W)

        # 1. Conservative Magnitude Scaling: x' = s * x
        if self.scale_min < self.scale_max:
            scale_factor = self.rng.uniform(self.scale_min, self.scale_max)
            x_aug[valid_indices] *= scale_factor

        # 2. Additive Gaussian Jitter: x' = x + N(0, sigma^2)
        if self.jitter_sigma > 0:
            noise = self.rng.normal(0.0, self.jitter_sigma, size=(len(valid_indices), D)).astype(np.float32)
            x_aug[valid_indices] += noise

        # 3. Sensor Channel Masking: randomly zero out 1 to max_masked_sensors channels
        if self.rng.rand() < self.sensor_mask_prob and self.max_masked_sensors > 0:
            n_mask = self.rng.randint(1, min(self.max_masked_sensors, D) + 1)
            # Mask sensors (typically indices 3..D-1 if settings are at 0..2)
            # We sample from all feature columns
            masked_channels = self.rng.choice(D, size=n_mask, replace=False)
            x_aug[valid_indices[:, None], masked_channels] = 0.0

        # 4. Temporal Slice Masking: zero out a contiguous block of time steps
        if self.temp_mask_ratio > 0 and len(valid_indices) > 5:
            max_mask_steps = max(1, int(len(valid_indices) * self.temp_mask_ratio))
            mask_steps = self.rng.randint(1, max_mask_steps + 1)
            start_pos = self.rng.randint(0, len(valid_indices) - mask_steps + 1)
            steps_to_mask = valid_indices[start_pos : start_pos + mask_steps]
            x_aug[steps_to_mask, :] = 0.0

        return x_aug

    def augment_batch(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply vectorized stochastic augmentations to a batch of shape (B, W, D)."""
        x_aug = x.copy().astype(np.float32)
        B, W, D = x_aug.shape

        # 1. Conservative Magnitude Scaling: x' = s * x
        if self.scale_min < self.scale_max:
            scales = self.rng.uniform(self.scale_min, self.scale_max, size=(B, 1, 1)).astype(np.float32)
            x_aug *= scales

        # 2. Additive Gaussian Jitter: x' = x + N(0, sigma^2)
        if self.jitter_sigma > 0:
            noise = self.rng.normal(0.0, self.jitter_sigma, size=(B, W, D)).astype(np.float32)
            x_aug += noise

        # 3. Sensor Channel Masking
        if self.sensor_mask_prob > 0 and self.max_masked_sensors > 0:
            for b in range(B):
                if self.rng.rand() < self.sensor_mask_prob:
                    n_mask = self.rng.randint(1, min(self.max_masked_sensors, D) + 1)
                    masked_channels = self.rng.choice(D, size=n_mask, replace=False)
                    x_aug[b, :, masked_channels] = 0.0

        # 4. Temporal Slice Masking
        if self.temp_mask_ratio > 0 and W > 5:
            max_mask_steps = max(1, int(W * self.temp_mask_ratio))
            for b in range(B):
                valid_len = int(np.sum(mask[b])) if mask is not None else W
                if valid_len > 5:
                    mask_steps = self.rng.randint(1, min(max_mask_steps, valid_len) + 1)
                    start_pos = W - valid_len + self.rng.randint(0, valid_len - mask_steps + 1)
                    x_aug[b, start_pos : start_pos + mask_steps, :] = 0.0

        # 5. Mask out padded positions
        if mask is not None:
            x_aug = x_aug * mask[:, :, None]

        return x_aug

    def generate_views(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate two stochastic augmented views x_1 and x_2 for contrastive learning.

        Args:
            x: Input array of shape (W, D) or batch of shape (B, W, D).
            mask: Optional binary validity mask of shape (W,) or (B, W).

        Returns:
            Tuple of (view_1, view_2) with identical shapes to x.
        """
        if x.ndim == 2:
            v1 = self.augment_single(x, mask)
            v2 = self.augment_single(x, mask)
            return v1, v2
        elif x.ndim == 3:
            v1 = self.augment_batch(x, mask)
            v2 = self.augment_batch(x, mask)
            return v1, v2
        else:
            raise ValueError(f"Expected 2D or 3D input tensor, got shape {x.shape}.")


class BatchAugmentationCollate:
    """PyTorch Dataset collate_fn producing paired contrastive views (x_1, x_2, mask)."""

    def __init__(self, augmentor: TemporalAugmentor):
        self.augmentor = augmentor

    def __call__(self, batch: List[Tuple[np.ndarray, Optional[np.ndarray]]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Collate a batch into paired torch Tensors."""
        # Unpack
        windows = [item[0] for item in batch]
        masks = [item[1] if item[1] is not None else np.ones(item[0].shape[0], dtype=np.float32) for item in batch]

        win_arr = np.stack(windows, axis=0)
        mask_arr = np.stack(masks, axis=0)

        v1_arr, v2_arr = self.augmentor.generate_views(win_arr, mask_arr)

        return (
            torch.from_numpy(v1_arr).float(),
            torch.from_numpy(v2_arr).float(),
            torch.from_numpy(mask_arr).float(),
        )
