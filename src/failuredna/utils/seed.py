"""Seed utilities to guarantee strict experimental reproducibility."""

import os
import random
from typing import Optional

import numpy as np


def seed_everything(seed: int = 42, deterministic_torch: bool = True) -> int:
    """Seed all pseudo-random number generators across Python, NumPy, and PyTorch.

    Args:
        seed: Integer random seed (default: 42).
        deterministic_torch: If True and PyTorch is available, enables deterministic
            convolution algorithms and disables benchmark mode.

    Returns:
        The integer seed applied.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed
