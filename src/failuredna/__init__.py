"""FailureDNA: Self-Supervised Discovery of Failure Signatures from Multivariate Industrial Time-Series.

A modular research and engineering framework for self-supervised representation
learning, latent degradation signature discovery, and prognostics on complex industrial telemetry.
"""

__version__ = "0.1.0"
__author__ = "FailureDNA Research Team"

from failuredna.utils.logging import get_logger
from failuredna.utils.seed import seed_everything

__all__ = ["__version__", "get_logger", "seed_everything"]
