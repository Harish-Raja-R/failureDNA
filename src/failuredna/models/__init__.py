"""FailureDNA neural architectures, augmentations, loss functions, and baselines."""

from failuredna.models.augmentations import TemporalAugmentor
from failuredna.models.baselines import (
    LinearWindowBaseline,
    OperatingConditionOnlyBaseline,
    RandomForestBaseline,
    RecurrentRULBaseline,
    XGBoostBaseline,
    compute_window_ruls,
    extract_window_summary_features,
)
from failuredna.models.encoder import (
    FailureDNAContrastiveModel,
    ProjectionHead,
    SupervisedRULHead,
    SupervisedTransformerModel,
    TemporalTransformerEncoder,
)
from failuredna.models.loss import NTXentLoss
from failuredna.models.trainer import (
    ContrastiveTrainer,
    SupervisedNeuralTrainer,
    extract_embeddings_from_windows,
)

__all__ = [
    "TemporalAugmentor",
    "NTXentLoss",
    "TemporalTransformerEncoder",
    "ProjectionHead",
    "SupervisedRULHead",
    "FailureDNAContrastiveModel",
    "SupervisedTransformerModel",
    "ContrastiveTrainer",
    "SupervisedNeuralTrainer",
    "extract_embeddings_from_windows",
    "compute_window_ruls",
    "extract_window_summary_features",
    "OperatingConditionOnlyBaseline",
    "LinearWindowBaseline",
    "RandomForestBaseline",
    "XGBoostBaseline",
    "RecurrentRULBaseline",
]
