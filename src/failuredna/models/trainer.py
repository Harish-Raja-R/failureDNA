"""Training loops, validation, early stopping, and checkpointing for FailureDNA.

Supports:
1. Self-supervised contrastive pretraining with NT-Xent loss.
2. Supervised sequence training (Transformer, LSTM, GRU).
3. Frozen embedding extraction.
4. Structured metric logging and checkpoint serialization.
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from failuredna.evaluation.metrics import compute_regression_metrics
from failuredna.models.augmentations import BatchAugmentationCollate, TemporalAugmentor
from failuredna.models.encoder import FailureDNAContrastiveModel
from failuredna.models.loss import NTXentLoss
from failuredna.utils.logging import get_logger

logger = get_logger(__name__)


class SequenceWindowDataset(Dataset):
    """PyTorch Dataset wrapping temporal windows, validity masks, and optional RUL targets."""

    def __init__(
        self,
        windows: np.ndarray,
        masks: Optional[np.ndarray] = None,
        targets: Optional[np.ndarray] = None,
    ):
        self.windows = windows.astype(np.float32)
        if masks is not None:
            self.masks = masks.astype(np.float32)
        else:
            self.masks = np.ones((len(windows), windows.shape[1]), dtype=np.float32)

        self.targets = targets.astype(np.float32) if targets is not None else None

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if self.targets is not None:
            return self.windows[idx], self.masks[idx], self.targets[idx]
        return self.windows[idx], self.masks[idx]


class ContrastiveTrainer:
    """Trainer for FailureDNA self-supervised temporal contrastive learning."""

    def __init__(
        self,
        model: FailureDNAContrastiveModel,
        augmentor: TemporalAugmentor,
        temperature: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cpu",
        checkpoint_dir: Optional[Path] = None,
    ):
        self.model = model.to(device)
        self.augmentor = augmentor
        self.criterion = NTXentLoss(temperature=temperature)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run single training epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for v1, v2, mask in dataloader:
            v1 = v1.to(self.device)
            v2 = v2.to(self.device)
            mask = mask.to(self.device)

            self.optimizer.zero_grad()
            _, z1 = self.model(v1, mask=mask)
            _, z2 = self.model(v2, mask=mask)

            loss = self.criterion(z1, z2)
            loss.backward()

            # Gradient clipping for Transformer stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        return float(total_loss / max(1, n_batches))

    def evaluate(self, dataloader: DataLoader) -> float:
        """Compute contrastive validation loss on paired validation views."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for v1, v2, mask in dataloader:
                v1 = v1.to(self.device)
                v2 = v2.to(self.device)
                mask = mask.to(self.device)

                _, z1 = self.model(v1, mask=mask)
                _, z2 = self.model(v2, mask=mask)
                loss = self.criterion(z1, z2)

                total_loss += float(loss.item())
                n_batches += 1

        return float(total_loss / max(1, n_batches))

    def fit(
        self,
        train_windows: np.ndarray,
        train_masks: Optional[np.ndarray],
        val_windows: np.ndarray,
        val_masks: Optional[np.ndarray],
        batch_size: int = 64,
        max_epochs: int = 25,
        patience: int = 5,
    ) -> Dict[str, Any]:
        """Execute full training loop with early stopping."""
        train_ds = SequenceWindowDataset(train_windows, train_masks)
        val_ds = SequenceWindowDataset(val_windows, val_masks)

        collate_fn = BatchAugmentationCollate(self.augmentor)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=True)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max_epochs, eta_min=1e-5)

        best_val_loss = float("inf")
        patience_counter = 0
        history: Dict[str, List[float]] = {"epoch": [], "train_loss": [], "val_loss": [], "lr": []}

        start_time = time.perf_counter()
        logger.info(f"Starting FailureDNA Contrastive Pretraining for {max_epochs} epochs (batch_size={batch_size})...")

        for epoch in range(1, max_epochs + 1):
            t_epoch = time.perf_counter()
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)
            current_lr = float(self.optimizer.param_groups[0]["lr"])
            scheduler.step()

            epoch_sec = time.perf_counter() - t_epoch
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["lr"].append(current_lr)

            logger.info(
                f"Epoch {epoch:02d}/{max_epochs:02d} | "
                f"Train NT-Xent: {train_loss:.4f} | Val NT-Xent: {val_loss:.4f} | "
                f"LR: {current_lr:.2e} | Time: {epoch_sec:.2f}s"
            )

            # Checkpoint on validation improvement
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                patience_counter = 0
                if self.checkpoint_dir:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "best_val_loss": best_val_loss,
                            "encoder_state_dict": self.model.encoder.state_dict(),
                        },
                        self.checkpoint_dir / "best_failuredna_model.pt",
                    )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch} (patience={patience}).")
                    break

        total_time_sec = time.perf_counter() - start_time
        logger.info(f"Contrastive pretraining completed in {total_time_sec:.2f}s. Best Val Loss: {best_val_loss:.4f}")

        # Load best model if available
        if self.checkpoint_dir and (self.checkpoint_dir / "best_failuredna_model.pt").exists():
            checkpoint = torch.load(self.checkpoint_dir / "best_failuredna_model.pt", map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded best checkpoint from epoch {checkpoint['epoch']} with val_loss={checkpoint['best_val_loss']:.4f}")

        return {
            "best_val_loss": best_val_loss,
            "total_time_seconds": total_time_sec,
            "epochs_completed": len(history["epoch"]),
            "history": history,
        }


def extract_embeddings_from_windows(
    model: nn.Module,
    windows: np.ndarray,
    masks: Optional[np.ndarray] = None,
    batch_size: int = 128,
    device: str = "cpu",
) -> np.ndarray:
    """Extract frozen sequence embeddings h of shape (N, D_emb) from trained model."""
    model.eval()
    model.to(device)

    dataset = SequenceWindowDataset(windows, masks)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings = []
    with torch.no_grad():
        for batch in loader:
            x, mask = batch[0].to(device), batch[1].to(device)
            if hasattr(model, "encoder"):
                emb = model.encoder(x, mask=mask)
            else:
                emb = model(x, mask=mask)
            embeddings.append(emb.cpu().numpy())

    return np.concatenate(embeddings, axis=0).astype(np.float32)


class SupervisedNeuralTrainer:
    """Trainer for supervised neural sequence models (Transformer, LSTM, GRU)."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cpu",
        checkpoint_dir: Optional[Path] = None,
    ):
        self.model = model.to(device)
        self.criterion = nn.HuberLoss(delta=10.0)  # Robust loss for RUL targets
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int = 64,
        max_epochs: int = 25,
        patience: int = 5,
    ) -> Dict[str, Any]:
        """Train supervised sequence model to predict scalar RUL."""
        train_ds = SequenceWindowDataset(X_train, targets=y_train)
        val_ds = SequenceWindowDataset(X_val, targets=y_val)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        best_val_rmse = float("inf")
        patience_counter = 0
        history: Dict[str, List[float]] = {"epoch": [], "train_loss": [], "val_rmse": [], "val_nasa": []}

        t_start = time.perf_counter()

        for epoch in range(1, max_epochs + 1):
            self.model.train()
            total_loss = 0.0
            n_batches = 0

            for x, mask, y in train_loader:
                x = x.to(self.device)
                mask = mask.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()
                pred = self.model(x, mask=mask)
                loss = self.criterion(pred, y)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += float(loss.item())
                n_batches += 1

            train_loss = total_loss / max(1, n_batches)

            # Evaluate on validation set
            self.model.eval()
            preds = []
            with torch.no_grad():
                for x, mask, _ in val_loader:
                    x = x.to(self.device)
                    mask = mask.to(self.device)
                    p = self.model(x, mask=mask)
                    preds.append(p.cpu().numpy())

            y_pred = np.concatenate(preds)
            metrics = compute_regression_metrics(y_val, y_pred)
            val_rmse = metrics["rmse"]
            val_nasa = metrics["nasa_score"]

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_rmse"].append(val_rmse)
            history["val_nasa"].append(val_nasa)

            if val_rmse < best_val_rmse - 0.05:
                best_val_rmse = val_rmse
                patience_counter = 0
                if self.checkpoint_dir:
                    torch.save(self.model.state_dict(), self.checkpoint_dir / "best_supervised_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        total_sec = time.perf_counter() - t_start
        if self.checkpoint_dir and (self.checkpoint_dir / "best_supervised_model.pt").exists():
            self.model.load_state_dict(torch.load(self.checkpoint_dir / "best_supervised_model.pt", map_location=self.device))

        return {
            "best_val_rmse": best_val_rmse,
            "total_time_seconds": total_sec,
            "history": history,
        }

    def predict(self, X: np.ndarray, masks: Optional[np.ndarray] = None, batch_size: int = 128) -> np.ndarray:
        """Generate RUL predictions."""
        self.model.eval()
        ds = SequenceWindowDataset(X, masks)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        preds = []
        with torch.no_grad():
            for x, mask in loader:
                x = x.to(self.device)
                mask = mask.to(self.device)
                p = self.model(x, mask=mask)
                preds.append(p.cpu().numpy())
        return np.concatenate(preds)
