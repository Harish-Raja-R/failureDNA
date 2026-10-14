"""FailureDNA Phase 3: Self-Supervised Temporal Representation Learning Experiment Runner.

Executes:
1. Leakage-free dataset splitting (174 train, 37 val, 38 internal test).
2. Supervised & Non-Neural Baselines (OLS/Ridge, RF, XGBoost, LSTM, Supervised Transformer).
3. Operating-condition confounding comparison (Pooled vs Regime-aware vs Explicit Conditioning).
4. FailureDNA NT-Xent Contrastive Pretraining.
5. Downstream Prognostic Probing (Ridge, RF, XGBoost on frozen embeddings).
6. Randomly initialized encoder negative control.
7. Controlled Ablations (W=50 vs W=30, Multi-seed).
8. Metric recording, NASA score evaluation, publication figures, and structured JSON/CSV logging.
"""

import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
import torch
import torch.nn as nn

from failuredna.data.loader import CYCLE_COL, ID_COL, load_cmapss_dataset
from failuredna.data.preprocessing import CMAPSSPreprocessor, RegimeAwarePreprocessor
from failuredna.data.split import split_engines_by_unit
from failuredna.data.windowing import WindowGenerator
from failuredna.evaluation.diagnostics import evaluate_embedding_diagnostics
from failuredna.evaluation.metrics import compute_regression_metrics, nasa_scoring_function
from failuredna.evaluation.probes import run_all_embedding_probes
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
from failuredna.models.trainer import (
    ContrastiveTrainer,
    SupervisedNeuralTrainer,
    extract_embeddings_from_windows,
)
from failuredna.utils.logging import get_logger
from failuredna.utils.seed import seed_everything

logger = get_logger("failuredna.phase3")


def prepare_preprocessed_partitions(
    train_raw: pd.DataFrame,
    val_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    strategy: str = "pooled",
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """Fit preprocessing strictly on train partition and transform val and internal test."""
    if strategy == "pooled":
        preprocessor = CMAPSSPreprocessor(scaler_type="robust", include_settings=True)
        train_proc = preprocessor.fit_transform(train_raw)
        val_proc = preprocessor.transform(val_raw)
        test_proc = preprocessor.transform(test_raw)
        feature_cols = preprocessor.get_feature_names()
    elif strategy == "regime_aware":
        preprocessor = RegimeAwarePreprocessor(
            n_regimes=6, include_settings=True, include_regime_onehot=False, seed=seed
        )
        train_proc = preprocessor.fit_transform(train_raw)
        val_proc = preprocessor.transform(val_raw)
        test_proc = preprocessor.transform(test_raw)
        feature_cols = preprocessor.get_feature_names()
    elif strategy == "regime_conditioned":
        preprocessor = RegimeAwarePreprocessor(
            n_regimes=6, include_settings=True, include_regime_onehot=True, seed=seed
        )
        train_proc = preprocessor.fit_transform(train_raw)
        val_proc = preprocessor.transform(val_raw)
        test_proc = preprocessor.transform(test_raw)
        feature_cols = preprocessor.get_feature_names()
    else:
        raise ValueError(f"Unknown preprocessing strategy '{strategy}'.")

    return train_proc, val_proc, test_proc, feature_cols


def extract_windows_and_targets(
    train_proc: pd.DataFrame,
    val_proc: pd.DataFrame,
    test_proc: pd.DataFrame,
    train_raw: pd.DataFrame,
    val_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    feature_cols: List[str],
    window_size: int = 50,
    stride: int = 1,
    cap_rul: float = 125.0,
) -> Dict[str, Any]:
    """Generate windows and aligned target RULs across partitions."""
    win_gen = WindowGenerator(
        window_size=window_size,
        stride=stride,
        feature_cols=feature_cols,
        padding_mode="edge",
    )

    X_train, meta_train, _ = win_gen.create_windows(train_proc)
    X_val, meta_val, _ = win_gen.create_windows(val_proc)
    X_test, meta_test, _ = win_gen.create_windows(test_proc)

    y_train = compute_window_ruls(meta_train, train_raw, cap_rul=cap_rul)
    y_val = compute_window_ruls(meta_val, val_raw, cap_rul=cap_rul)
    y_test = compute_window_ruls(meta_test, test_raw, cap_rul=cap_rul)

    # Masks: 0 for padded positions, 1 for valid
    mask_train = np.ones((len(meta_train), window_size), dtype=np.float32)
    for i, pad_len in enumerate(meta_train["pad_length"]):
        if pad_len > 0:
            mask_train[i, :pad_len] = 0.0

    mask_val = np.ones((len(meta_val), window_size), dtype=np.float32)
    for i, pad_len in enumerate(meta_val["pad_length"]):
        if pad_len > 0:
            mask_val[i, :pad_len] = 0.0

    mask_test = np.ones((len(meta_test), window_size), dtype=np.float32)
    for i, pad_len in enumerate(meta_test["pad_length"]):
        if pad_len > 0:
            mask_test[i, :pad_len] = 0.0

    return {
        "X_train": X_train,
        "y_train": y_train,
        "mask_train": mask_train,
        "meta_train": meta_train,
        "X_val": X_val,
        "y_val": y_val,
        "mask_val": mask_val,
        "meta_val": meta_val,
        "X_test": X_test,
        "y_test": y_test,
        "mask_test": mask_test,
        "meta_test": meta_test,
    }


def run_baselines_suite(
    windows_dict: Dict[str, Any],
    seed: int = 42,
    smoke_test: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Execute all supervised and non-neural baselines."""
    results = {}
    X_train, y_train = windows_dict["X_train"], windows_dict["y_train"]
    X_val, y_val = windows_dict["X_val"], windows_dict["y_val"]
    X_test, y_test = windows_dict["X_test"], windows_dict["y_test"]

    # 1. Operating Condition Only Baseline
    logger.info("Evaluating Baseline 1: Operating-Condition-Only Baseline...")
    t0 = time.perf_counter()
    b_ops = OperatingConditionOnlyBaseline().fit(X_train, y_train)
    fit_ops = time.perf_counter() - t0
    y_pred_val_ops = b_ops.predict(X_val)
    y_pred_test_ops = b_ops.predict(X_test)
    results["operating_condition_only"] = {
        "name": "Operating Condition Only Baseline",
        "fit_time_seconds": fit_ops,
        "val_metrics": compute_regression_metrics(y_val, y_pred_val_ops),
        "internal_test_metrics": compute_regression_metrics(y_test, y_pred_test_ops),
    }

    # Extract summary features for tabular baselines
    logger.info("Extracting summary window statistics for tabular baselines...")
    X_train_tab = extract_window_summary_features(X_train)
    X_val_tab = extract_window_summary_features(X_val)
    X_test_tab = extract_window_summary_features(X_test)

    # 2. Linear / Ridge Baseline
    logger.info("Evaluating Baseline 2: Linear / Ridge Window Baseline...")
    t0 = time.perf_counter()
    b_lin = LinearWindowBaseline(alpha=1.0).fit(X_train_tab, y_train)
    fit_lin = time.perf_counter() - t0
    results["linear_ridge"] = {
        "name": "Linear / Ridge Regression (Window Statistics)",
        "fit_time_seconds": fit_lin,
        "val_metrics": compute_regression_metrics(y_val, b_lin.predict(X_val_tab)),
        "internal_test_metrics": compute_regression_metrics(y_test, b_lin.predict(X_test_tab)),
    }

    # 3. Random Forest Baseline
    logger.info("Evaluating Baseline 3: Random Forest Baseline...")
    t0 = time.perf_counter()
    n_trees = 20 if smoke_test else 100
    b_rf = RandomForestBaseline(n_estimators=n_trees, max_depth=10, random_state=seed).fit(X_train_tab, y_train)
    fit_rf = time.perf_counter() - t0
    results["random_forest"] = {
        "name": "Random Forest (Window Statistics)",
        "fit_time_seconds": fit_rf,
        "val_metrics": compute_regression_metrics(y_val, b_rf.predict(X_val_tab)),
        "internal_test_metrics": compute_regression_metrics(y_test, b_rf.predict(X_test_tab)),
    }

    # 4. XGBoost Baseline
    logger.info("Evaluating Baseline 4: XGBoost Baseline...")
    t0 = time.perf_counter()
    n_xgb = 20 if smoke_test else 150
    b_xgb = XGBoostBaseline(n_estimators=n_xgb, max_depth=6, learning_rate=0.05, random_state=seed).fit(
        X_train_tab, y_train, X_val_tab, y_val
    )
    fit_xgb = time.perf_counter() - t0
    results["xgboost"] = {
        "name": "XGBoost (Window Statistics)",
        "fit_time_seconds": fit_xgb,
        "val_metrics": compute_regression_metrics(y_val, b_xgb.predict(X_val_tab)),
        "internal_test_metrics": compute_regression_metrics(y_test, b_xgb.predict(X_test_tab)),
    }

    # 5. Recurrent Neural Baseline (LSTM)
    logger.info("Evaluating Baseline 5: Supervised LSTM Baseline...")
    input_dim = X_train.shape[2]
    lstm_model = RecurrentRULBaseline(input_dim=input_dim, hidden_dim=64, num_layers=2, cell_type="lstm")
    lstm_trainer = SupervisedNeuralTrainer(lstm_model, learning_rate=1e-3, device="cpu")
    epochs_rnn = 3 if smoke_test else 15
    lstm_res = lstm_trainer.fit(X_train, y_train, X_val, y_val, batch_size=128, max_epochs=epochs_rnn, patience=4)
    y_pred_val_lstm = lstm_trainer.predict(X_val, windows_dict["mask_val"])
    y_pred_test_lstm = lstm_trainer.predict(X_test, windows_dict["mask_test"])
    results["supervised_lstm"] = {
        "name": "Supervised LSTM Sequence Model",
        "fit_time_seconds": lstm_res["total_time_seconds"],
        "val_metrics": compute_regression_metrics(y_val, y_pred_val_lstm),
        "internal_test_metrics": compute_regression_metrics(y_test, y_pred_test_lstm),
    }

    # 6. Supervised Transformer Baseline
    logger.info("Evaluating Baseline 6: Supervised Transformer Baseline...")
    t_enc = TemporalTransformerEncoder(input_dim=input_dim, hidden_dim=64, n_heads=4, n_layers=2)
    s_head = SupervisedRULHead(input_dim=64, hidden_dim=32)
    sup_tf_model = SupervisedTransformerModel(t_enc, s_head)
    tf_trainer = SupervisedNeuralTrainer(sup_tf_model, learning_rate=1e-3, device="cpu")
    epochs_tf = 3 if smoke_test else 15
    tf_res = tf_trainer.fit(X_train, y_train, X_val, y_val, batch_size=128, max_epochs=epochs_tf, patience=4)
    y_pred_val_tf = tf_trainer.predict(X_val, windows_dict["mask_val"])
    y_pred_test_tf = tf_trainer.predict(X_test, windows_dict["mask_test"])
    results["supervised_transformer"] = {
        "name": "Supervised Transformer Backbone",
        "fit_time_seconds": tf_res["total_time_seconds"],
        "val_metrics": compute_regression_metrics(y_val, y_pred_val_tf),
        "internal_test_metrics": compute_regression_metrics(y_test, y_pred_test_tf),
    }

    return results


def train_and_evaluate_failuredna(
    windows_dict: Dict[str, Any],
    config_name: str,
    hidden_dim: int = 128,
    n_heads: int = 8,
    n_layers: int = 4,
    dim_ff: int = 256,
    max_epochs: int = 20,
    batch_size: int = 128,
    temperature: float = 0.1,
    seed: int = 42,
    checkpoints_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train FailureDNA self-supervised encoder and evaluate downstream probes."""
    X_train, y_train = windows_dict["X_train"], windows_dict["y_train"]
    mask_train = windows_dict["mask_train"]
    X_val, y_val = windows_dict["X_val"], windows_dict["y_val"]
    mask_val = windows_dict["mask_val"]
    X_test, y_test = windows_dict["X_test"], windows_dict["y_test"]
    mask_test = windows_dict["mask_test"]

    input_dim = X_train.shape[2]
    ckpt_dir = checkpoints_dir / config_name if checkpoints_dir else None

    # Instantiate FailureDNA Model
    encoder = TemporalTransformerEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_feedforward=dim_ff,
        dropout=0.1,
        pooling="masked_mean",
    )
    proj_head = ProjectionHead(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=64)
    model = FailureDNAContrastiveModel(encoder, proj_head)

    # Augmentor
    augmentor = TemporalAugmentor(
        jitter_sigma=0.015,
        sensor_mask_prob=0.15,
        max_masked_sensors=3,
        temp_mask_ratio=0.15,
        scale_min=0.95,
        scale_max=1.05,
        random_state=seed,
    )

    trainer = ContrastiveTrainer(
        model=model,
        augmentor=augmentor,
        temperature=temperature,
        learning_rate=1e-3,
        weight_decay=1e-4,
        device="cpu",
        checkpoint_dir=ckpt_dir,
    )

    train_res = trainer.fit(
        train_windows=X_train,
        train_masks=mask_train,
        val_windows=X_val,
        val_masks=mask_val,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=5,
    )

    # Extract frozen embeddings
    logger.info(f"Extracting frozen representations for {config_name}...")
    emb_train = extract_embeddings_from_windows(model, X_train, mask_train, batch_size=256)
    emb_val = extract_embeddings_from_windows(model, X_val, mask_val, batch_size=256)
    emb_test = extract_embeddings_from_windows(model, X_test, mask_test, batch_size=256)

    # Evaluate Downstream Probes
    probe_results = run_all_embedding_probes(emb_train, y_train, emb_val, y_val, emb_test, y_test, seed=seed)

    # Evaluate Embedding Diagnostics
    diagnostics = evaluate_embedding_diagnostics(emb_val, y_val, n_clusters=4, random_state=seed)

    return {
        "config_name": config_name,
        "pretraining": train_res,
        "probes": probe_results,
        "diagnostics": diagnostics,
        "sample_embeddings_val": emb_val[:200].tolist(),  # Save small sample for visualization
        "sample_y_val": y_val[:200].tolist(),
    }


def evaluate_random_encoder_control(
    windows_dict: Dict[str, Any],
    hidden_dim: int = 128,
    seed: int = 42,
) -> Dict[str, Any]:
    """Negative control: evaluate random untrained encoder representations."""
    X_train, y_train = windows_dict["X_train"], windows_dict["y_train"]
    mask_train = windows_dict["mask_train"]
    X_val, y_val = windows_dict["X_val"], windows_dict["y_val"]
    mask_val = windows_dict["mask_val"]
    X_test, y_test = windows_dict["X_test"], windows_dict["y_test"]
    mask_test = windows_dict["mask_test"]

    input_dim = X_train.shape[2]
    torch.manual_seed(seed)
    random_encoder = TemporalTransformerEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        n_heads=4,
        n_layers=2,
        pooling="masked_mean",
    )

    emb_train = extract_embeddings_from_windows(random_encoder, X_train, mask_train, batch_size=256)
    emb_val = extract_embeddings_from_windows(random_encoder, X_val, mask_val, batch_size=256)
    emb_test = extract_embeddings_from_windows(random_encoder, X_test, mask_test, batch_size=256)

    probes = run_all_embedding_probes(emb_train, y_train, emb_val, y_val, emb_test, y_test, seed=seed)
    diagnostics = evaluate_embedding_diagnostics(emb_val, y_val, n_clusters=4, random_state=seed)

    return {
        "config_name": "random_untrained_encoder",
        "probes": probes,
        "diagnostics": diagnostics,
    }


def plot_phase3_figures(
    experiment_results: Dict[str, Any],
    figures_dir: Path,
):
    """Render publication-grade comparison and diagnostic figures."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.0)

    # --- Figure 10: Prognostic Benchmark Comparison ---
    models = []
    rmse_vals = []
    nasa_vals = []
    category = []

    # Baselines
    for k, v in experiment_results["baselines"].items():
        models.append(v["name"])
        rmse_vals.append(v["val_metrics"]["rmse"])
        nasa_vals.append(v["val_metrics"]["nasa_score"])
        category.append("Supervised Baseline")

    # FailureDNA Probes
    fd_probes = experiment_results["failuredna_pooled"]["probes"]
    models.append("FailureDNA + Ridge Probe")
    rmse_vals.append(fd_probes["ridge"]["val_metrics"]["rmse"])
    nasa_vals.append(fd_probes["ridge"]["val_metrics"]["nasa_score"])
    category.append("FailureDNA (Frozen SSL)")

    models.append("FailureDNA + Random Forest Probe")
    rmse_vals.append(fd_probes["random_forest"]["val_metrics"]["rmse"])
    nasa_vals.append(fd_probes["random_forest"]["val_metrics"]["nasa_score"])
    category.append("FailureDNA (Frozen SSL)")

    models.append("FailureDNA + XGBoost Probe")
    rmse_vals.append(fd_probes["xgboost"]["val_metrics"]["rmse"])
    nasa_vals.append(fd_probes["xgboost"]["val_metrics"]["nasa_score"])
    category.append("FailureDNA (Frozen SSL)")

    # Negative Control
    rnd_probe = experiment_results["random_encoder"]["probes"]["xgboost"]
    models.append("Random Encoder + XGBoost")
    rmse_vals.append(rnd_probe["val_metrics"]["rmse"])
    nasa_vals.append(rnd_probe["val_metrics"]["nasa_score"])
    category.append("Negative Control")

    df_comp = pd.DataFrame({
        "Model": models,
        "Validation RMSE": rmse_vals,
        "NASA Score": nasa_vals,
        "Category": category,
    })

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Panel A: RMSE
    sns.barplot(
        data=df_comp,
        x="Validation RMSE",
        y="Model",
        hue="Category",
        dodge=False,
        palette="Blues_r",
        ax=axes[0],
    )
    axes[0].set_title("Panel A: Validation RMSE (Lower is Better)", fontweight="bold")
    axes[0].set_xlabel("RMSE (Cycles)")

    # Panel B: NASA Asymmetric Score
    sns.barplot(
        data=df_comp,
        x="NASA Score",
        y="Model",
        hue="Category",
        dodge=False,
        palette="Reds_r",
        ax=axes[1],
    )
    axes[1].set_title("Panel B: NASA Asymmetric Score (Lower is Better)", fontweight="bold")
    axes[1].set_xlabel("NASA Score (S)")

    plt.suptitle("C-MAPSS FD004: Prognostic RUL Benchmarking (Supervised Baselines vs FailureDNA Representations)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig_path = figures_dir / "10_prognostic_performance_comparison.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    logger.info(f"Saved figure: {fig_path.name}")

    # --- Figure 11: Latent Representation Geometry & Singular Values ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: PCA Projection
    emb_sample = np.array(experiment_results["failuredna_pooled"]["sample_embeddings_val"])
    y_sample = np.array(experiment_results["failuredna_pooled"]["sample_y_val"])

    pca = PCA(n_components=2)
    coords = pca.fit_transform(emb_sample)
    scatter = axes[0].scatter(
        coords[:, 0],
        coords[:, 1],
        c=y_sample,
        cmap="viridis_r",
        alpha=0.85,
        s=30,
        edgecolors="none",
    )
    cbar = plt.colorbar(scatter, ax=axes[0])
    cbar.set_label("True RUL (Cycles)")
    axes[0].set_title(f"Panel A: Latent PCA Space Colored by RUL (EV={pca.explained_variance_ratio_.sum()*100:.1f}%)", fontweight="bold")
    axes[0].set_xlabel("Principal Component 1")
    axes[0].set_ylabel("Principal Component 2")

    # Panel B: Singular Value Spectrum
    diag = experiment_results["failuredna_pooled"]["diagnostics"]
    rnd_diag = experiment_results["random_encoder"]["diagnostics"]

    top_vars = [diag["pca_top1_explained_variance"], diag["pca_top3_explained_variance"]]
    rnd_top_vars = [rnd_diag["pca_top1_explained_variance"], rnd_diag["pca_top3_explained_variance"]]

    x_labels = ["Top 1 PC", "Top 3 PCs"]
    x_pos = np.arange(len(x_labels))
    width = 0.35

    axes[1].bar(x_pos - width / 2, top_vars, width, label="FailureDNA Pretrained", color="#1f77b4")
    axes[1].bar(x_pos + width / 2, rnd_top_vars, width, label="Random Encoder", color="#7f7f7f")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(x_labels)
    axes[1].set_ylabel("Explained Variance Ratio")
    axes[1].set_title("Panel B: Dimensional Spectrum (Guarding Against Collapse)", fontweight="bold")
    axes[1].legend()

    plt.suptitle("FailureDNA Latent Geometry & Dimensional Health Diagnostics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig_path = figures_dir / "11_latent_geometry_and_spectrum.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    logger.info(f"Saved figure: {fig_path.name}")

    # --- Figure 12: Preprocessing Ablation Comparison ---
    abl_models = ["Pooled Robust (A)", "Regime-Aware (B)", "Regime-Conditioned (C)"]
    abl_rmse = [
        experiment_results["failuredna_pooled"]["probes"]["xgboost"]["val_metrics"]["rmse"],
        experiment_results["failuredna_regime_aware"]["probes"]["xgboost"]["val_metrics"]["rmse"],
        experiment_results["failuredna_regime_conditioned"]["probes"]["xgboost"]["val_metrics"]["rmse"],
    ]
    abl_spearman = [
        abs(experiment_results["failuredna_pooled"]["diagnostics"]["distance_to_healthy_spearman_r"]),
        abs(experiment_results["failuredna_regime_aware"]["diagnostics"]["distance_to_healthy_spearman_r"]),
        abs(experiment_results["failuredna_regime_conditioned"]["diagnostics"]["distance_to_healthy_spearman_r"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.barplot(x=abl_models, y=abl_rmse, palette="Blues_d", ax=axes[0])
    axes[0].set_title("Panel A: Downstream RUL RMSE (XGBoost Probe)", fontweight="bold")
    axes[0].set_ylabel("Validation RMSE (Cycles)")

    sns.barplot(x=abl_models, y=abl_spearman, palette="Greens_d", ax=axes[1])
    axes[1].set_title("Panel B: Distance-to-Baseline Spearman |r_s| with RUL", fontweight="bold")
    axes[1].set_ylabel("Absolute Spearman Rank Correlation")

    plt.suptitle("C-MAPSS FD004: Preprocessing Strategy Ablation (Operating Condition Confounding)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig_path = figures_dir / "12_preprocessing_ablation_comparison.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    logger.info(f"Saved figure: {fig_path.name}")


def main():
    parser = argparse.ArgumentParser(description="FailureDNA Phase 3 Experiments")
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--output-dir", type=str, default="reports/results")
    parser.add_argument("--figures-dir", type=str, default="reports/figures")
    parser.add_argument("--checkpoints-dir", type=str, default="models/checkpoints")
    parser.add_argument("--smoke-test", action="store_true", help="Run rapid smoke test for resource verification")
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir)
    checkpoints_dir = Path(args.checkpoints_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("======================================================================")
    logger.info("             FailureDNA Phase 3: Self-Supervised Learning             ")
    logger.info("======================================================================")

    # 1. Load Raw C-MAPSS FD004
    dataset = load_cmapss_dataset(args.data_dir, variant="FD004")
    raw_train = dataset["train"]

    # 2. Engine-level Disjoint Partitioning (70% / 15% / 15%)
    train_raw, val_raw, itest_raw, split_info = split_engines_by_unit(
        raw_train, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=args.seed
    )
    logger.info(
        f"Verified Engine Partition: {len(train_raw[ID_COL].unique())} train, "
        f"{len(val_raw[ID_COL].unique())} val, {len(itest_raw[ID_COL].unique())} internal-test."
    )

    if args.smoke_test:
        logger.info("SMOKE TEST MODE ENABLED: Subsampling engines for rapid resource verification...")
        train_units = train_raw[ID_COL].unique()[:20]
        val_units = val_raw[ID_COL].unique()[:5]
        itest_units = itest_raw[ID_COL].unique()[:5]
        train_raw = train_raw[train_raw[ID_COL].isin(train_units)].reset_index(drop=True)
        val_raw = val_raw[val_raw[ID_COL].isin(val_units)].reset_index(drop=True)
        itest_raw = itest_raw[itest_raw[ID_COL].isin(itest_units)].reset_index(drop=True)

    # 3. Controlled Preprocessing Variants
    # Variant A: Conventional Pooled Robust Scaling
    logger.info("Fitting Preprocessing Variant A (Pooled Robust Scaling)...")
    tr_p, val_p, itest_p, cols_p = prepare_preprocessed_partitions(
        train_raw, val_raw, itest_raw, strategy="pooled", seed=args.seed
    )
    win_pooled = extract_windows_and_targets(
        tr_p, val_p, itest_p, train_raw, val_raw, itest_raw, cols_p, window_size=args.window_size
    )

    # Variant B: Regime-Aware Preprocessing
    logger.info("Fitting Preprocessing Variant B (Regime-Aware Standardization)...")
    tr_ra, val_ra, itest_ra, cols_ra = prepare_preprocessed_partitions(
        train_raw, val_raw, itest_raw, strategy="regime_aware", seed=args.seed
    )
    win_regime_aware = extract_windows_and_targets(
        tr_ra, val_ra, itest_ra, train_raw, val_raw, itest_raw, cols_ra, window_size=args.window_size
    )

    # Variant C: Explicit Regime Conditioning
    logger.info("Fitting Preprocessing Variant C (Explicit Regime Conditioning)...")
    tr_rc, val_rc, itest_rc, cols_rc = prepare_preprocessed_partitions(
        train_raw, val_raw, itest_raw, strategy="regime_conditioned", seed=args.seed
    )
    win_regime_cond = extract_windows_and_targets(
        tr_rc, val_rc, itest_rc, train_raw, val_raw, itest_raw, cols_rc, window_size=args.window_size
    )

    # 4. Supervised and Non-Neural Baselines
    baselines_results = run_baselines_suite(win_pooled, seed=args.seed, smoke_test=args.smoke_test)

    # 5. Primary FailureDNA SSL Pretraining (Variant A: Pooled)
    logger.info("Training Primary FailureDNA Contrastive Model (Variant A)...")
    max_epochs = 3 if args.smoke_test else 15
    fd_pooled_results = train_and_evaluate_failuredna(
        win_pooled,
        config_name="failuredna_pooled",
        hidden_dim=128,
        n_heads=8,
        n_layers=4,
        max_epochs=max_epochs,
        batch_size=128,
        temperature=0.1,
        seed=args.seed,
        checkpoints_dir=checkpoints_dir,
    )

    # 6. FailureDNA SSL Pretraining (Variant B: Regime-Aware)
    logger.info("Training FailureDNA Contrastive Model (Variant B: Regime-Aware)...")
    fd_regime_results = train_and_evaluate_failuredna(
        win_regime_aware,
        config_name="failuredna_regime_aware",
        hidden_dim=128,
        n_heads=8,
        n_layers=4,
        max_epochs=max_epochs,
        batch_size=128,
        temperature=0.1,
        seed=args.seed,
        checkpoints_dir=checkpoints_dir,
    )

    # 7. FailureDNA SSL Pretraining (Variant C: Explicit Conditioning)
    logger.info("Training FailureDNA Contrastive Model (Variant C: Regime-Conditioned)...")
    fd_cond_results = train_and_evaluate_failuredna(
        win_regime_cond,
        config_name="failuredna_regime_conditioned",
        hidden_dim=128,
        n_heads=8,
        n_layers=4,
        max_epochs=max_epochs,
        batch_size=128,
        temperature=0.1,
        seed=args.seed,
        checkpoints_dir=checkpoints_dir,
    )

    # 8. Negative Control: Random Untrained Encoder
    logger.info("Evaluating Negative Control: Random Untrained Encoder...")
    random_encoder_results = evaluate_random_encoder_control(win_pooled, hidden_dim=128, seed=args.seed)

    # 9. Ablation Study: Window Size W=30 vs W=50
    logger.info("Running Ablation Study: Window Size W=30...")
    win_w30 = extract_windows_and_targets(
        tr_p, val_p, itest_p, train_raw, val_raw, itest_raw, cols_p, window_size=30
    )
    fd_w30_results = train_and_evaluate_failuredna(
        win_w30,
        config_name="failuredna_w30",
        hidden_dim=128,
        n_heads=8,
        n_layers=4,
        max_epochs=max_epochs,
        batch_size=128,
        temperature=0.1,
        seed=args.seed,
        checkpoints_dir=checkpoints_dir,
    )

    # Assemble Complete Experiment Results
    experiment_results = {
        "experiment_meta": {
            "date": "October 2026",
            "phase": "Phase 3: Self-Supervised Learning",
            "device": "cpu",
            "smoke_test": args.smoke_test,
            "seed": args.seed,
            "engine_counts": {
                "train": len(train_raw[ID_COL].unique()),
                "val": len(val_raw[ID_COL].unique()),
                "internal_test": len(itest_raw[ID_COL].unique()),
            },
        },
        "baselines": baselines_results,
        "failuredna_pooled": fd_pooled_results,
        "failuredna_regime_aware": fd_regime_results,
        "failuredna_regime_conditioned": fd_cond_results,
        "random_encoder": random_encoder_results,
        "ablation_w30": fd_w30_results,
    }

    # Save Structured JSON Summary
    summary_path = output_dir / "phase3_experiments_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, indent=2)
    logger.info(f"Phase 3 Experiments summary saved to: '{summary_path.as_posix()}'")

    # Generate Publication Figures
    logger.info("Rendering Phase 3 publication-quality figures...")
    plot_phase3_figures(experiment_results, figures_dir)

    print("\n" + "=" * 70)
    print("      FailureDNA Phase 3 Experiments Completed Successfully")
    print("=" * 70)
    print(f"Results JSON: {summary_path}")
    print(f"Figures Saved: {figures_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
