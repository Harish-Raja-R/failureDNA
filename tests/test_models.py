"""Unit tests for Phase 3 models, losses, augmentations, and evaluation metrics."""

import numpy as np
import pandas as pd
import pytest
import torch

from failuredna.data.loader import CYCLE_COL, ID_COL, SENSOR_COLS, SETTING_COLS
from failuredna.data.preprocessing import RegimeAwarePreprocessor
from failuredna.evaluation.metrics import compute_regression_metrics, nasa_scoring_function
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
    MaskedTemporalPooling,
    ProjectionHead,
    SupervisedRULHead,
    SupervisedTransformerModel,
    TemporalTransformerEncoder,
)
from failuredna.models.loss import NTXentLoss


# --- 1. Model Shapes & Forward Passes ---

def test_temporal_transformer_encoder_shapes():
    """Verify encoder output dimensions and sequence mask handling."""
    B, W, D = 4, 50, 24
    x = torch.randn(B, W, D)
    mask = torch.ones(B, W)
    mask[:, :10] = 0.0  # Left 10 cycles are padded

    encoder = TemporalTransformerEncoder(input_dim=D, hidden_dim=128, n_heads=4, n_layers=2)
    h = encoder(x, mask=mask)

    assert h.shape == (B, 128)
    assert not torch.isnan(h).any()
    assert not torch.isinf(h).any()


def test_projection_head_shapes():
    """Verify projection head shape and normalization."""
    B, D_rep, D_proj = 8, 128, 64
    h = torch.randn(B, D_rep)
    proj = ProjectionHead(input_dim=D_rep, hidden_dim=128, output_dim=D_proj)
    z = proj(h)

    assert z.shape == (B, D_proj)
    assert not torch.isnan(z).any()


def test_contrastive_model_forward():
    """Verify integrated FailureDNAContrastiveModel."""
    B, W, D = 4, 30, 24
    x = torch.randn(B, W, D)
    encoder = TemporalTransformerEncoder(input_dim=D, hidden_dim=64, n_heads=4, n_layers=2)
    proj = ProjectionHead(input_dim=64, hidden_dim=64, output_dim=32)
    model = FailureDNAContrastiveModel(encoder, proj)

    h, z = model(x)
    assert h.shape == (B, 64)
    assert z.shape == (B, 32)


def test_masked_temporal_pooling():
    """Verify that masked pooling discounts padded positions."""
    B, W, D = 2, 5, 4
    x = torch.zeros(B, W, D)
    # Put large numbers in the first 2 time steps
    x[:, :2, :] = 100.0
    # Put 1.0 in the remaining 3 steps
    x[:, 2:, :] = 1.0

    # Mask: 0 for first 2 steps (padded), 1 for remaining 3 steps (valid)
    mask = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0, 1.0]])

    pooling = MaskedTemporalPooling(method="masked_mean", d_model=D)
    out = pooling(x, mask=mask)

    # Should equal 1.0 (mean of the 3 valid time steps), completely ignoring 100.0
    np.testing.assert_allclose(out.detach().numpy(), 1.0, atol=1e-5)


# --- 2. NT-Xent Contrastive Loss ---

def test_nt_xent_loss_finite_gradients():
    """Verify NT-Xent loss computation, finite values, and gradient flow."""
    B, D_proj = 8, 32
    z1 = torch.randn(B, D_proj, requires_grad=True)
    z2 = torch.randn(B, D_proj, requires_grad=True)

    criterion = NTXentLoss(temperature=0.1)
    loss = criterion(z1, z2)

    assert loss.dim() == 0  # Scalar
    assert loss.item() > 0.0
    assert torch.isfinite(loss)

    loss.backward()
    assert z1.grad is not None
    assert torch.isfinite(z1.grad).all()
    assert z2.grad is not None
    assert torch.isfinite(z2.grad).all()


def test_nt_xent_loss_invalid_inputs():
    """Verify validation on invalid batch size or temperature."""
    criterion = NTXentLoss(temperature=0.1)
    z1 = torch.randn(1, 32)
    z2 = torch.randn(1, 32)

    with pytest.raises(ValueError, match="Batch size must be >= 2"):
        criterion(z1, z2)

    with pytest.raises(ValueError, match="temperature must be > 0"):
        NTXentLoss(temperature=0.0)


# --- 3. Augmentation Pipeline ---

def test_temporal_augmentor_shapes_and_stochasticity():
    """Verify temporal augmentations produce valid, perturbed views."""
    W, D = 50, 24
    x = np.ones((W, D), dtype=np.float32) * 5.0
    augmentor = TemporalAugmentor(
        jitter_sigma=0.02,
        sensor_mask_prob=1.0,  # Force masking for test
        max_masked_sensors=2,
        scale_min=0.98,
        scale_max=1.02,
        random_state=42,
    )

    v1, v2 = augmentor.generate_views(x)
    assert v1.shape == (W, D)
    assert v2.shape == (W, D)
    assert not np.array_equal(v1, v2), "Views should be stochastically distinct"
    assert not np.array_equal(v1, x), "Augmented view should be perturbed from original"


# --- 4. NASA Asymmetric Score & Regression Metrics ---

def test_nasa_scoring_function_asymmetry():
    """Verify NASA score penalizes late predictions (d < 0) more severely than early (d > 0)."""
    y_true = np.array([50.0])

    # Late prediction: predicted 40 when true is 50 (d = -10, underestimation of wear)
    # Formula: exp(10/13) - 1 ≈ exp(0.7692) - 1 ≈ 2.158 - 1 = 1.158
    y_late = np.array([40.0])
    score_late = nasa_scoring_function(y_true, y_late)

    # Early prediction: predicted 60 when true is 50 (d = +10, overestimation of wear)
    # Formula: exp(10/10) - 1 = e - 1 ≈ 1.718
    # Wait, let's verify formula convention:
    # d_i = y_pred - y_true:
    # If y_pred < y_true (e.g. y_pred=40, y_true=50 => d = -10).
    # In C-MAPSS literature:
    # d < 0 means underestimation of remaining life? No:
    # If true RUL = 50 and model predicts 40, model predicts earlier failure (early alert).
    # If true RUL = 50 and model predicts 60, model predicts machine has MORE life than it actually does (late alert, catastrophic failure before replacement!).
    # So d = y_pred - y_true > 0 is late prediction!
    # Let's check:
    # In NASA C-MAPSS PHM 2008 definition:
    # d = y_pred - y_true:
    # if d < 0 (y_pred < y_true, early prediction): penalty = exp(-d/13) - 1
    # if d >= 0 (y_pred >= y_true, late prediction, failure arrives earlier than anticipated!): penalty = exp(d/10) - 1
    # For d = +20 (late alert by 20 cycles): penalty = exp(20/10) - 1 = exp(2) - 1 = 6.389
    # For d = -20 (early alert by 20 cycles): penalty = exp(20/13) - 1 = exp(1.538) - 1 = 3.657
    # 6.389 > 3.657! Late alert penalty is substantially higher!
    diff_mag = 20.0
    late_pred = np.array([y_true[0] + diff_mag])
    early_pred = np.array([y_true[0] - diff_mag])

    score_late = nasa_scoring_function(y_true, late_pred)
    score_early = nasa_scoring_function(y_true, early_pred)

    assert score_late > score_early, f"Late prediction penalty ({score_late:.2f}) must exceed early penalty ({score_early:.2f})"
    # Zero error
    assert nasa_scoring_function(y_true, y_true) == 0.0


def test_compute_regression_metrics():
    """Verify MAE, RMSE, R2, and NASA score computations."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 31.0, 39.0])

    metrics = compute_regression_metrics(y_true, y_pred)
    assert np.isclose(metrics["mae"], 1.5)
    assert np.isclose(metrics["rmse"], np.sqrt(2.5))
    assert metrics["r2"] > 0.90
    assert metrics["nasa_score"] > 0.0


# --- 5. RUL Target Computation and Capping ---

def test_compute_window_ruls_capping():
    """Verify RUL target alignment and piecewise-linear capping."""
    meta_df = pd.DataFrame({
        "unit_id": [1, 1, 1],
        "end_cycle": [50, 100, 200],
    })
    full_df = pd.DataFrame({
        ID_COL: [1, 1, 1, 1],
        CYCLE_COL: [1, 50, 100, 250],  # Max cycle is 250
    })

    # Capped at 125
    ruls_capped = compute_window_ruls(meta_df, full_df, cap_rul=125.0)
    # True RULs: 250 - 50 = 200 -> capped to 125
    #            250 - 100 = 150 -> capped to 125
    #            250 - 200 = 50 -> 50
    np.testing.assert_array_equal(ruls_capped, [125.0, 125.0, 50.0])

    # Uncapped
    ruls_uncapped = compute_window_ruls(meta_df, full_df, cap_rul=None)
    np.testing.assert_array_equal(ruls_uncapped, [200.0, 150.0, 50.0])


# --- 6. Baselines & Window Feature Extraction ---

def test_extract_window_summary_features():
    """Verify tabular feature extraction from 3D windows."""
    N, W, D = 10, 50, 24
    windows = np.random.randn(N, W, D).astype(np.float32)

    feats_5d = extract_window_summary_features(windows, include_min_max=True)
    assert feats_5d.shape == (N, 5 * D)

    feats_3d = extract_window_summary_features(windows, include_min_max=False)
    assert feats_3d.shape == (N, 3 * D)


def test_recurrent_baseline_forward():
    """Verify LSTM and GRU baselines forward pass and gradient flow."""
    B, W, D = 4, 30, 24
    x = torch.randn(B, W, D)

    lstm = RecurrentRULBaseline(input_dim=D, hidden_dim=64, num_layers=2, cell_type="lstm")
    pred_lstm = lstm(x)
    assert pred_lstm.shape == (B,)

    gru = RecurrentRULBaseline(input_dim=D, hidden_dim=64, num_layers=2, cell_type="gru")
    pred_gru = gru(x)
    assert pred_gru.shape == (B,)


def test_tabular_baselines_fit_predict():
    """Verify Linear, RF, and XGBoost baselines on synthetic feature matrices."""
    N, D_tab = 50, 48
    X = np.random.randn(N, D_tab).astype(np.float32)
    y = np.random.uniform(10.0, 125.0, size=N).astype(np.float32)

    # Linear
    lin = LinearWindowBaseline().fit(X, y)
    p_lin = lin.predict(X)
    assert len(p_lin) == N

    # RF
    rf = RandomForestBaseline(n_estimators=10).fit(X, y)
    p_rf = rf.predict(X)
    assert len(p_rf) == N

    # XGBoost
    xgb_m = XGBoostBaseline(n_estimators=10).fit(X, y)
    p_xgb = xgb_m.predict(X)
    assert len(p_xgb) == N


# --- 7. Regime-Aware Preprocessor Zero-Variance Guard ---

def test_regime_aware_preprocessor_guards_zero_variance():
    """Verify RegimeAwarePreprocessor handles invariant channels without division by zero."""
    rng = np.random.RandomState(42)
    rows = []
    # Cluster 1: Sea-level (alt=0, mach=0, TRA=100)
    for _ in range(50):
        rows.append([0.0 + rng.normal(0, 0.01), 0.0 + rng.normal(0, 0.001), 100.0])
    # Cluster 2: Mid-altitude (alt=20, mach=0.7, TRA=100)
    for _ in range(50):
        rows.append([20.0 + rng.normal(0, 0.01), 0.7 + rng.normal(0, 0.001), 100.0])

    df = pd.DataFrame(rows, columns=SETTING_COLS)
    df[ID_COL] = np.repeat([1, 2], 50)
    df[CYCLE_COL] = np.tile(np.arange(1, 51), 2)

    # Add dynamic and invariant sensors
    for s in SENSOR_COLS:
        if s in ("s_1", "s_18"):
            df[s] = 500.0  # Constant sensor (zero variance)
        else:
            df[s] = rng.uniform(10.0, 100.0, size=len(df))

    prep = RegimeAwarePreprocessor(n_regimes=2, include_settings=True, include_regime_onehot=True, seed=42)
    transformed = prep.fit_transform(df)

    assert not transformed.isna().any().any()
    assert not np.isinf(transformed.to_numpy()).any()
    # Constant channels should be exactly 0.0
    assert (transformed["s_1"] == 0.0).all()
    assert (transformed["s_18"] == 0.0).all()
    # One-hot regime indicators present
    assert "regime_0" in transformed.columns
    assert "regime_1" in transformed.columns
