"""Unit tests for Failure Signature Discovery, Trajectory Dynamics, and Regime Independence."""

import numpy as np
import pandas as pd
import pytest

from failuredna.evaluation.signatures import (
    FailureSignatureClusterer,
    audit_regime_vs_signature_independence,
    compute_signature_sensor_profiles,
    compute_state_transition_matrix,
    evaluate_signature_diagnostics,
)


def test_signature_clusterer_fit_predict_shapes():
    """Verify FailureSignatureClusterer fit and predict behavior."""
    N, D = 100, 32
    embeddings = np.random.randn(N, D).astype(np.float32)

    # K-Means
    km_clust = FailureSignatureClusterer(n_clusters=4, method="kmeans", random_state=42)
    km_clust.fit(embeddings)
    preds = km_clust.predict(embeddings)

    assert preds.shape == (N,)
    assert set(np.unique(preds)).issubset({0, 1, 2, 3})
    assert km_clust.cluster_centers_.shape == (4, D)

    # GMM
    gmm_clust = FailureSignatureClusterer(n_clusters=3, method="gmm", random_state=42)
    gmm_clust.fit(embeddings)
    preds_gmm = gmm_clust.predict(embeddings)

    assert preds_gmm.shape == (N,)
    assert set(np.unique(preds_gmm)).issubset({0, 1, 2})


def test_signature_ordering_by_lifecycle():
    """Verify cluster indices are reordered monotonically along engine lifecycles."""
    N = 120
    # Create 3 synthetic clusters cleanly separated along one dimension
    c0 = np.random.randn(40, 4) + np.array([10.0, 0, 0, 0])  # Early lifecycle
    c1 = np.random.randn(40, 4) + np.array([20.0, 0, 0, 0])  # Mid lifecycle
    c2 = np.random.randn(40, 4) + np.array([30.0, 0, 0, 0])  # Late lifecycle
    embeddings = np.vstack([c0, c1, c2]).astype(np.float32)

    # Lifecycles: c0 has cycles 1..40, c1 has 41..80, c2 has 81..120
    cycles = np.concatenate([np.linspace(0.1, 0.3, 40), np.linspace(0.4, 0.6, 40), np.linspace(0.7, 1.0, 40)])

    clusterer = FailureSignatureClusterer(n_clusters=3, method="kmeans", random_state=42)
    clusterer.fit(embeddings)
    mapping = clusterer.order_clusters_by_lifecycle(embeddings, cycles)

    ordered_preds = clusterer.predict(embeddings)
    # Mean cycle in cluster 0 should be less than cluster 1, which is less than cluster 2
    m0 = np.mean(cycles[ordered_preds == 0])
    m1 = np.mean(cycles[ordered_preds == 1])
    m2 = np.mean(cycles[ordered_preds == 2])
    assert m0 < m1 < m2, f"Clusters must be ordered monotonically: {m0:.2f} < {m1:.2f} < {m2:.2f}"


def test_evaluate_signature_diagnostics():
    """Verify multi-K diagnostics and stability calculation."""
    N, D = 80, 16
    embeddings = np.random.randn(N, D).astype(np.float32)

    diag = evaluate_signature_diagnostics(embeddings, n_clusters_range=[2, 3], method="kmeans", n_stability_seeds=3)
    assert "2" in diag
    assert "3" in diag
    assert "silhouette_score" in diag["2"]
    assert "mean_stability_ari" in diag["2"]
    assert -1.0 <= diag["2"]["silhouette_score"] <= 1.0


def test_state_transition_matrix_properties():
    """Verify transition matrix rows sum to 1.0 and dwell times are non-negative."""
    # 2 engines with 10 cycles each
    meta_df = pd.DataFrame({
        "unit_id": [1] * 10 + [2] * 10,
        "end_cycle": list(range(1, 11)) + list(range(1, 11)),
    })
    # Monotonic progression: 0, 0, 0, 1, 1, 1, 2, 2, 3, 3
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 3, 3] * 2)

    P, dwells, stats = compute_state_transition_matrix(labels, meta_df, n_clusters=4)

    assert P.shape == (4, 4)
    # Row sums should equal 1.0
    for i in range(4):
        assert np.isclose(np.sum(P[i, :]), 1.0)

    # Forward transitions should dominate
    assert stats["forward_transitions"] > 0
    assert stats["backward_transitions"] == 0
    assert stats["forward_monotonicity_ratio"] == 1.0

    # Dwell times should be positive
    for k in range(4):
        assert dwells[k] > 0.0


def test_signature_sensor_profiles_shape():
    """Verify sensor profile dataframe format."""
    N, W, D = 30, 20, 5
    windows = np.random.randn(N, W, D).astype(np.float32)
    labels = np.random.choice([0, 1, 2], size=N)
    feat_names = [f"sensor_{i}" for i in range(D)]

    profiles = compute_signature_sensor_profiles(windows, labels, feat_names, n_clusters=3)
    assert profiles.shape == (5, 3)
    assert list(profiles.columns) == ["Signature_0", "Signature_1", "Signature_2"]
    assert list(profiles.index) == feat_names


def test_regime_independence_audit():
    """Verify cross-tabulation and mutual information scoring."""
    signatures = np.array([0, 0, 1, 1, 2, 2, 3, 3] * 10)
    # Perfectly random regimes 0..5
    rng = np.random.RandomState(42)
    regimes = rng.choice(6, size=len(signatures))

    audit = audit_regime_vs_signature_independence(signatures, regimes)
    assert "contingency_table" in audit
    assert "normalized_mutual_info" in audit
    assert 0.0 <= audit["normalized_mutual_info"] <= 1.0
    assert "is_low_association" in audit
    assert "entropy_signatures_nats" in audit
    assert "entropy_regimes_nats" in audit
    assert "mutual_info_bits" in audit


def test_dwell_time_estimator_boundary_cases_and_censoring():
    """Verify dwell-time estimator behavior across boundary and censoring scenarios."""
    # Synthetic dataset with 3 engines testing boundary conditions
    # Engine 1: Single-step alternation: 0, 1, 0, 1, 0 (all episodes length 1)
    # Engine 2: Monotonic progression: 0 (3 cycles), 1 (4 cycles), 2 (5 cycles)
    # Engine 3: Trajectory completely in State 2 (10 cycles)
    meta_df = pd.DataFrame({
        "unit_id": [1] * 5 + [2] * 12 + [3] * 10,
        "end_cycle": list(range(1, 6)) + list(range(1, 13)) + list(range(1, 11)),
    })
    labels = np.array(
        [0, 1, 0, 1, 0] +
        [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2] +
        [2] * 10
    )

    P, dwells, stats = compute_state_transition_matrix(labels, meta_df, n_clusters=4)

    # State 0 checks:
    # Engine 1 episodes: [1, 1, 1] (two uncensored, one censored at end)
    # Engine 2 episodes: [3] (uncensored)
    # Total episodes for State 0: [1, 1, 1, 3], mean = 6 / 4 = 1.50
    assert np.isclose(dwells[0], 1.50)
    assert stats["total_episodes"][0] == 4
    assert stats["uncensored_episode_counts"][0] == 3
    assert stats["censored_episode_counts"][0] == 1

    # State 3 is unvisited (Boundary: zero visits)
    assert dwells[3] == 0.0
    assert stats["total_episodes"][3] == 0
    assert np.isclose(P[3, 3], 1.0)  # Absorbing by convention if unvisited

    # State 2 checks:
    # Engine 2: [5] (censored at end of Engine 2)
    # Engine 3: [10] (censored at end of Engine 3)
    # Mean dwell = (5 + 10) / 2 = 7.5
    assert np.isclose(dwells[2], 7.50)
    assert stats["censored_episode_counts"][2] == 2
    assert stats["uncensored_episode_counts"][2] == 0

    # Self-transition probabilities must satisfy P_ii in [0, 1]
    for k in range(4):
        assert 0.0 <= P[k, k] <= 1.0


def test_nmi_information_theory_consistency():
    """Verify information-theoretic quantities, log base conversion, and NMI formula."""
    # Create fixed known contingency matrix
    # State 0: [10, 10], State 1: [20, 20] -> Perfectly independent!
    signatures = np.array([0] * 20 + [1] * 40)
    regimes = np.array([0, 1] * 10 + [0, 1] * 20)

    audit = audit_regime_vs_signature_independence(signatures, regimes)

    # When independent, MI must be exactly 0 (within float precision)
    assert np.isclose(audit["mutual_info_nats"], 0.0, atol=1e-6)
    assert np.isclose(audit["normalized_mutual_info"], 0.0, atol=1e-6)

    # Base change consistency: bits = nats * log2(e)
    log2_e = np.log2(np.e)
    assert np.isclose(audit["entropy_signatures_bits"], audit["entropy_signatures_nats"] * log2_e)
    assert np.isclose(audit["entropy_regimes_bits"], audit["entropy_regimes_nats"] * log2_e)
    assert np.isclose(audit["mutual_info_bits"], audit["mutual_info_nats"] * log2_e)

    # Verify NMI arithmetic formula: 2 * MI / (H_U + H_V)
    # Now with non-zero correlation
    sig2 = np.array([0] * 30 + [1] * 30)
    reg2 = np.array([0] * 25 + [1] * 5 + [0] * 5 + [1] * 25)
    audit2 = audit_regime_vs_signature_independence(sig2, reg2)

    h_u = audit2["entropy_signatures_nats"]
    h_v = audit2["entropy_regimes_nats"]
    mi = audit2["mutual_info_nats"]
    expected_nmi = 2.0 * mi / (h_u + h_v)
    assert np.isclose(audit2["nmi_arithmetic"], expected_nmi, atol=1e-6)


def test_trajectory_slicing_incomplete_engine_detection():
    """Verify that trajectory slicing cannot silently turn an incomplete engine into a terminal episode."""
    # Engine 1 has true max cycle 100, but only observed up to cycle 50 (sliced)
    # Engine 2 has true max cycle 60, observed up to cycle 60 (complete)
    meta_df = pd.DataFrame({
        "unit_id": [1] * 5 + [2] * 6,
        "end_cycle": [10, 20, 30, 40, 50] + [10, 20, 30, 40, 50, 60],
    })
    labels = np.array([0, 0, 1, 1, 1] + [0, 0, 1, 1, 2, 2])
    true_max_cycles = {1: 100, 2: 60}

    # Strict mode must raise ValueError on sliced trajectory
    with pytest.raises(ValueError, match="Incomplete trajectory detected for engine 1"):
        compute_state_transition_matrix(
            labels,
            meta_df,
            n_clusters=3,
            engine_max_cycles=true_max_cycles,
            strict_complete=True,
        )

    # Non-strict mode must categorize the sliced episode into artificially_truncated, NOT genuine_terminal
    _, dwells, stats = compute_state_transition_matrix(
        labels,
        meta_df,
        n_clusters=3,
        engine_max_cycles=true_max_cycles,
        strict_complete=False,
    )

    # Engine 1 ended in State 1 at cycle 50 (incomplete):
    assert stats["artificially_truncated_episode_counts"][1] == 1
    assert stats["genuine_terminal_episode_counts"][1] == 0

    # Engine 2 ended in State 2 at cycle 60 (complete):
    assert stats["artificially_truncated_episode_counts"][2] == 0
    assert stats["genuine_terminal_episode_counts"][2] == 1

    # Both are counted in total censored episodes
    assert stats["censored_episode_counts"][1] == 1
    assert stats["censored_episode_counts"][2] == 1
    assert stats["censored_episode_counts"][0] == 0


def test_absorbing_transition_matrix_and_dwell_reconciliation():
    """Verify absorbing transition matrix with terminal failure state Delta and dwell reconciliation."""
    # Synthetic dataset with 3 engines:
    # Engine 1: completes in State 1 (length 4) -> Terminal exit to Delta
    # Engine 2: transitions 0 -> 1 -> 0 (length 2, 2, 2) -> completes in State 0 -> Terminal exit to Delta
    # Engine 3: completes in State 1 (length 6) -> Terminal exit to Delta
    meta_df = pd.DataFrame({
        "unit_id": [1] * 4 + [2] * 6 + [3] * 6,
        "end_cycle": list(range(1, 5)) + list(range(1, 7)) + list(range(1, 7)),
    })
    labels = np.array(
        [1, 1, 1, 1] +
        [0, 0, 1, 1, 0, 0] +
        [1, 1, 1, 1, 1, 1]
    )
    engine_max = {1: 4, 2: 6, 3: 6}

    P_op, dwells, stats = compute_state_transition_matrix(
        labels, meta_df, n_clusters=2, engine_max_cycles=engine_max, strict_complete=True
    )

    # 1. Operational transition matrix (2x2)
    assert P_op.shape == (2, 2)
    for i in range(2):
        assert np.isclose(np.sum(P_op[i, :]), 1.0)

    # 2. Absorbing transition matrix (3x3 with state 2 = Delta)
    assert "absorbing_transition_matrix" in stats
    P_abs = stats["absorbing_transition_matrix"]
    assert P_abs.shape == (3, 3)

    # Row sums must all equal 1.0
    for i in range(3):
        assert np.isclose(np.sum(P_abs[i, :]), 1.0)

    # Absorbing state Delta has P_abs[2, 2] == 1.0
    assert np.isclose(P_abs[2, 2], 1.0)
    assert np.isclose(P_abs[2, 0], 0.0)
    assert np.isclose(P_abs[2, 1], 0.0)

    # Terminal failure transitions exist for State 0 (Engine 2) and State 1 (Engines 1 & 3)
    assert P_abs[0, 2] > 0.0
    assert P_abs[1, 2] > 0.0

    # 3. Dwell-time reconciliation:
    # Absorbing Markov dwell time 1 / (1 - P_abs[k, k]) must algebraically equal empirical mean dwell
    assert "markov_absorbing_dwell_times" in stats
    assert "markov_conditional_dwell_times" in stats
    for k in range(2):
        assert np.isclose(stats["markov_absorbing_dwell_times"][k], dwells[k])

    # 4. Conditional dwell time only counts transitions to other operational states:
    # State 1 has 1 uncensored exit to State 0 (in Engine 2) and 2 terminal exits to Delta (Engines 1 & 3).
    # Conditional dwell > absorbing dwell because terminal exits are excluded from the conditional denominator.
    assert stats["markov_conditional_dwell_times"][1] > stats["markov_absorbing_dwell_times"][1]


def test_dwell_distribution_and_episode_length_conventions():
    """Verify episode-length conventions, quantile distributions, and algebraic identity."""
    # Engine 1: 10 steps with known runs:
    # State 0: 3 steps (cycles 1..3) -> operational transition to State 1
    # State 1: 4 steps (cycles 4..7) -> operational transition to State 0
    # State 0: 3 steps (cycles 8..10) -> terminal failure at cycle 10
    meta_df = pd.DataFrame({
        "unit_id": [1] * 10,
        "end_cycle": list(range(1, 11)),
    })
    labels = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0])
    engine_max = {1: 10}

    P_op, dwells, stats = compute_state_transition_matrix(
        labels, meta_df, n_clusters=2, engine_max_cycles=engine_max, strict_complete=True
    )

    # 1. Episode length conventions
    dist = stats["episode_distributions"]
    # State 0 has 2 episodes: lengths [3, 3] -> sum = 6
    assert dist[0]["all"]["count"] == 2
    assert dist[0]["all"]["sum_steps"] == 6
    assert dist[0]["all"]["mean"] == 3.0
    assert dist[0]["all"]["median"] == 3.0
    assert dist[0]["all"]["min"] == 3
    assert dist[0]["all"]["max"] == 3
    # One operational (length 3) and one terminal (length 3)
    assert dist[0]["operational"]["count"] == 1
    assert dist[0]["operational"]["mean"] == 3.0
    assert dist[0]["terminal"]["count"] == 1
    assert dist[0]["terminal"]["mean"] == 3.0

    # State 1 has 1 episode: length 4 (cycles 4..7: 7 - 4 + 1 = 4)
    assert dist[1]["all"]["count"] == 1
    assert dist[1]["all"]["sum_steps"] == 4
    assert dist[1]["all"]["mean"] == 4.0
    assert dist[1]["all"]["median"] == 4.0
    assert dist[1]["operational"]["count"] == 1
    assert dist[1]["terminal"]["count"] == 0

    # Total steps across all episodes equals total trajectory length
    total_steps = dist[0]["all"]["sum_steps"] + dist[1]["all"]["sum_steps"]
    assert total_steps == 10

    # 2. Algebraic identity demonstration:
    # 1 / (1 - P_abs[k, k]) == empirical mean dwell holds by construction
    # for ANY sequence, including this deterministic, non-Markovian sequence!
    assert np.isclose(stats["markov_absorbing_dwell_times"][0], dwells[0])
    assert np.isclose(stats["markov_absorbing_dwell_times"][1], dwells[1])

    # 3. Operational conditional dwell denominator:
    # State 0: 6 total steps. Final step at cycle 10 is terminal failure.
    # Non-terminal steps preceding transitions = 5 (transitions: 0->0, 0->0, 0->1, 0->0, 0->0)
    # Self-transitions = 4, operational exits = 1
    # P_op[0, 0] = 4 / 5 = 0.8 -> tau_conditional = 1 / (1 - 0.8) = 5.0
    # In contrast, absorbing tau_abs = 6 / 2 = 3.0
    assert np.isclose(stats["markov_conditional_dwell_times"][0], 5.0)
    assert np.isclose(stats["markov_absorbing_dwell_times"][0], 3.0)

    # 4. Explicit distinction between model-based reciprocal hazard estimate (5.0)
    # and observed sample mean among operational-exit episodes (3.0):
    observed_op_mean_state_0 = dist[0]["operational"]["mean"]
    model_based_reciprocal_hazard_state_0 = stats["markov_conditional_dwell_times"][0]
    assert np.isclose(observed_op_mean_state_0, 3.0)
    assert np.isclose(model_based_reciprocal_hazard_state_0, 5.0)
    assert model_based_reciprocal_hazard_state_0 != observed_op_mean_state_0

    # 5. Inclusive Time-to-Terminal-Failure (TTF) formula verification:
    # For State 1, first entry is cycle 4, terminal failure cycle is 10.
    # Inclusive TTF = T_e - min{t | s_t = 1} + 1 = 10 - 4 + 1 = 7 cycles.
    first_entry_cycle_s1 = int(meta_df[labels == 1]["end_cycle"].min())
    failure_cycle = int(meta_df["end_cycle"].max())
    ttf_inclusive = failure_cycle - first_entry_cycle_s1 + 1
    assert ttf_inclusive == 7
    # Distinct from individual episode duration (4 cycles)
    assert ttf_inclusive > dist[1]["all"]["mean"]



