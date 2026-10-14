"""Failure Signature Discovery & Trajectory Evolution Modeling.

Provides unsupervised clustering of FailureDNA latent representations,
trajectory transition dynamics, cluster stability, temporal coherence,
sensor characterization profiles, and operating-regime independence checks.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    mutual_info_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

from failuredna.utils.logging import get_logger

logger = get_logger(__name__)


class FailureSignatureClusterer:
    """Discovers discrete degradation signatures from FailureDNA latent representations.

    Implements:
    1. Multi-K clustering evaluation (K-Means and Gaussian Mixture Models).
    2. Cluster stability measurement (Adjusted Rand Index across random restarts).
    3. Temporal coherence assessment (monotonic correlation between signature ordering and engine lifecycle).
    4. Empirical state transition probability matrix and mean dwell times.
    5. Sensor signature profiling (mean standardized sensor shifts per signature).
    6. Operating regime independence audit (ensuring signatures do not confound with flight setpoints).
    """

    def __init__(
        self,
        n_clusters: int = 4,
        method: str = "kmeans",
        random_state: int = 42,
    ):
        """Initialize FailureSignatureClusterer.

        Args:
            n_clusters: Number of signature clusters K (default: 4).
            method: 'kmeans' or 'gmm'.
            random_state: Random seed for reproducibility.
        """
        if n_clusters < 2:
            raise ValueError(f"n_clusters must be >= 2, got {n_clusters}.")
        if method not in ("kmeans", "gmm"):
            raise ValueError(f"method must be 'kmeans' or 'gmm', got '{method}'.")

        self.n_clusters = n_clusters
        self.method = method
        self.random_state = random_state

        self.model: Optional[Union[KMeans, GaussianMixture]] = None
        self.is_fitted_ = False
        self.cluster_centers_: Optional[np.ndarray] = None
        self.ordered_cluster_mapping_: Optional[Dict[int, int]] = None

    def fit(self, embeddings: np.ndarray) -> "FailureSignatureClusterer":
        """Fit clustering model on training embeddings.

        Args:
            embeddings: 2D numpy array of shape (N, D).

        Returns:
            Fitted instance of self.
        """
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D array, got shape {embeddings.shape}.")

        if self.method == "kmeans":
            self.model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10,
                max_iter=300,
            )
            self.model.fit(embeddings)
            self.cluster_centers_ = self.model.cluster_centers_
        else:
            self.model = GaussianMixture(
                n_components=self.n_clusters,
                random_state=self.random_state,
                n_init=3,
                max_iter=200,
            )
            self.model.fit(embeddings)
            self.cluster_centers_ = self.model.means_

        self.is_fitted_ = True
        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster labels for given embeddings.

        Args:
            embeddings: 2D numpy array of shape (N, D).

        Returns:
            1D array of cluster indices (0 to K-1).
        """
        if not self.is_fitted_ or self.model is None:
            raise RuntimeError("FailureSignatureClusterer must be fitted before predict.")

        raw_labels = self.model.predict(embeddings)

        # If temporal reordering has been established, map labels to monotonic order
        if self.ordered_cluster_mapping_ is not None:
            return np.array([self.ordered_cluster_mapping_.get(lbl, lbl) for lbl in raw_labels], dtype=np.int32)
        return raw_labels.astype(np.int32)

    def order_clusters_by_lifecycle(
        self,
        embeddings: np.ndarray,
        cycle_progressions: np.ndarray,
    ) -> Dict[int, int]:
        """Order cluster indices monotonically by their mean lifecycle progression.

        Ensures Cluster 0 represents early healthy operation, progressing to
        Cluster K-1 representing terminal wear.

        Args:
            embeddings: Training embeddings.
            cycle_progressions: Normalized engine progression (e.g., cycle / max_cycle) or raw cycle.

        Returns:
            Mapping dictionary from raw cluster ID to ordered cluster ID.
        """
        if not self.is_fitted_ or self.model is None:
            raise RuntimeError("Model must be fitted before ordering clusters.")

        raw_labels = self.model.predict(embeddings)
        mean_lifecycles = {}
        for k in range(self.n_clusters):
            mask = raw_labels == k
            if np.any(mask):
                mean_lifecycles[k] = float(np.mean(cycle_progressions[mask]))
            else:
                mean_lifecycles[k] = float(k)

        # Sort clusters by average lifecycle
        sorted_clusters = sorted(mean_lifecycles.keys(), key=lambda k: mean_lifecycles[k])
        self.ordered_cluster_mapping_ = {old_k: new_k for new_k, old_k in enumerate(sorted_clusters)}
        return self.ordered_cluster_mapping_


def evaluate_signature_diagnostics(
    embeddings: np.ndarray,
    n_clusters_range: List[int] = [3, 4, 5, 6],
    method: str = "kmeans",
    n_stability_seeds: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Evaluate clustering quality, stability across seeds, and model selection metrics.

    Args:
        embeddings: 2D array of embeddings (N, D).
        n_clusters_range: List of K values to evaluate.
        method: 'kmeans' or 'gmm'.
        n_stability_seeds: Number of random restarts to assess Adjusted Rand Index stability.
        random_state: Base random seed.

    Returns:
        Dictionary of metrics per K.
    """
    results_by_k = {}

    for k in n_clusters_range:
        # Fit primary model
        if method == "kmeans":
            primary_km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            primary_labels = primary_km.fit_predict(embeddings)
            sil = float(silhouette_score(embeddings, primary_labels)) if len(embeddings) < 5000 else float(silhouette_score(embeddings[:5000], primary_labels[:5000]))
            dbi = float(davies_bouldin_score(embeddings, primary_labels))
            chi = float(calinski_harabasz_score(embeddings, primary_labels))
            bic = None
            aic = None
        else:
            primary_gmm = GaussianMixture(n_components=k, random_state=random_state, n_init=3)
            primary_labels = primary_gmm.fit_predict(embeddings)
            sil = float(silhouette_score(embeddings, primary_labels)) if len(embeddings) < 5000 else float(silhouette_score(embeddings[:5000], primary_labels[:5000]))
            dbi = float(davies_bouldin_score(embeddings, primary_labels))
            chi = float(calinski_harabasz_score(embeddings, primary_labels))
            bic = float(primary_gmm.bic(embeddings))
            aic = float(primary_gmm.aic(embeddings))

        # Stability across random restarts (Adjusted Rand Index)
        ari_scores = []
        for seed_offset in range(1, n_stability_seeds + 1):
            seed = random_state + seed_offset * 17
            if method == "kmeans":
                km_restart = KMeans(n_clusters=k, random_state=seed, n_init=5)
                lbl_restart = km_restart.fit_predict(embeddings)
            else:
                gmm_restart = GaussianMixture(n_components=k, random_state=seed, n_init=2)
                lbl_restart = gmm_restart.fit_predict(embeddings)
            ari_scores.append(float(adjusted_rand_score(primary_labels, lbl_restart)))

        results_by_k[str(k)] = {
            "n_clusters": k,
            "silhouette_score": sil,
            "davies_bouldin_index": dbi,
            "calinski_harabasz_score": chi,
            "bic": bic,
            "aic": aic,
            "mean_stability_ari": float(np.mean(ari_scores)),
            "std_stability_ari": float(np.std(ari_scores)),
        }

    return results_by_k


def compute_state_transition_matrix(
    cluster_labels: np.ndarray,
    meta_df: pd.DataFrame,
    n_clusters: int,
    engine_max_cycles: Optional[Dict[int, int]] = None,
    strict_complete: bool = False,
) -> Tuple[np.ndarray, Dict[int, float], Dict[str, Any]]:
    """Compute empirical transition matrix P_ij and mean state dwell times.

    Trajectories are tracked per engine unit ID in temporal sequence.
    Explicitly distinguishes uncensored state transitions, genuine engine
    failure terminations, and artificial analysis truncations.

    Args:
        cluster_labels: 1D array of assigned signature clusters.
        meta_df: Metadata containing 'unit_id' and 'end_cycle'.
        n_clusters: Number of signature states K.
        engine_max_cycles: Optional dictionary mapping unit_id to ground-truth
            maximum engine cycle. Used to detect whether a trajectory was
            artificially truncated or completed until natural engine failure.
        strict_complete: If True and engine_max_cycles is provided, raises
            ValueError if any trajectory is detected as incomplete/sliced.

    Returns:
        Tuple of:
        - transition_matrix: (K, K) float array where row sums equal 1.0 (or 0 if unvisited).
        - dwell_times: Dict mapping state k to mean cycles spent before transition.
        - trajectory_stats: Transition counts, forward-monotonicity ratio,
          and decomposed episode counts (uncensored vs genuine terminal vs artificially truncated).
    """
    df = meta_df.copy()
    df["state"] = cluster_labels
    df = df.sort_values(by=["unit_id", "end_cycle"]).reset_index(drop=True)

    counts = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    state_dwells: Dict[int, List[int]] = {k: [] for k in range(n_clusters)}
    uncensored_dwells: Dict[int, List[int]] = {k: [] for k in range(n_clusters)}
    censored_dwells: Dict[int, List[int]] = {k: [] for k in range(n_clusters)}
    genuine_terminal_dwells: Dict[int, List[int]] = {k: [] for k in range(n_clusters)}
    artificially_truncated_dwells: Dict[int, List[int]] = {k: [] for k in range(n_clusters)}

    for unit_id_val, engine_group in df.groupby("unit_id"):
        uid = int(unit_id_val)
        states = engine_group["state"].to_numpy()
        cycles = engine_group["end_cycle"].to_numpy()
        if len(states) <= 1:
            continue

        # Check trajectory completeness if ground-truth max cycles provided
        is_complete = True
        obs_max_cycle = int(cycles.max())
        if engine_max_cycles is not None:
            true_max_cycle = engine_max_cycles.get(uid, obs_max_cycle)
            if obs_max_cycle < true_max_cycle:
                is_complete = False
                if strict_complete:
                    raise ValueError(
                        f"Incomplete trajectory detected for engine {uid}: "
                        f"observed up to cycle {obs_max_cycle}, but true engine end-of-life is cycle {true_max_cycle}. "
                        f"Trajectory slicing must not silently turn an incomplete engine into a terminal episode."
                    )

        # Count consecutive step transitions
        for t in range(len(states) - 1):
            curr_s = states[t]
            next_s = states[t + 1]
            counts[curr_s, next_s] += 1

        # Calculate run-length dwell times
        current_state = states[0]
        run_length = 1
        for s in states[1:]:
            if s == current_state:
                run_length += 1
            else:
                state_dwells[current_state].append(run_length)
                uncensored_dwells[current_state].append(run_length)
                current_state = s
                run_length = 1

        # Trajectory boundary episode
        state_dwells[current_state].append(run_length)
        censored_dwells[current_state].append(run_length)
        if is_complete:
            genuine_terminal_dwells[current_state].append(run_length)
        else:
            artificially_truncated_dwells[current_state].append(run_length)

    # Normalize operational transition matrix (conditional on non-termination)
    transition_matrix = np.zeros((n_clusters, n_clusters), dtype=np.float64)
    for i in range(n_clusters):
        row_sum = np.sum(counts[i, :])
        if row_sum > 0:
            transition_matrix[i, :] = counts[i, :] / row_sum
        else:
            transition_matrix[i, i] = 1.0  # Absorbing state if unvisited

    # Construct absorbing transition matrix P_abs of shape (K + 1, K + 1)
    # Index K represents terminal failure absorption state Delta
    counts_abs = np.zeros((n_clusters + 1, n_clusters + 1), dtype=np.int64)
    counts_abs[:n_clusters, :n_clusters] = counts
    for k in range(n_clusters):
        counts_abs[k, n_clusters] = len(genuine_terminal_dwells[k])

    absorbing_transition_matrix = np.zeros((n_clusters + 1, n_clusters + 1), dtype=np.float64)
    for i in range(n_clusters):
        row_sum_abs = np.sum(counts_abs[i, :])
        if row_sum_abs > 0:
            absorbing_transition_matrix[i, :] = counts_abs[i, :] / row_sum_abs
        else:
            absorbing_transition_matrix[i, i] = 1.0
    # State K is absorbing (terminal functional failure Delta)
    absorbing_transition_matrix[n_clusters, n_clusters] = 1.0

    # Compute empirical mean dwell times across all episodes
    dwell_times = {}
    for k in range(n_clusters):
        dwells = state_dwells[k]
        dwell_times[k] = float(np.mean(dwells)) if len(dwells) > 0 else 0.0

    # Forward monotonicity: fraction of off-diagonal transitions where next_s > curr_s
    off_diag_forward = 0
    off_diag_backward = 0
    for i in range(n_clusters):
        for j in range(n_clusters):
            if i != j:
                if j > i:
                    off_diag_forward += counts[i, j]
                else:
                    off_diag_backward += counts[i, j]

    total_off_diag = off_diag_forward + off_diag_backward
    forward_ratio = float(off_diag_forward / total_off_diag) if total_off_diag > 0 else 1.0

    # Markov theoretical dwell times:
    # 1. markov_conditional: 1 / (1 - P_op[k, k]) conditional on transitioning to another active state
    # 2. markov_absorbing: 1 / (1 - P_abs[k, k]) accounting for both inter-state transitions and terminal failure
    markov_conditional = {}
    markov_absorbing = {}
    for k in range(n_clusters):
        p_kk = transition_matrix[k, k]
        p_kk_abs = absorbing_transition_matrix[k, k]
        markov_conditional[k] = float(1.0 / (1.0 - p_kk)) if p_kk < 1.0 else float("inf")
        markov_absorbing[k] = float(1.0 / (1.0 - p_kk_abs)) if p_kk_abs < 1.0 else float("inf")

    def _compute_distribution_summary(run_list: List[int]) -> Dict[str, Any]:
        if len(run_list) == 0:
            return {
                "count": 0,
                "sum_steps": 0,
                "mean": 0.0,
                "std": 0.0,
                "median": 0.0,
                "min": 0,
                "q25": 0.0,
                "q50": 0.0,
                "q75": 0.0,
                "max": 0,
                "iqr": 0.0,
            }
        arr = np.array(run_list, dtype=np.float64)
        q25 = float(np.percentile(arr, 25))
        q50 = float(np.percentile(arr, 50))
        q75 = float(np.percentile(arr, 75))
        return {
            "count": int(len(arr)),
            "sum_steps": int(np.sum(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": q50,
            "min": int(np.min(arr)),
            "q25": q25,
            "q50": q50,
            "q75": q75,
            "max": int(np.max(arr)),
            "iqr": float(q75 - q25),
        }

    episode_distributions = {
        k: {
            "all": _compute_distribution_summary(state_dwells[k]),
            "operational": _compute_distribution_summary(uncensored_dwells[k]),
            "terminal": _compute_distribution_summary(genuine_terminal_dwells[k]),
        }
        for k in range(n_clusters)
    }

    stats = {
        "total_transitions": int(np.sum(counts)),
        "forward_transitions": int(off_diag_forward),
        "backward_transitions": int(off_diag_backward),
        "forward_monotonicity_ratio": forward_ratio,
        "total_episodes": {k: len(state_dwells[k]) for k in range(n_clusters)},
        "uncensored_episode_counts": {k: len(uncensored_dwells[k]) for k in range(n_clusters)},
        "censored_episode_counts": {k: len(censored_dwells[k]) for k in range(n_clusters)},
        "genuine_terminal_episode_counts": {k: len(genuine_terminal_dwells[k]) for k in range(n_clusters)},
        "artificially_truncated_episode_counts": {k: len(artificially_truncated_dwells[k]) for k in range(n_clusters)},
        "uncensored_mean_dwell_times": {
            k: float(np.mean(uncensored_dwells[k])) if len(uncensored_dwells[k]) > 0 else 0.0
            for k in range(n_clusters)
        },
        "censored_mean_dwell_times": {
            k: float(np.mean(censored_dwells[k])) if len(censored_dwells[k]) > 0 else 0.0
            for k in range(n_clusters)
        },
        "genuine_terminal_mean_dwell_times": {
            k: float(np.mean(genuine_terminal_dwells[k])) if len(genuine_terminal_dwells[k]) > 0 else 0.0
            for k in range(n_clusters)
        },
        "artificially_truncated_mean_dwell_times": {
            k: float(np.mean(artificially_truncated_dwells[k])) if len(artificially_truncated_dwells[k]) > 0 else 0.0
            for k in range(n_clusters)
        },
        "absorbing_transition_matrix": absorbing_transition_matrix,
        "markov_absorbing_dwell_times": markov_absorbing,
        "markov_conditional_dwell_times": markov_conditional,
        "markov_theoretical_dwell_times": markov_conditional,
        "episode_distributions": episode_distributions,
    }

    return transition_matrix, dwell_times, stats


def compute_signature_sensor_profiles(
    windows: np.ndarray,
    cluster_labels: np.ndarray,
    feature_names: List[str],
    n_clusters: int,
) -> pd.DataFrame:
    """Compute mean standardized sensor values within each discovered signature.

    Takes the final observed cycle of each window (the current telemetry state).

    Args:
        windows: (N, W, D) telemetry array.
        cluster_labels: 1D cluster assignments.
        feature_names: Sensor / channel names (length D).
        n_clusters: Number of clusters K.

    Returns:
        DataFrame with rows = feature_names, columns = Signature 0, ..., Signature K-1.
    """
    last_step = windows[:, -1, :]  # (N, D)
    profile_data = {}

    for k in range(n_clusters):
        mask = cluster_labels == k
        if np.any(mask):
            mean_vals = np.mean(last_step[mask], axis=0)
        else:
            mean_vals = np.zeros(last_step.shape[1])
        profile_data[f"Signature_{k}"] = mean_vals

    return pd.DataFrame(profile_data, index=feature_names)


def audit_regime_vs_signature_independence(
    signature_labels: np.ndarray,
    regime_labels: np.ndarray,
) -> Dict[str, Any]:
    """Audit the statistical association between signature clusters and operating regimes.

    Computes contingency cross-tabulation, marginal entropies, mutual information,
    and normalized mutual information (NMI).

    Args:
        signature_labels: 1D array of failure signature clusters.
        regime_labels: 1D array of operating flight regimes (0 to 5).

    Returns:
        Dictionary containing contingency cross-tabulation and mutual information scores.
    """
    ct = pd.crosstab(
        pd.Series(signature_labels, name="Signature"),
        pd.Series(regime_labels, name="Operating_Regime"),
    )

    n_total = len(signature_labels)
    sig_counts = np.bincount(signature_labels)
    reg_counts = np.bincount(regime_labels)

    p_sig = sig_counts[sig_counts > 0] / n_total
    p_reg = reg_counts[reg_counts > 0] / n_total

    h_sig_nats = float(-np.sum(p_sig * np.log(p_sig)))
    h_reg_nats = float(-np.sum(p_reg * np.log(p_reg)))

    mi_nats = float(mutual_info_score(signature_labels, regime_labels))
    nmi_arithmetic = float(normalized_mutual_info_score(signature_labels, regime_labels, average_method="arithmetic"))
    nmi_geometric = float(normalized_mutual_info_score(signature_labels, regime_labels, average_method="geometric"))

    log2_e = float(np.log2(np.e))
    h_sig_bits = float(h_sig_nats * log2_e)
    h_reg_bits = float(h_reg_nats * log2_e)
    mi_bits = float(mi_nats * log2_e)

    n_sig = len(np.unique(signature_labels))
    n_reg = len(np.unique(regime_labels))

    return {
        "contingency_table": ct.to_dict(),
        "normalized_mutual_info": nmi_arithmetic,
        "nmi_arithmetic": nmi_arithmetic,
        "nmi_geometric": nmi_geometric,
        "mutual_info_nats": mi_nats,
        "mutual_info_bits": mi_bits,
        "entropy_signatures_nats": h_sig_nats,
        "entropy_regimes_nats": h_reg_nats,
        "entropy_signatures_bits": h_sig_bits,
        "entropy_regimes_bits": h_reg_bits,
        "n_signatures": n_sig,
        "n_regimes": n_reg,
        "is_low_association": bool(nmi_arithmetic < 0.25),
        "is_independent": bool(nmi_arithmetic < 0.25),
        "interpretation": (
            f"Low measured empirical association between signature clusters and flight regimes (NMI = {nmi_arithmetic:.4f}). "
            "Note: Low NMI indicates low statistical mutual dependence on this partition; it does not constitute physical independence."
        ),
    }
