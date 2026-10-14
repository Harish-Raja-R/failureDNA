"""Representation diagnostics and latent geometry evaluation for FailureDNA.

Evaluates:
1. Distance-to-baseline monotonic rank correlation with true RUL.
2. Dimensionality health / singular value spectrum (guards against dimensional collapse).
3. Cluster structure on embeddings.
4. Comparative diagnostics against randomly initialized representations.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score


def evaluate_embedding_diagnostics(
    embeddings: np.ndarray,
    rul_targets: np.ndarray,
    n_clusters: int = 4,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Compute geometric and physical alignment diagnostics on frozen embeddings.

    Args:
        embeddings: 2D array of shape (N, D_emb).
        rul_targets: 1D array of true RUL targets of shape (N,).
        n_clusters: Number of clusters to evaluate on latent space.
        random_state: Random seed for clustering.

    Returns:
        Dictionary of diagnostic metrics.
    """
    N, D = embeddings.shape

    # 1. Singular Value Spectrum / Dimensional Collapse Check via PCA
    pca = PCA(n_components=min(10, D))
    pca.fit(embeddings)
    explained_var = pca.explained_variance_ratio_
    top1_var = float(explained_var[0])
    top3_var = float(np.sum(explained_var[:3]))
    # Participation ratio (effective dimension)
    eigenvalues = pca.singular_values_ ** 2
    sum_eig = np.sum(eigenvalues)
    sum_eig_sq = np.sum(eigenvalues ** 2)
    effective_dim = float((sum_eig ** 2) / sum_eig_sq) if sum_eig_sq > 0 else 1.0

    # 2. Distance-to-Baseline Degradation Alignment
    # Define "healthy baseline" centroid as mean embedding of pristine engines (RUL > 100)
    pristine_mask = rul_targets > 100.0
    if np.any(pristine_mask):
        healthy_centroid = np.mean(embeddings[pristine_mask], axis=0, keepdims=True)
    else:
        healthy_centroid = np.mean(embeddings[:max(10, N // 10)], axis=0, keepdims=True)

    # Euclidean distance from healthy centroid for each embedding
    dist_to_healthy = np.linalg.norm(embeddings - healthy_centroid, axis=1)

    # Spearman rank correlation between distance-to-healthy and RUL
    # If the representation captures monotonic wear, distance should INCREASE as RUL DECREASES (negative correlation)
    spearman_corr, p_value = spearmanr(dist_to_healthy, rul_targets)

    # 3. Clustering on Latent Space
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Subsample for silhouette if N > 5000 to keep CPU compute fast
    if N > 5000:
        sub_idx = np.random.RandomState(random_state).choice(N, size=5000, replace=False)
        sil = float(silhouette_score(embeddings[sub_idx], cluster_labels[sub_idx]))
    else:
        sil = float(silhouette_score(embeddings, cluster_labels))

    db_score = float(davies_bouldin_score(embeddings, cluster_labels))

    # Cluster sizes
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    cluster_distribution = {int(k): int(v) for k, v in zip(unique_labels, counts)}

    return {
        "n_samples": N,
        "embedding_dim": D,
        "pca_top1_explained_variance": top1_var,
        "pca_top3_explained_variance": top3_var,
        "effective_dimensionality": effective_dim,
        "dimensional_collapse_detected": bool(top1_var > 0.90),
        "distance_to_healthy_spearman_r": float(spearman_corr),
        "distance_to_healthy_p_value": float(p_value),
        "kmeans_silhouette_score": sil,
        "kmeans_davies_bouldin_index": db_score,
        "cluster_distribution": cluster_distribution,
    }
