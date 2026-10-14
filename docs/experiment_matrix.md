# FailureDNA: Comprehensive Experiment Matrix & Execution Protocol

This document details the complete experimental agenda for **FailureDNA** (EXP-01 through EXP-18). Every experiment is systematically tracked in **MLflow**, adhering strictly to the engine-level data isolation and research integrity protocols.

---

## Experiment Registry Overview

| Experiment ID | Category | Name | Primary Objective | Key Deliverable |
| :--- | :--- | :--- | :--- | :--- |
| **EXP-01** | Baseline | Raw Statistical Baseline | Establish linear heuristic benchmark | RUL baseline metrics |
| **EXP-02** | Baseline | Supervised Tree (XGBoost) | Gradient boosting on raw features | Tabular baseline metrics |
| **EXP-03** | Baseline | Deep Recurrent (LSTM) | Sequential supervised baseline | Neural sequential baseline |
| **EXP-04** | Baseline | Deep Recurrent (GRU) | Gated recurrent supervised baseline | Bidirectional GRU metrics |
| **EXP-05** | Baseline | Supervised Temporal Transformer | Benchmark architecture with full supervision | Supervised Transformer metrics |
| **EXP-06** | Core SSL | FailureDNA SSL Pretraining | Unsupervised temporal representation learning | Frozen encoder weights |
| **EXP-07** | Downstream | FailureDNA + XGBoost RUL | Evaluate linear/tree separability of SSL embedding | RUL prognostics performance |
| **EXP-08** | Downstream | FailureDNA + MLP RUL | Non-linear probe on frozen embeddings | Neural downstream prognostics |
| **EXP-09** | Discovery | K-Means Signature Discovery | Partition latent space into discrete wear regimes | K-Means cluster signatures |
| **EXP-10** | Discovery | HDBSCAN Density Signatures | Non-parametric density discovery of wear modes | Density clusters & noise profile |
| **EXP-11** | Discovery | GMM Probabilistic Signatures | Soft clustering and probability distribution | Gaussian mixture signatures |
| **EXP-12** | Evolution | Signature Evolution Modeling | Map cycle-by-cycle trajectory & transition graph | State transition matrix & DAG |
| **EXP-13** | Anomaly | Anomaly Detection Benchmark | Unsupervised anomaly scoring on embeddings | AUROC & anomaly scores |
| **EXP-14** | Early Warning | Degradation Early Warning | Measure detection horizon prior to failure | Mean lead time to failure (MLTF) |
| **EXP-15** | Ablation | Ablation: Contrastive vs Autoencoding | Quantify benefit of InfoNCE over MSE reconstruction | Reconstruction vs Contrastive |
| **EXP-16** | Ablation | Ablation: Augmentation Decomposition | Assess individual augmentation impact | Augmentation sensitivity |
| **EXP-17** | Sensitivity | Window-Size Sensitivity ($W$) | Evaluate temporal receptive field ($W \in \{30, 50, 70\}$) | Window-size trade-off curve |
| **EXP-18** | Sensitivity | Embedding-Dimension Sensitivity ($d_{emb}$) | Evaluate capacity scaling ($d_{emb} \in \{64, 128, 256\}$) | Dimension scaling curve |

---

## Detailed Experiment Specifications

### EXP-01: Raw Statistical Baseline
- **Category**: Baseline (Heuristic / Linear)
- **Hypothesis**: Simple aggregate statistics of sensor readings provide an initial linear baseline but cannot capture complex multi-regime degradation dynamics in FD004.
- **Input Data**: Raw C-MAPSS FD004 features summarized per window: rolling mean, rolling standard deviation, and minimum/maximum values across a window of length $W=50$.
- **Model**: Ordinary Least Squares (OLS) Linear Regression with Ridge penalty ($\alpha = 1.0$).
- **Optimization**: Closed-form analytical solution / MSE minimization.
- **Validation Protocol**: 70% train engines, 15% validation engines, 15% internal test engines.
- **Evaluation Metrics**: RMSE, MAE, $R^2$, NASA Asymmetric Score ($S$).
- **MLflow Tracking**: Train/test engine IDs, feature summary definitions, coefficients, residual plots.
- **Decision Gate**: Provides the absolute baseline floor; any advanced DL model must outperform EXP-01.

---

### EXP-02: Supervised Gradient Boosted Trees (XGBoost)
- **Category**: Baseline (Non-linear Tabular ML)
- **Hypothesis**: XGBoost operating on flattened temporal windows ($W \times D = 50 \times 14 = 700$ features) will capture non-linear sensor thresholds but fail to exploit sequential temporal ordering.
- **Input Data**: Flattened normalized sensor windows.
- **Model**: `xgboost.XGBRegressor` (n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8).
- **Optimization**: Squared Error loss with early stopping on validation engines.
- **Validation Protocol**: Fitted strictly on training engine windows; validated on validation engine windows.
- **Evaluation Metrics**: RMSE, MAE, $R^2$, NASA Asymmetric Score ($S$).
- **MLflow Tracking**: Hyperparameters, feature importance scores, learning curves, residual distribution.
- **Decision Gate**: Establishes standard industry benchmark for feature-engineered regression.

---

### EXP-03: Supervised Recurrent Baseline (LSTM)
- **Category**: Baseline (Deep Sequential)
- **Hypothesis**: A 2-layer LSTM can learn temporal transitions directly from raw sensor sequences, establishing a standard recurrent deep learning baseline.
- **Input Data**: Sequential 3D tensor: $(\text{Batch}, W=50, D=14)$.
- **Model Architecture**:
  - Input Linear projection: $D \to 128$
  - 2-layer LSTM ($d_{hidden}=128$, dropout=0.1)
  - Last step hidden state $h_W \in \mathbb{R}^{128}$
  - Linear regression head: $128 \to 64 \to 1$
- **Optimization**: AdamW, initial LR=$1\times 10^{-3}$, MSE loss, Cosine Annealing, batch size=64.
- **Validation Protocol**: Engine-level partitioned data.
- **Evaluation Metrics**: RMSE, MAE, $R^2$, NASA Score ($S$).
- **MLflow Tracking**: Epoch loss curves, validation metrics, model checkpoints.

---

### EXP-04: Supervised Recurrent Baseline (GRU)
- **Category**: Baseline (Deep Sequential)
- **Hypothesis**: Bidirectional GRU offers faster convergence and comparable representational capacity to LSTM with fewer gating parameters.
- **Input Data**: Sequential tensor $(\text{Batch}, W=50, D=14)$.
- **Model Architecture**:
  - 2-layer Bidirectional GRU ($d_{hidden}=64$ per direction $\to 128$ total, dropout=0.1)
  - Mean pooling across sequence dimension
  - Linear regression head: $128 \to 1$
- **Optimization**: AdamW, LR=$1\times 10^{-3}$, MSE loss, batch size=64.
- **Validation Protocol**: Identical engine split as EXP-03.
- **Evaluation Metrics**: RMSE, MAE, $R^2$, NASA Score ($S$), training throughput (samples/sec).
- **MLflow Tracking**: Epoch metrics, model state dict, loss logs.

---

### EXP-05: Supervised Temporal Transformer
- **Category**: Baseline (Supervised Attention Backbone)
- **Hypothesis**: A Transformer encoder trained end-to-end with direct RUL supervision captures long-range dependencies but may overfit to specific failure trajectories in the training set.
- **Input Data**: Sequential tensor $(\text{Batch}, W=50, D=14)$.
- **Model Architecture**:
  - Linear projection $D \to 128$
  - Learnable 1D positional encoding
  - 4 Pre-LayerNorm Transformer Encoder layers (8 heads, $d_{ff}=256$, dropout=0.1)
  - Global average pooling $\to h \in \mathbb{R}^{128}$
  - Direct linear head: $128 \to 1$
- **Optimization**: AdamW, LR=$5\times 10^{-4}$, MSE loss, batch size=64.
- **Validation Protocol**: Identical engine split.
- **Evaluation Metrics**: RMSE, MAE, $R^2$, NASA Score ($S$).
- **MLflow Tracking**: Training/validation loss curves, attention weight heatmaps, model checkpoint.
- **Decision Gate**: Direct comparison architecture against self-supervised FailureDNA.

---

### EXP-06: FailureDNA Self-Supervised Pretraining (Core Method)
- **Category**: Core Self-Supervised Representation Learning
- **Hypothesis**: Contrastive learning (NT-Xent) on stochastic temporal views of multivariate windows trains the Transformer encoder to extract invariant degradation features without using any labels.
- **Input Data**: Unlabeled temporal sensor windows $(\text{Batch}, W=50, D=14)$.
- **Augmentations**:
  - Stochastic view 1: Gaussian jitter ($\sigma=0.02$) + sensor channel mask ($p=0.15$).
  - Stochastic view 2: Temporal slice mask ($p=0.10$) + magnitude scaling ($\alpha \in [0.95, 1.05]$).
- **Model Architecture**:
  - Transformer Encoder backbone $f_\theta$ ($d_{model}=128, H=8, L=4$)
  - Non-linear projection head $g_\phi$: $\text{Linear}(128, 128) \to \text{GELU} \to \text{Linear}(128, 64) \to \text{L2 Norm}$
- **Loss Function**: NT-Xent / InfoNCE loss with temperature $\tau = 0.07$.
  $$\mathcal{L}_{NT-Xent} = -\frac{1}{2N} \sum_{i=1}^{2N} \log \frac{\exp(\text{sim}(z_i, z_{pos(i)}) / \tau)}{\sum_{k \ne i} \exp(\text{sim}(z_i, z_k) / \tau)}$$
- **Optimization**: AdamW (LR=$1\times 10^{-3}$, Weight Decay=$1\times 10^{-4}$), Cosine Annealing, 100 epochs, batch size=128.
- **Validation Protocol**: Validation InfoNCE loss evaluated on held-out validation engines.
- **Evaluation Metrics**: Train & Val InfoNCE Loss, Representation Alignment ($\mathcal{L}_{align}$), Representation Uniformity ($\mathcal{L}_{uniform}$).
- **MLflow Tracking**: Encoder weights checkpoint (`encoder_best.pt`), loss history, augmentation configs.
- **Decision Gate**: Encoder must achieve monotonic decrease in validation InfoNCE loss and stable uniformity score without collapse.

---

### EXP-07: Downstream Prognostics (FailureDNA + XGBoost)
- **Category**: Downstream Prognostics Evaluation
- **Hypothesis**: Features extracted from the frozen FailureDNA encoder provide linearly/monotonically separable inputs that enable XGBoost to achieve competitive RUL predictions without end-to-end backpropagation.
- **Input Data**: 128-dimensional frozen embeddings $h = f_{\theta^*}(X)$ generated by EXP-06.
- **Model**: `xgboost.XGBRegressor` trained on train engine embeddings.
- **Validation Protocol**: Evaluated on held-out test engine embeddings.
- **Evaluation Metrics**: RMSE, MAE, $R^2$, NASA Asymmetric Score ($S$).
- **Comparison Target**: Compare directly against EXP-02 (Raw XGBoost) and EXP-05 (Supervised Transformer).
- **MLflow Tracking**: Hyperparameters, RUL parity plots, error histograms, SHAP summary plots.

---

### EXP-08: Downstream Prognostics (FailureDNA + MLP)
- **Category**: Downstream Prognostics Evaluation
- **Hypothesis**: A lightweight 2-layer MLP probe on top of frozen FailureDNA embeddings matches or exceeds fully supervised deep baselines while training in a fraction of the time.
- **Input Data**: 128-dimensional frozen embeddings $h \in \mathbb{R}^{128}$.
- **Model**: $\text{Linear}(128 \to 64) \to \text{ReLU} \to \text{Dropout}(0.1) \to \text{Linear}(64 \to 1)$.
- **Optimization**: AdamW, LR=$1\times 10^{-3}$, MSE loss, 50 epochs.
- **Evaluation Metrics**: RMSE, MAE, $R^2$, NASA Score ($S$).
- **MLflow Tracking**: Training curves, validation metrics, model artifact.

---

### EXP-09: Failure Signature Discovery (K-Means)
- **Category**: Signature Discovery & Latent Geometry
- **Hypothesis**: Unsupervised clustering of the FailureDNA latent space reveals discrete degradation stages that correspond to physically distinct wear regimes.
- **Input Data**: Frozen FailureDNA embeddings $h \in \mathbb{R}^{128}$ from train and validation engines.
- **Algorithm**: K-Means clustering evaluated across $K \in \{3, 4, 5, 6, 7, 8, 9, 10\}$.
- **Metrics**: Silhouette Score, Davies-Bouldin Index (DBI), Calinski-Harabasz Index (CHI), Cluster Stability (Bootstrap ARI across 10 resamples).
- **Secondary Evaluation**: Engine coverage (percentage of engines passing through cluster) and temporal monotonicity ratio (does cluster index correlate with cycle progression?).
- **MLflow Tracking**: Cluster metrics vs $K$, 2D UMAP scatter colored by cluster assignments, cluster centroids.
- **Decision Gate**: Select optimal $K^*$ based on multi-metric consensus (not silhouette score alone).

---

### EXP-10: Failure Signature Discovery (HDBSCAN)
- **Category**: Signature Discovery & Latent Geometry
- **Hypothesis**: Density-based clustering identifies natural variable-density wear clusters and flags transitional operational anomalies as noise without forcing every point into a cluster.
- **Input Data**: Dimension-reduced embeddings (e.g. UMAP or PCA-16) of FailureDNA space.
- **Algorithm**: `HDBSCAN(min_cluster_size=50, min_samples=15)`.
- **Metrics**: Relative validity index (DBCV), outlier/noise ratio, number of discovered persistent clusters.
- **MLflow Tracking**: Persistence diagrams, noise percentage, cluster membership histograms.

---

### EXP-11: Failure Signature Discovery (Gaussian Mixture Models)
- **Category**: Signature Discovery & Latent Geometry
- **Hypothesis**: GMM provides soft probabilistic cluster assignments, modeling the continuous thermodynamic transition between health regimes.
- **Input Data**: Frozen FailureDNA embeddings $h \in \mathbb{R}^{128}$.
- **Algorithm**: GMM with full and diagonal covariance, evaluated for components $K \in [3, 8]$.
- **Metrics**: Bayesian Information Criterion (BIC), Akaike Information Criterion (AIC), Log-Likelihood, Silhouette Score.
- **MLflow Tracking**: BIC/AIC curve vs components, posterior probability distributions.

---

### EXP-12: Signature Evolution & Transition Graph
- **Category**: Temporal Dynamics & Process Mining
- **Hypothesis**: As engines progress through operating life, their trajectory traces a directed, low-entropy path across discovered signatures ($S_1 \to S_2 \to \dots \to S_{fail}$).
- **Input Data**: Time-ordered cluster assignments for all test engines: $S(t) \in \{1, \dots, K^*\}$.
- **Method**:
  1. Construct empirical transition probability matrix:
     $$P_{i, j} = \frac{N(S_i \to S_j)}{\sum_k N(S_i \to S_k)}$$
  2. Filter insignificant transitions ($P_{ij} < 0.05$) to produce a Directed Acyclic Graph (DAG).
  3. Compute dwell times (mean cycles spent in each signature state).
  4. Correlate entry cycle into terminal signature state with ground-truth remaining life.
- **Deliverable**: Transition matrix heatmap, interactive network graph, engine trajectory state plots.
- **MLflow Tracking**: Transition matrix CSV, graph JSON/DOT artifact, transition entropy metrics.

---

### EXP-13: Unsupervised Anomaly Detection Benchmark
- **Category**: Anomaly Detection
- **Hypothesis**: Distance in the FailureDNA embedding space from the nominal healthy baseline ($t \le 30$) increases monotonically as internal degradation accumulates.
- **Methods**:
  - **Method A**: Mahalanobis distance from the empirical mean and covariance of healthy baseline embeddings.
  - **Method B**: Isolation Forest trained strictly on healthy initial cycle embeddings.
- **Evaluation Protocol**: Evaluate anomaly scores across entire engine lifespans on test engines.
- **Metrics**: AUROC (early healthy vs late degraded), False Alarm Rate (FAR in first 20% of life).
- **MLflow Tracking**: Anomaly score trajectories, ROC curves, AUROC scores.

---

### EXP-14: Degradation Early Warning Horizon
- **Category**: Prognostic Early Warning
- **Hypothesis**: The initial transition out of the healthy baseline cluster signals imminent degradation significantly earlier than threshold-based raw sensor alarms.
- **Definition of Alarm**: First cycle where an engine transitions into a non-healthy degradation signature and remains out of the healthy state for $\ge 3$ consecutive cycles.
- **Metrics**:
  - **Detection Rate (DR)**: Percentage of failing engines successfully alerted.
  - **Mean Lead Time to Failure (MLTF)**: $\frac{1}{N_{engines}} \sum (T_{failure} - T_{alarm})$.
  - **Median Lead Time**: Median cycles of advance warning.
  - **Early False Alarm Rate (FAR)**: Percentage of alarms triggered when $RUL > 125$.
- **MLflow Tracking**: Lead time distributions, alarm trigger cycle charts.

---

### EXP-15: Ablation — Contrastive vs Autoencoding (MSE)
- **Category**: Architectural Ablation
- **Hypothesis**: Contrastive instance discrimination learns degradation invariants significantly better than Mean Squared Error (MSE) temporal autoencoding, which spends capacity reconstructing high-frequency sensor noise.
- **Model**: Identical Temporal Transformer encoder, but replacing the contrastive head with a 1D ConvTranspose decoder reconstructing $\hat{X} \in \mathbb{R}^{W \times D}$.
- **Loss**: $\mathcal{L}_{recon} = \frac{1}{W \cdot D} \|X - \hat{X}\|_F^2$.
- **Downstream Probe**: Train identical downstream XGBoost model on the autoencoder embeddings.
- **Comparison**: Compare downstream RUL RMSE and clustering silhouette against EXP-06/EXP-07.
- **MLflow Tracking**: Reconstruction error curves, downstream RUL metrics.

---

### EXP-16: Ablation — Augmentation Strategy Decomposition
- **Category**: Architectural Ablation
- **Hypothesis**: Combining temporal slice masking with sensor channel masking is essential for forcing the attention mechanism to learn inter-sensor and cross-time dependencies.
- **Configurations Evaluated**:
  - `CFG-A`: Gaussian Jitter only ($\sigma=0.02$).
  - `CFG-B`: Sensor Channel Masking only ($p=0.15$).
  - `CFG-C`: Temporal Slice Masking only ($p=0.10$).
  - `CFG-D`: Magnitude Scaling only ($\alpha \in [0.95, 1.05]$).
  - `CFG-E`: Full FailureDNA suite (all combined).
- **Metrics**: Validation InfoNCE loss, downstream RUL RMSE, embedding uniformity.
- **MLflow Tracking**: Ablation comparison table, downstream metric bar charts.

---

### EXP-17: Sensitivity Analysis — Window Size ($W$)
- **Category**: Hyperparameter Sensitivity
- **Hypothesis**: A window size of $W=50$ offers an optimal trade-off between capturing aerodynamic trends and minimizing latency / early-cycle data truncation.
- **Configurations Evaluated**:
  - $W = 30$ cycles
  - $W = 50$ cycles (Default)
  - $W = 70$ cycles
- **Evaluation**: Train SSL encoder for each $W$, extract embeddings, and evaluate downstream XGBoost RUL RMSE.
- **MLflow Tracking**: $W$ vs RMSE, training throughput, window truncation stats.

---

### EXP-18: Sensitivity Analysis — Embedding Dimension ($d_{emb}$)
- **Category**: Hyperparameter Sensitivity
- **Hypothesis**: $d_{emb}=128$ provides adequate capacity for representing multi-fault degradation without suffering from the curse of dimensionality during downstream clustering.
- **Configurations Evaluated**:
  - $d_{emb} = 64$
  - $d_{emb} = 128$ (Default)
  - $d_{emb} = 256$
- **Evaluation**: Downstream RUL RMSE, K-Means Silhouette Score, training memory footprint.
- **MLflow Tracking**: Dimension scaling curves, clustering scores.

---

## Execution Gates & Sequencing Plan

```
[ Phase 1: Preprocessing & Data Validation ]
                   │
                   ▼
[ Phase 2: Baselines EXP-01 through EXP-05 ]
                   │
                   ▼
[ Phase 3: Core SSL Pretraining EXP-06 ]
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
[ Phase 4: Downstream ]   [ Phase 5: Discovery ]
   EXP-07, EXP-08            EXP-09, EXP-10, EXP-11
         │                   │
         └─────────┬─────────┘
                   ▼
[ Phase 6: Evolution & Anomaly EXP-12, EXP-13, EXP-14 ]
                   │
                   ▼
[ Phase 7: Ablations & Sensitivity EXP-15 through EXP-18 ]
                   │
                   ▼
[ Phase 8: Optuna Tuning & Final Production Artifacts ]
```

Every experiment run will be logged directly to the local MLflow server or structured JSON logger, serializing run parameters, metrics, model artifacts, and evaluation plots without manual data entry.

---

## Phase 3: Empirical Execution Matrix & Observed Results

### Summary of Executed Phase 3 Configurations

| Run ID | Category | Method / Variant | Preprocessing | Backbone Architecture | Target / Probe | Val RMSE | Val $R^2$ | Val NASA Score | Test RMSE | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **BASE-01** | Baseline | Operating-Condition Only | None (Settings 1–3) | Linear OLS | Capped RUL (125) | 41.85 | -0.031 | 270,326.2 | 42.07 | Completed |
| **BASE-02** | Baseline | Linear / Ridge Window Stats | Pooled Robust | Summary Features (5 stats) | Ridge ($\alpha=1.0$) | 19.04 | 0.787 | 8,529.5 | 20.54 | Completed |
| **BASE-03** | Baseline | Random Forest Window Stats | Pooled Robust | Summary Features (5 stats) | RF ($N=100, D=10$) | 19.03 | 0.787 | 11,352.2 | 20.47 | Completed |
| **BASE-04** | Baseline | XGBoost Window Stats | Pooled Robust | Summary Features (5 stats) | XGBoost ($N=150, D=6$) | 24.40 | 0.650 | 12,491.2 | 24.61 | Completed |
| **BASE-05** | Baseline | Supervised LSTM | Pooled Robust | 2-layer LSTM ($d=64$) | Direct Regression Head | 72.82 | -2.122 | 835,912.1 | 65.61 | Completed |
| **BASE-06** | Baseline | Supervised Transformer | Pooled Robust | 4-layer Transformer ($d=64$) | Direct Regression Head | 83.63 | -3.118 | 2,224,780.4 | 75.82 | Completed |
| **SSL-A1** | FailureDNA | Variant A: Pooled Contrastive | Pooled Robust | 4L-8H Transformer ($d=128$) | Ridge Probe | 23.82 | 0.666 | 28,281.4 | 28.12 | Completed |
| **SSL-A2** | FailureDNA | Variant A: Pooled Contrastive | Pooled Robust | 4L-8H Transformer ($d=128$) | Random Forest Probe | 28.74 | 0.514 | 51,514.0 | 29.60 | Completed |
| **SSL-A3** | FailureDNA | Variant A: Pooled Contrastive | Pooled Robust | 4L-8H Transformer ($d=128$) | XGBoost Probe | 28.44 | 0.524 | 43,542.3 | 29.18 | Completed |
| **SSL-B1** | FailureDNA | Variant B: Regime-Aware SSL | Regime-Aware Standard | 4L-8H Transformer ($d=128$) | Ridge Probe | 21.38 | 0.731 | 12,926.5 | 23.40 | Completed |
| **SSL-B2** | FailureDNA | Variant B: Regime-Aware SSL | Regime-Aware Standard | 4L-8H Transformer ($d=128$) | Random Forest Probe | 23.50 | 0.675 | 20,640.6 | 26.83 | Completed |
| **SSL-B3** | FailureDNA | Variant B: Regime-Aware SSL | Regime-Aware Standard | 4L-8H Transformer ($d=128$) | XGBoost Probe | 24.22 | 0.655 | 23,296.0 | 27.83 | Completed |
| **SSL-C1** | FailureDNA | Variant C: Regime-Conditioned | Regime-Aware + One-Hot | 4L-8H Transformer ($d=128$) | Ridge Probe | **20.02** | **0.764** | **11,217.9** | **21.92** | Completed |
| **SSL-C2** | FailureDNA | Variant C: Regime-Conditioned | Regime-Aware + One-Hot | 4L-8H Transformer ($d=128$) | Random Forest Probe | 24.35 | 0.651 | 40,677.1 | 24.97 | Completed |
| **SSL-C3** | FailureDNA | Variant C: Regime-Conditioned | Regime-Aware + One-Hot | 4L-8H Transformer ($d=128$) | XGBoost Probe | 25.41 | 0.620 | 51,754.2 | 26.71 | Completed |
| **CTRL-01** | Control | Random Untrained Encoder | Pooled Robust | 2L-4H Random Transformer | Ridge Probe | 21.67 | 0.724 | 13,066.4 | 26.33 | Completed |
| **CTRL-02** | Control | Random Untrained Encoder | Pooled Robust | 2L-4H Random Transformer | Random Forest Probe | 20.74 | 0.747 | 16,155.4 | 24.82 | Completed |
| **CTRL-03** | Control | Random Untrained Encoder | Pooled Robust | 2L-4H Random Transformer | XGBoost Probe | 20.52 | 0.752 | 20,231.8 | 24.81 | Completed |
| **ABL-W30** | Ablation | Window Size $W=30$ Cycles | Pooled Robust | 4L-8H Transformer ($d=128$) | Ridge Probe | 22.11 | 0.704 | 28,516.4 | 30.72 | Completed |
| **ABL-W50** | Ablation | Window Size $W=50$ Cycles | Pooled Robust | 4L-8H Transformer ($d=128$) | Ridge Probe | 23.82 | 0.666 | 28,281.4 | 28.12 | Completed |

---

## Phase 4: Scientific Validation & Failure Signature Discovery Matrix

### Matched-Control & Multi-Seed Validation Results

| Experiment ID | Evaluation Condition | Model Type | Architecture | Preprocessing | Probe | Val RMSE (cycles) | Val $R^2$ | NASA Score ($S$) | Rank Corr $\|r_s\|$ | Effective Dim | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **EXP4-01** | Handcrafted Control | Tabular Summary Stats | 5 Stats (Mean/Std/Min/Max/Last) | Regime-Aware | Ridge | 25.88 | 0.588 | 47,152.2 | N/A | N/A | Completed |
| **EXP4-02** | Handcrafted Control | Tabular Summary Stats | 5 Stats (Mean/Std/Min/Max/Last) | Regime-Aware | Random Forest | **19.29** | **0.771** | **16,810.5** | N/A | N/A | Completed |
| **EXP4-03** | Core Method | FailureDNA (Pretrained SSL) | 4L-8H Transformer ($d=128$) | Pooled Robust | Ridge | 24.38 | 0.634 | 30,843.0 | 0.465 | 1.76 | Completed |
| **EXP4-04** | Core Method | FailureDNA (Pretrained SSL) | 4L-8H Transformer ($d=128$) | Pooled Robust | Random Forest | 28.79 | 0.490 | 58,327.2 | 0.465 | 1.76 | Completed |
| **EXP4-05** | Core Method | FailureDNA (Pretrained SSL) | 4L-8H Transformer ($d=128$) | Regime-Aware | Ridge | 21.23 | 0.723 | 13,458.0 | 0.662 | **4.64** | Completed |
| **EXP4-06** | Core Method | FailureDNA (Pretrained SSL) | 4L-8H Transformer ($d=128$) | Regime-Aware | Random Forest | 24.22 | 0.639 | 24,145.6 | 0.662 | **4.64** | Completed |
| **EXP4-07** | Core Method | FailureDNA (Pretrained SSL) | 4L-8H Transformer ($d=128$) | Regime-Conditioned | Ridge | 22.21 | 0.697 | 20,688.3 | 0.623 | 4.16 | Completed |
| **EXP4-08** | Core Method | FailureDNA (Pretrained SSL) | 4L-8H Transformer ($d=128$) | Regime-Conditioned | Random Forest | 25.12 | 0.612 | 49,152.8 | 0.623 | 4.16 | Completed |
| **EXP4-09** | Matched Negative Control | Random Untrained Encoder | 4L-8H Transformer ($d=128$, Seed 42-44) | Pooled Robust | Ridge | 21.77 | 0.709 | 14,227.9 | 0.360 | 1.19 | Completed |
| **EXP4-10** | Matched Negative Control | Random Untrained Encoder | 4L-8H Transformer ($d=128$, Seed 42-44) | Regime-Aware | Ridge | 22.03 | 0.700 | 15,629.0 | 0.708 | 2.51 | Completed |
| **EXP4-11** | Matched Negative Control | Random Untrained Encoder | 4L-8H Transformer ($d=128$, Seed 42-44) | Regime-Conditioned | Ridge | 20.44 | 0.738 | 20,768.3 | 0.709 | 2.27 | Completed |

### Discovered Signature Diagnostics & Transition Dynamics

| State Index | Signature Role | Mean Dwell Time (cycles) | Self-Transition $P_{ii}$ | Primary Next State | Key Sensor Characteristics |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **State 0** | Baseline Healthy Operation | 5.4 | 0.816 | State 1 ($P_{01}=0.160$) | Nominal HPC/LPT temps, low bleed enthalpy deviation |
| **State 1** | Developing Operational Wear | 20.4 | 0.952 | State 3 ($P_{13}=0.018$) | Stable operating plateau, slight positive drift in $T_{30}$ |
| **State 2** | Transitional Stress / Dynamic Shifts | 4.9 | 0.795 | State 0 / 1 ($P_{20}=0.091, P_{21}=0.091$) | Transient variance spikes, elevated bypass ratio fluctuations |
| **State 3** | Accelerated Degradation / Terminal | 16.9 | 0.960 | Absorbing ($P_{33}=0.960$) | Severe positive $T_{30}, T_{50}$ deviations, elevated core speed $N_c$ |

*Operating Regime Independence Audit*: Discovered signatures exhibit Normalized Mutual Information with flight regimes of **$NMI = 0.0031$** (near zero), confirming that discovered signatures reflect intrinsic mechanical wear independent of flight altitude, Mach, and throttle setpoints.


