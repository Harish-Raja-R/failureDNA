# FailureDNA: Self-Supervised Discovery of Failure Signatures from Multivariate Industrial Time-Series
## CAT-II Comprehensive Research Specification & Technical Blueprint

- **Course**: 19MAM56 – Advanced Machine Learning Lab
- **Milestone**: CAT-II Project Work
- **Domain**: Self-Supervised Representation Learning, Industrial Predictive Maintenance, Time-Series Prognostics
- **Primary Benchmark**: NASA Turbofan Engine Degradation Simulation Dataset (C-MAPSS), Sub-dataset FD004

---

## 1. Final Project Title
**FailureDNA: Self-Supervised Discovery of Failure Signatures from Multivariate Industrial Time-Series**

---

## 2. Problem Statement
Traditional industrial predictive maintenance (PdM) paradigms predominantly frame equipment health through supervised classification or regression:
$$\text{"Will this machine fail within time window } \Delta t\text{?"} \quad \text{or} \quad \text{"What is the remaining useful life (RUL) } y \in \mathbb{R}^+\text{?"}$$

While computationally tractable, this formulation suffers from three fundamental structural limitations in real-world engineering environments:
1. **Severe Label Scarcity & Asymmetry**: Industrial assets are engineered for extreme reliability; run-to-failure trajectories are rare, costly, and hazardous. Generating supervised failure labels requires sacrificing critical capital assets or simulating catastrophic failures under artificial testbed constraints.
2. **Degradation is a Dynamic Process, Not a Discrete Event**: Machinery degradation is not an instantaneous transition from "healthy" to "failed." Turbomachinery and complex industrial systems progress through continuous thermodynamic, aerodynamic, and mechanical wear regimes (e.g., micro-crack propagation, blade erosion, seal degradation, thermal barrier breakdown).
3. **Black-Box Prognostics Without Behavioral Understanding**: Point predictions of RUL fail to explain *how* a machine is deteriorating or *which latent behavioral failure mode* is developing.

**FailureDNA** departs from the traditional formulation by addressing the foundational question:
$$\text{"What latent temporal behavioral patterns emerge as complex industrial machinery progressively degrades toward failure?"}$$

We frame failure signature discovery as an unsupervised temporal representation learning problem, followed by latent geometry analysis and discrete signature evolution modeling.

---

## 3. Research Question
### Primary Research Question
> **Can self-supervised temporal representation learning discover distinct, physically meaningful machine degradation and failure signatures from multivariate industrial time-series without utilizing failure labels, ground-truth RUL, fault-mode annotations, or future trajectory information during representation learning?**

### Secondary Research Questions
1. **Geometric Coherence**: Do unsupervised clusters discovered within the self-supervised embedding space exhibit monotonic degradation ordering and temporal consistency across unseen, independent engine trajectories?
2. **Signature Evolution**: Can the progressive deterioration of complex multi-component assets be mapped as a deterministic or stochastic transition graph between discovered temporal signatures?
3. **Downstream Utility**: Does contrastive self-supervised pretraining produce representations that meet or exceed fully supervised baselines in downstream tasks including RUL regression, unsupervised anomaly detection, and early warning lead-time detection?
4. **Operating Regime Invariance**: Can the representation isolate mechanical degradation signatures from severe operating condition shifts (altitude, Mach number, throttle settings)?

---

## 4. Motivation
In critical cyber-physical systems—such as commercial aviation turbofans, power generation gas turbines, and industrial centrifugal compressors—unplanned downtime induces catastrophic operational disruption and safety risks. 

Standard supervised models overfit to specific failure instances present in historical training sets and fail to generalize when machines degrade under complex, multi-regime operating conditions. Furthermore, operators demand **actionable diagnostic signatures** that reveal the nature and stage of degradation rather than ungrounded numerical scalar predictions.

Self-supervised representation learning (SSL) has revolutionized natural language processing and computer vision by capturing rich contextual structures from unannotated corpora. By tailoring contrastive temporal representation learning to multivariate sensor time-series, FailureDNA extracts latent dynamical invariants, isolating intrinsic mechanical health states from extrinsic flight-condition confounding factors.

---

## 5. Research Objectives
1. **Formulate a Contrastive SSL Pipeline**: Develop a temporal contrastive self-supervised learning framework (InfoNCE / NT-Xent) that maps sliding multivariate sensor windows to a normalized latent hypersphere without label supervision.
2. **Domain-Preserving Temporal Augmentations**: Implement stochastic temporal augmentations (Gaussian jitter, sensor channel masking, temporal slice masking, magnitude scaling, controlled temporal perturbation) that preserve physical causality and aerodynamic invariants.
3. **Temporal Transformer Encoder**: Design and train a deep multi-head self-attention encoder with learnable positional encoding to model long-range temporal dependencies across sensor telemetry.
4. **Failure Signature Discovery**: Discover discrete degradation regimes by applying geometric and density-based clustering algorithms (K-Means, GMM, HDBSCAN) directly to the FailureDNA embedding space.
5. **Signature Evolution & Transition Modeling**: Construct engine-level trajectory models, transition matrices, and directed failure graphs tracing the path from healthy baselines to terminal breakdown.
6. **Downstream Prognostics & Anomaly Benchmarking**: Rigorously evaluate representation utility on RUL regression (XGBoost, MLP), anomaly detection (Mahalanobis distance, Isolation Forest), and early warning lead time.
7. **Full-Stack Industrial Deployment**: Package the entire system into a production-ready FastAPI inference engine with Pydantic validation, containerize with Docker, deploy to cloud infrastructure, and provide an interactive React/TypeScript analytical dashboard.

---

## 6. Dataset Justification: NASA C-MAPSS FD004
We select the NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) benchmark, specifically **FD004**, as the core evaluation platform.

| Property | FD001 | FD002 | FD003 | **FD004 (Target)** |
| :--- | :--- | :--- | :--- | :--- |
| **Operating Conditions** | 1 (Sea Level) | 6 (Full Flight Envelope) | 1 (Sea Level) | **6 (Full Flight Envelope)** |
| **Fault Modes** | 1 (HPC Degradation) | 1 (HPC Degradation) | 2 (HPC + Fan) | **2 (HPC + Fan Degradation)** |
| **Train Trajectories** | 100 engines | 260 engines | 100 engines | **249 engines** |
| **Test Trajectories** | 100 engines | 259 engines | 100 engines | **248 engines** |
| **Trajectory Lengths** | 128 – 362 cycles | 128 – 378 cycles | 141 – 525 cycles | **128 – 543 cycles** |
| **Total Sensor Feeds** | 21 channels | 21 channels | 21 channels | **21 channels (+ 3 operational)** |
| **Degradation Complexity** | Low | High | Medium | **Extremely High** |

### Why FD004?
- **Maximum Operational Realism**: Unlike FD001 (which operates at a single trivial sea-level condition), FD004 operates across 6 distinct combinations of altitude (0 – 42,000 ft), Mach number (0.0 – 0.84), and Throttle Resolver Angle (TRA 20 – 100°).
- **Multiple Coupled Failure Modes**: Both High-Pressure Compressor (HPC) degradation and Fan degradation occur simultaneously or independently across units, reflecting true industrial complexity.
- **Confounding Environmental Dynamics**: Raw sensor readings fluctuate by orders of magnitude simply due to altitude and throttle changes, requiring the representation learning framework to untangle environmental dynamics from true mechanical wear.

---

## 7. Dataset Assumptions That Must Be Verified
All assumptions regarding C-MAPSS FD004 will be empirically audited against raw files during Phase 1:

1. **File Schema & Delimiters**: Confirm raw files (`train_FD004.txt`, `test_FD004.txt`, `RUL_FD004.txt`) are whitespace-delimited with 26 space-separated columns:
   - Column 1: Engine Unit ID (integer)
   - Column 2: Operational Cycle Number (integer, 1-indexed)
   - Columns 3–5: Operational Settings 1, 2, 3 (Altitude, Mach, TRA)
   - Columns 6–26: Sensor Measurements 1 through 21 (continuous float)
2. **Sensor Variance & Invariance**: In single-condition datasets (FD001), sensors 1, 5, 10, 16, 18, 19 have zero variance. In FD004, operating shifts may induce artificial variance in these channels. We must empirically audit each sensor’s variance within fixed operating clusters before dropping or retaining channels.
3. **Operating Regime Identifiability**: Verify whether the 3 operational settings cluster cleanly into 6 distinct operating regimes via unsupervised clustering (e.g., K-Means on settings 1–3).
4. **Trajectory Termination Semantics**: In `train_FD004.txt`, trajectories run continuously until system failure ($RUL = 0$ at the final cycle of each engine). In `test_FD004.txt`, trajectories are pruned at an arbitrary cycle prior to failure, and the actual remaining cycles to failure are stored in `RUL_FD004.txt`.
5. **No Data Imputation Needed**: C-MAPSS simulation logs are clean of random missing values (`NaN`), but contain sensor noise that must be handled via robust scaling rather than missing-value imputation.

---

## 8. Proposed Methodology
The FailureDNA architecture consists of a rigorous seven-stage pipeline:

```
[ Raw C-MAPSS FD004 Stream ]
             │
             ▼
[ 1. Data Validation & Schema Audit ]
             │
             ▼
[ 2. Operating Regime Clustering & Sensor Normalization ]
             │
             ▼
[ 3. Strict Engine-Level Partitioning (Train / Val / Test) ]
             │
             ▼
[ 4. Non-Leaking Temporal Window Extraction (W = 50, Stride = 1) ]
             │
             ▼
[ 5. Stochastic Temporal Augmentation Engine (View A & View B) ]
             │
             ▼
[ 6. Temporal Transformer Encoder (d_model=128, H=8, L=4) ]
             │
             ▼
[ 7. Non-Linear Projection Head (g_phi: 128 -> 128 -> 64) ]
             │
             ▼
[ 8. NT-Xent Contrastive Loss Optimization ]
             │
             ▼
================== FREEZE REPRESENTATION BACKBONE ==================
             │
             ▼
[ 9. FailureDNA Latent Embeddings (h in R^128) ]
             │
      ┌──────┴───────────────────────────┬────────────────────────────┐
      ▼                                  ▼                            ▼
[ Unsupervised Signature Discovery ] [ Downstream Prognostics ] [ Anomaly Detection ]
(K-Means, GMM, HDBSCAN)              (XGBoost, MLP for RUL)     (Mahalanobis, IF)
      │                                  │                            │
      ▼                                  ▼                            ▼
[ Signature Evolution Graphs ]       [ RUL Evaluation: ]        [ Early Warning: ]
[ Transition Matrix P_ij ]           [ RMSE, MAE, R^2, Score ]  [ Lead Time, FAR ]
```

---

## 9. Mathematical Formulation of the Self-Supervised Objective
Let a multivariate temporal window extracted from an engine trajectory be defined as:
$$X \in \mathbb{R}^{W \times D}$$
where $W$ is the temporal sequence length (e.g., $W=50$ cycles) and $D$ is the number of active sensor channels.

### Stochastic Augmentation Operators
We define an augmentation family $\mathcal{T}$. For each window $X$, we independently sample two stochastic augmentation operators $t_A, t_B \sim \mathcal{T}$ yielding two correlated augmented views of the same temporal context:
$$\tilde{X}_A = t_A(X), \quad \tilde{X}_B = t_B(X)$$

### Temporal Feature Encoding & Projection
The temporal views pass through a shared Temporal Transformer backbone $f_\theta: \mathbb{R}^{W \times D} \to \mathbb{R}^{d_{emb}}$:
$$h_A = f_\theta(\tilde{X}_A), \quad h_B = f_\theta(\tilde{X}_B) \quad \in \mathbb{R}^{d_{emb}}$$

The embeddings are mapped to a lower-dimensional latent contrastive space via a multi-layer non-linear projection head $g_\phi: \mathbb{R}^{d_{emb}} \to \mathbb{R}^{d_{proj}}$:
$$z_A = g_\phi(h_A), \quad z_B = g_\phi(h_B) \quad \in \mathbb{R}^{d_{proj}}$$
Both projections are normalized onto the unit hypersphere:
$$\hat{z} = \frac{z}{\|z\|_2}$$

### Contrastive Similarity Metric
The cosine similarity between two normalized representations is:
$$\text{sim}(u, v) = u^\top v$$

### InfoNCE / NT-Xent Objective
For a minibatch of $N$ temporal windows, stochastic augmentation generates $2N$ views. For any positive pair $(i, j)$ originating from the identical underlying window $X$, all other $2(N-1)$ augmented windows within the minibatch are treated as negative examples.

The contrastive loss for positive pair $(i, j)$ is formulated as:
$$\ell_{i, j} = -\log \frac{\exp\left(\text{sim}(\hat{z}_i, \hat{z}_j) / \tau\right)}{\sum_{k=1, k \ne i}^{2N} \exp\left(\text{sim}(\hat{z}_i, \hat{z}_k) / \tau\right)}$$
where $\tau > 0$ denotes the temperature hyperparameter controlling the sharpness of penalty against hard negatives.

The total self-supervised objective computed across all positive pairs in the minibatch is:
$$\mathcal{L}_{\text{FailureDNA}} = \frac{1}{2N} \sum_{k=1}^{N} \left[ \ell_{2k-1, 2k} + \ell_{2k, 2k-1} \right]$$

**Theoretical Guarantees**:
- **Alignment**: Positive pairs $(h_A, h_B)$ map to nearby points on the hypersphere, enforcing invariance to noise, minor sensor dropout, and jitter.
- **Uniformity**: The denominator forces negative pairs apart, maximizing entropy and preventing representation collapse without requiring negative pair labels.

---

## 10. Model Architecture Specification
The model architecture is defined modularly and configurably:

```
Input Temporal Window: (Batch, W=50, D_sensors)
          │
          ▼
Linear / 1D Conv Projection: (Batch, W=50, d_model=128)
          │
          ▼
Add Learnable / Sinusoidal Positional Encoding: (Batch, W=50, 128)
          │
          ▼
Dropout (p = 0.1)
          │
          ▼
┌────────────────────────────────────────────────────────────────┐
│  Transformer Encoder Layer x 4 (Pre-LayerNorm Architecture)    │
│  ├── LayerNorm(x)                                              │
│  ├── Multi-Head Self-Attention (Heads=8, d_k=16, d_v=16)       │
│  ├── Dropout(0.1) + Residual Connection                        │
│  ├── LayerNorm(x)                                              │
│  ├── Position-Wise Feed-Forward (d_model=128 -> d_ff=256 -> 128)│
│  └── Dropout(0.1) + Residual Connection                        │
└────────────────────────────────────────────────────────────────┘
          │
          ▼
Global Temporal Pooling (Mean Pooling over W or Attention Pooling)
          │
          ▼
FailureDNA Backbone Embedding: h in R^128 (Latent Feature Vector)
          │
          ▼
Non-Linear Projection Head g_phi:
  Linear(128 -> 128) -> LayerNorm -> GELU -> Linear(128 -> 64) -> L2 Normalization
          │
          ▼
Contrastive Latent Vector: z in R^64 (Unit Hypersphere)
```

### Initial Hyperparameter Baseline (Configurable)
- Window Length ($W$): 50 cycles
- Window Stride ($S$): 1 cycle
- Input Features ($D$): 14–17 active sensor channels (empirically determined)
- Transformer Embedding Dimension ($d_{model}$): 128
- Self-Attention Heads ($H$): 8
- Feed-Forward Hidden Dimension ($d_{ff}$): 256
- Encoder Layers ($L$): 4
- Dropout Probability ($p$): 0.1
- Projection Dimension ($d_{proj}$): 64
- Contrastive Temperature ($\tau$): 0.07
- Optimizer: AdamW (Weight Decay = $1\times 10^{-4}$)
- Learning Rate: $1\times 10^{-3}$ with Cosine Annealing Schedule

---

## 11. Data Leakage Prevention Strategy
To ensure strict scientific integrity, the following leakage-prevention protocols are enforced across all pipelines:

1. **Engine-Level Trajectory Partitioning**:
   - Splitting is performed strictly at the **Engine ID** level. Never randomly sample sliding windows across the full dataset.
   - 249 training engines in `train_FD004.txt` are partitioned into:
     - **SSL Training Engines (70%)**: $\approx 174$ engines
     - **SSL Validation Engines (15%)**: $\approx 37$ engines
     - **Held-Out Evaluation Engines (15%)**: $\approx 38$ engines
   - The official NASA test set (`test_FD004.txt` + `RUL_FD004.txt`, 248 engines) is held completely isolated until final benchmark verification.
2. **Strict Temporal Isolation**:
   - Temporal sliding windows never span across engine boundaries.
   - For an engine with lifespan $T$, windows are generated strictly within $t \in [W, T]$.
3. **No-Peeking Preprocessing**:
   - All scalers (StandardScaler, RobustScaler, MinMax) and operating-regime clusterers are fit **exclusively** on the SSL training engines.
   - Validation and test engines are transformed using parameters frozen from the training set.
4. **Zero-Label Representation Learning**:
   - Representation pretraining does not ingest RUL values, failure cycle numbers, operational cycle indices, or fault mode tags.
   - Downstream models (XGBoost, MLP) receive representations only after the SSL backbone weights are completely frozen.
5. **Causal Temporal Windows**:
   - Augmentation operators never incorporate future information beyond window horizon $t$.

---

## 12. Experiment Matrix
Eighteen core experiments are planned, tracked via MLflow, and cataloged in [experiment_matrix.md](file:///c:/Users/Harish/Downloads/FailureDNA/docs/experiment_matrix.md):

| Experiment ID | Category | Name / Description | Model / Method | Primary Metrics |
| :--- | :--- | :--- | :--- | :--- |
| **EXP-01** | Baseline | Raw Statistical Baseline | Summary Stats + Linear Reg | RMSE, MAE, $R^2$ |
| **EXP-02** | Baseline | Supervised Tree Baseline | Raw Flattened Windows + XGBoost | RMSE, MAE, $R^2$ |
| **EXP-03** | Baseline | Recurrent Baseline (LSTM) | 2-Layer LSTM (Supervised) | RMSE, MAE, $R^2$ |
| **EXP-04** | Baseline | Recurrent Baseline (GRU) | 2-Layer GRU (Supervised) | RMSE, MAE, $R^2$ |
| **EXP-05** | Baseline | Supervised Temporal Transformer | Transformer Encoder + Linear Reg Head | RMSE, MAE, $R^2$ |
| **EXP-06** | Core SSL | FailureDNA SSL Pretraining | Temporal Transformer + NT-Xent | InfoNCE Loss, Alignment, Uniformity |
| **EXP-07** | Downstream | FailureDNA + XGBoost RUL | Frozen SSL Embeddings + XGBoost | RMSE, MAE, $R^2$, NASA Score |
| **EXP-08** | Downstream | FailureDNA + MLP RUL | Frozen SSL Embeddings + 2-Layer MLP | RMSE, MAE, $R^2$, NASA Score |
| **EXP-09** | Discovery | K-Means Signature Discovery | K-Means on Embeddings ($K \in [3, 10]$) | Silhouette, DB Index, CH Index, Stability |
| **EXP-10** | Discovery | HDBSCAN Density Signatures | HDBSCAN Density Clustering | Noise Ratio, Silhouette, Persistence |
| **EXP-11** | Discovery | GMM Probabilistic Signatures | Gaussian Mixture Models | BIC, AIC, Silhouette, Log-Likelihood |
| **EXP-12** | Evolution | Signature Evolution Modeling | Transition Matrices & Markov Graphs | Transition Entropy, Monotonicity |
| **EXP-13** | Anomaly | Anomaly Detection Benchmark | Mahalanobis Distance & Isolation Forest | AUROC, False Alarm Rate (FAR) |
| **EXP-14** | Early Warning | Degradation Early Warning | First Signature Transition Horizon | Detection Rate, Mean Lead Time (MLTF) |
| **EXP-15** | Ablation | Ablation: No Contrastive Objective | Reconstruction Autoencoder (MSE) | Embedding Quality, Downstream RMSE |
| **EXP-16** | Ablation | Ablation: Augmentation Strategy | Ablate Jitter, Masking, Scaling | InfoNCE Loss, Downstream RMSE |
| **EXP-17** | Sensitivity| Window-Size Sensitivity | Evaluate $W \in \{30, 50, 70\}$ | Contrastive Loss, Downstream RMSE |
| **EXP-18** | Sensitivity| Embedding-Dimension Sensitivity| Evaluate $d_{emb} \in \{64, 128, 256\}$ | Contrastive Loss, Downstream RMSE |

---

## 13. Baseline Strategy
To establish a rigorous performance benchmark, FailureDNA will be compared against five standard baseline architectures implemented and evaluated under identical engine-level splits and preprocessing protocols:

1. **Statistical Heuristic Baseline (EXP-01)**:
   - Rolling mean, standard deviation, and trend slope of sensor features mapped to RUL via ordinary least squares regression.
2. **Supervised Gradient Boosted Trees (EXP-02)**:
   - XGBoost trained directly on flattened raw temporal windows ($W \times D$).
3. **Supervised Long Short-Term Memory Network (EXP-03)**:
   - 2-layer LSTM ($d_{hidden}=128$, dropout=0.1) trained end-to-end with supervised MSE loss.
4. **Supervised Gated Recurrent Unit Network (EXP-04)**:
   - 2-layer Bidirectional GRU ($d_{hidden}=128$, dropout=0.1) trained end-to-end on raw sensor windows.
5. **Supervised Temporal Transformer (EXP-05)**:
   - Identical Transformer architecture as FailureDNA, but trained directly from scratch using supervised MSE regression on RUL rather than contrastive self-supervised pretraining.

---

## 14. Benchmarking Strategy
To prevent biased or misleading comparisons, we establish a formal benchmarking taxonomy:

### Category A: Locally Reproduced Baselines
Models implemented directly in this codebase (EXP-01 through EXP-05), trained on identical training engines, validated on identical validation splits, and tested on identical held-out units.

### Category B: Published Literature Benchmarks on C-MAPSS FD004
Direct citations from peer-reviewed literature with documented differences in evaluation protocol:
- **Zheng et al. (2017)**: Multilayer CNN for C-MAPSS RUL estimation.
- **Li et al. (2018)**: Deep Convolutional Neural Network with directed cycle clipping ($RUL_{max} = 125$).
- **Ellefsen et al. (2019)**: Semi-supervised Genetic Algorithm + RBM/LSTM.
- **Mo et al. (2021)**: Dual-stream Transformer for C-MAPSS multi-condition prognostics.

### Category C: FailureDNA Proposed Experiments
Evaluated under identical metrics:
- **Root Mean Squared Error (RMSE)**:
  $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2}$$
- **Mean Absolute Error (MAE)**:
  $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |\hat{y}_i - y_i|$$
- **NASA C-MAPSS Asymmetric Scoring Function ($S$)**:
  Penalizes late predictions (where estimated RUL > actual RUL, risking catastrophic failure) much more heavily than early predictions:
  $$d_i = \hat{y}_i - y_i, \quad S = \sum_{i=1}^N \begin{cases} \exp\left(-\frac{d_i}{13}\right) - 1 & \text{for } d_i < 0 \\ \exp\left(\frac{d_i}{10}\right) - 1 & \text{for } d_i \ge 0 \end{cases}$$

---

## 15. Ablation Strategy
To isolate the exact contribution of each architectural component, FailureDNA includes dedicated ablation studies:
1. **Contrastive Objective vs Autoencoding (EXP-15)**:
   - Replace InfoNCE loss with a standard Mean Squared Error (MSE) temporal reconstruction autoencoder. Measures whether contrastive instance discrimination captures degradation dynamics better than pixel/point-wise reconstruction.
2. **Augmentation Decomposition (EXP-16)**:
   - Individually toggle augmentations:
     - Gaussian Jitter only
     - Sensor Channel Masking only
     - Temporal Masking only
     - Magnitude Scaling only
     - All combined
3. **Temporal Window Length ($W$) Sensitivity (EXP-17)**:
   - Test $W \in \{30, 50, 70\}$ cycles to quantify the receptive field requirements for thermodynamic degradation detection.
4. **Embedding Dimensionality ($d_{emb}$) Sensitivity (EXP-18)**:
   - Test $d_{emb} \in \{64, 128, 256\}$ to measure capacity limits and geometric saturation.

---

## 16. Evaluation Metrics
### Representation Learning Metrics
- **NT-Xent Loss**: Convergence of contrastive loss across training and validation splits.
- **Alignment**: Average distance between normalized positive pair views:
  $$\mathcal{L}_{\text{align}} = \mathbb{E}_{(x, x^+)} \left[ \|\hat{z} - \hat{z}^+\|_2^2 \right]$$
- **Uniformity**: Preservation of information across the unit hypersphere:
  $$\mathcal{L}_{\text{uniform}} = \log \mathbb{E}_{x, y \stackrel{i.i.d.}{\sim} p_{data}} \left[ \exp\left(-2\|\hat{z}_x - \hat{z}_y\|_2^2\right) \right]$$

### Clustering & Signature Metrics
- **Silhouette Coefficient**: Degree of cluster separation and cohesion ($[-1, 1]$).
- **Davies-Bouldin Index (DBI)**: Ratio of intra-cluster scatter to inter-cluster separation (lower is superior).
- **Calinski-Harabasz Index (CHI)**: Ratio of between-cluster variance to within-cluster variance.
- **Cluster Stability (Bootstrap ARI)**: Adjusted Rand Index evaluated across bootstrap resamples of the embedding space.
- **Temporal Monotonicity Index**: Fraction of engine trajectories whose cluster progression strictly adheres to the wear order.

### Prognostics & Downstream Metrics
- **RMSE, MAE, $R^2$ Score**: Regression accuracy on held-out RUL.
- **NASA Asymmetric Penalty ($S$)**: Penalizing unsafe over-predictions.

### Anomaly & Early Warning Metrics
- **Area Under ROC (AUROC)**: Discrimination between initial healthy regime ($RUL > 125$) and late degradation regime ($RUL < 30$).
- **Detection Rate (DR)**: Percentage of engines correctly flagged before reaching critical status.
- **False Alarm Rate (FAR)**: Proportion of false degradation alarms raised during the initial healthy phase ($t < 30$ cycles).
- **Mean Lead Time to Failure (MLTF)**: Average operational cycles between initial signature alarm and terminal engine failure.

---

## 17. Visualization Plan (Publication-Quality)
At least 5 of the following 13 publication-quality visualizations will be highlighted in the final IEEE report:

1. **Dataset Overview & Trajectory Lifespans**:
   - Histogram and box plots of engine lifespans (128 to 543 cycles) across training and testing partitions.
2. **Sensor Trajectory Dynamics**:
   - Multi-panel line plots of raw sensor feeds (e.g., T24, T50, P30, Nf, Nc) across engine lifetime, showing operational switching confounded with degradation trends.
3. **Sensor Correlation Heatmap**:
   - Clustered correlation matrix before and after operating-condition normalization.
4. **Operating-Condition Flight Envelope Distribution**:
   - 3D scatter plot of Altitude, Mach number, and TRA showing the 6 discrete operating clusters.
5. **Contrastive Loss Training Dynamics**:
   - Convergence curves for training and validation InfoNCE loss over training epochs, tracking learning rate schedules.
6. **2D UMAP / t-SNE Projection of FailureDNA Embeddings**:
   - Unsupervised 2D manifold projection colored continuously by true engine cycle and true RUL (showing continuous degradation trajectories without labels).
7. **Discovered Failure Signatures & Cluster Boundaries**:
   - UMAP embedding colored by discrete cluster assignments ($S_1, S_2, \dots, S_K$), annotated with centroid prototypes.
8. **Signature Transition Graph**:
   - Directed acyclic graph (DAG) showing state transitions, edge weights representing empirical transition probabilities ($P_{ij}$), and median cycle durations per state.
9. **Actual vs Predicted RUL Scatter Plot**:
   - Downstream regression fidelity comparing FailureDNA+XGBoost, Supervised Transformer, and baselines against the $y = x$ parity line.
10. **RUL Error Distribution & Residual Diagnostics**:
    - Residual distribution histogram ($e = \hat{y} - y$) showing zero-centering, variance, and asymmetric safety bounds.
11. **Anomaly Score Trajectory Over Asset Lifespan**:
    - Mahalanobis distance / Isolation Forest anomaly scores plotted as a function of cycle count, demonstrating early warning inflection points.
12. **SHAP Feature Importance for Downstream Prognostics**:
    - Summary beeswarm plot of TreeExplainer SHAP values identifying the most predictive latent dimensions of the FailureDNA embedding.
13. **Representative Sensor Profiles for Discovered Signatures**:
    - Reconstructed temporal sensor profiles corresponding to the medoids of each discovered failure signature cluster.

---

## 18. Explainability Plan
Machine learning systems in high-consequence industrial operations require rigorous interpretability:

1. **Downstream Feature Attribution (SHAP)**:
   - Apply `shap.TreeExplainer` to downstream XGBoost models operating on FailureDNA representations to quantify how individual latent dimensions influence RUL predictions.
2. **Gradient-Based Temporal Attribution (Integrated Gradients)**:
   - For downstream neural models, compute Integrated Gradients with respect to the input temporal window to identify which historical cycles and sensor channels triggered the representation shift.
3. **Signature Prototypes & Inverse Mapping**:
   - Compute the cluster centroids and nearest empirical medoid windows for each discovered signature.
   - Map centroid representations back to physical engineering units to explain the physical manifestation (e.g., "Signature 4 represents high-pressure compressor exit temperature (T50) surge coupled with fan speed (Nf) droop").
4. **Attention Weight Disclaimer**:
   - **Critical Scientific Notice**: Transformer self-attention weights will *not* be claimed as direct causal explanations, adhering to modern NLP/interpretability literature (e.g., Jain & Wallace, *Attention is not Explanation*). Attention weights will be presented strictly as internal feature-interaction heatmaps.

---

## 19. Deployment Architecture
FailureDNA is engineered for production deployment as a containerized microservice:

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Browser                          │
│     React 18 + TypeScript + Vite + Tailwind/Modern CSS      │
│  (Interactive Engine Telemetry Explorer & Signature Graph)  │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP / REST (JSON)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 FastAPI Production Service                  │
│  ├── Pydantic v2 Request/Response Validation                │
│  ├── CORS Middleware & Request Logging                      │
│  ├── Endpoints:                                             │
│  │   ├── GET  /health                                       │
│  │   ├── POST /api/v1/embed (Window -> FailureDNA Embedding)│
│  │   ├── POST /api/v1/predict/signature (Window -> Cluster) │
│  │   ├── POST /api/v1/predict/rul (Window -> Predicted RUL) │
│  │   ├── POST /api/v1/predict/anomaly (Window -> Score)     │
│  │   └── POST /api/v1/analyze/trajectory (Full Engine Run)  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│             FailureDNA Core Inference Engine                │
│  ├── Frozen Temporal Transformer (TorchScript / PyTorch)    │
│  ├── Preprocessing Scalers & Operating Condition Matcher    │
│  ├── Fitted Signature Clusterer (K-Means / GMM)             │
│  └── Downstream Calibrated Prognostic Heads (XGBoost / MLP) │
└─────────────────────────────────────────────────────────────┘
                               ▲
                               │ Containerized
┌──────────────────────────────┴──────────────────────────────┐
│                    Docker Container                         │
│  - Multi-stage build (Python 3.10 slim)                     │
│  - Cloud Targets: Hugging Face Spaces / AWS App Runner      │
└─────────────────────────────────────────────────────────────┘
```

---

## 20. CAT-II Rubric Mapping
This specification maps directly to the 14 mandatory project requirements:

| CAT-II Requirement | Implementation in FailureDNA | Artifact / Module Path |
| :--- | :--- | :--- |
| **1. Complex Real-World Dataset** | NASA C-MAPSS FD004 (multi-condition, multi-fault turbofan telemetry) | `data/raw/`, `src/failuredna/data/` |
| **2. Advanced Preprocessing** | Operating-condition clustering, regime-wise standardization, non-leaking windowing | `src/failuredna/data/preprocess.py` |
| **3. At Least 5 Visualizations** | 13 publication-quality plots planned; 5+ embedded in IEEE report | `reports/figures/`, `src/failuredna/utils/viz.py` |
| **4. Advanced ML/DL Architecture** | Temporal Transformer Encoder + InfoNCE SSL + Downstream Heads | `src/failuredna/models/`, `src/failuredna/ssl/` |
| **5. Modular End-to-End Pipeline** | Clean separation of data, ssl, clustering, prognostics, api, UI | `src/failuredna/` modular package |
| **6. Experiment Tracking** | MLflow automated tracking (metrics, params, artifacts, weights) | `mlruns/`, `experiments/` |
| **7. Hyperparameter Optimization**| Optuna multi-trial tuning on SSL loss & downstream metrics | `experiments/tune.py` |
| **8. Benchmarking Against Baselines**| 5 distinct baselines + published C-MAPSS FD004 literature benchmarks | `docs/experiments.md`, `experiments/benchmark.py` |
| **9. Comparative & Error Analysis**| Residual analysis, asymmetric score evaluation, lead-time checks | `reports/tables/`, `src/failuredna/prognostics/` |
| **10. FastAPI Production API** | Async FastAPI service with Pydantic v2 schemas and tests | `api/main.py`, `api/schemas.py` |
| **11. Docker Containerization** | Multi-stage Dockerfile and docker-compose orchestration | `Dockerfile`, `docker-compose.yml` |
| **12. Public Cloud Deployment** | Hugging Face Spaces Docker / AWS cloud-ready container | `Dockerfile`, `README.md` |
| **13. IEEE-Style Report (<= 10 pp)**| Comprehensive academic research paper adhering to IEEE format | `reports/ieee_report.md` |
| **14. Reproducible GitHub Repo** | Deterministic seeds, requirements, pyproject.toml, clean git repo | Root repository, `docs/reproducibility.md` |

---

## 21. Risks and Limitations
1. **Operating Condition Confounding**:
   - *Risk*: The Transformer may cluster windows by flight altitude rather than engine wear.
   - *Mitigation*: Rigorous operating-regime normalization and contrastive augmentations that enforce invariance across regime switches.
2. **Representation Collapse in SSL**:
   - *Risk*: The encoder produces trivial constant embeddings for all inputs.
   - *Mitigation*: Temperature-tuned InfoNCE with large batch sizes ($N \ge 64$), pre-LayerNorm, AdamW weight decay, and monitoring of alignment/uniformity metrics.
3. **Temporal Distortion from Augmentations**:
   - *Risk*: Aggressive time warping or jitter may alter the physical thermodynamic properties of turbofan cycles.
   - *Mitigation*: Conservative augmentation thresholds (jitter standard deviation $\sigma \le 0.03$, mask fraction $\le 15\%$).
4. **C-MAPSS Truncated Test Set**:
   - *Risk*: Test trajectories do not run to failure, complicating full-lifespan signature visualization.
   - *Mitigation*: Reserve 15% of the full-lifespan training engines as held-out test trajectories for full life-cycle signature evolution analysis.

---

## 22. Reproducibility Strategy
- **Random Seed Control**: Global seed (default: 42) injected into Python random, NumPy, PyTorch CPU/CUDA, and cuDNN deterministic flags.
- **Environment Management**: Exact dependency pinning via `pyproject.toml` and `requirements.txt`.
- **Configuration-Driven Execution**: All experiments driven by YAML configs in `configs/` (`base.yaml`, `data.yaml`, `model.yaml`, `training.yaml`).
- **One-Command Reproduction**: Automated reproduction scripts via CLI (`python experiments/run_all.py --config configs/experiments.yaml`).

---

## 23. Expected Final Repository Structure
```
FailureDNA/
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
│
├── configs/
│   ├── base.yaml
│   ├── data.yaml
│   ├── model.yaml
│   ├── training.yaml
│   └── experiments.yaml
│
├── data/
│   ├── raw/                  <- Raw C-MAPSS FD004 txt files
│   ├── interim/              <- Cleaned, regime-annotated datasets
│   ├── processed/            <- Engine-split normalized window arrays
│   └── embeddings/           <- Frozen SSL latent feature arrays
│
├── notebooks/                <- Exploratory and analytical notebooks
│
├── src/
│   └── failuredna/
│       ├── __init__.py
│       ├── data/             <- Loader, validation, regime clustering, windowing
│       ├── augmentation/     <- Stochastic temporal transforms (jitter, mask, scale)
│       ├── models/           <- Temporal Transformer, positional encoding, projection head
│       ├── ssl/              <- NT-Xent / InfoNCE contrastive training engine
│       ├── embeddings/       <- Extraction and dimensionality reduction (UMAP/PCA)
│       ├── clustering/       <- K-Means, GMM, HDBSCAN signature discovery
│       ├── anomaly/          <- Mahalanobis distance, Isolation Forest anomaly heads
│       ├── prognostics/      <- Downstream RUL models (XGBoost, MLP, baselines)
│       ├── explainability/   <- SHAP, Integrated Gradients, signature prototypes
│       └── utils/            <- Logging, metrics, seed management, plotting
│
├── experiments/              <- Experiment runner scripts (EXP-01 through EXP-18)
│
├── api/                      <- FastAPI service, routers, and Pydantic schemas
│
├── frontend/                 <- React + TypeScript analytical dashboard
│
├── tests/                    <- Unit, integration, and regression test suites
│
├── mlruns/                   <- Local MLflow tracking store
│
├── reports/
│   ├── figures/              <- Publication-quality generated plots (SVG/PNG)
│   ├── tables/               <- CSV/Markdown comparative performance tables
│   ├── results/              <- Serialized metric outputs
│   └── ieee_report.md        <- Formal IEEE CAT-II research paper
│
└── docs/
    ├── research_specification.md
    ├── experiment_matrix.md
    ├── architecture.md
    ├── methodology.md
    ├── data_dictionary.md
    └── reproducibility.md
```

---

## 24. Proposed IEEE Paper Structure (<= 10 Pages)
1. **Title, Abstract, and Keywords**:
   - 250-word structured abstract summarizing the paradigm shift, methodology, and key quantitative findings.
2. **I. Introduction**:
   - Motivation for self-supervised discovery in complex industrial time-series; limitations of supervised PdM; summary of core contributions.
3. **II. Related Work**:
   - Review of turbofan prognostics on C-MAPSS; self-supervised representation learning in time-series; time-series clustering and regime discovery.
4. **III. Problem Formulation & Dataset Mechanics**:
   - Mathematical formulation of the multivariate temporal setting; detailed aerodynamic analysis of NASA C-MAPSS FD004.
5. **IV. The FailureDNA Framework**:
   - Operating regime normalization; stochastic temporal augmentations; Temporal Transformer architecture; NT-Xent contrastive objective.
6. **V. Experimental Setup & Benchmarking Protocol**:
   - Engine-level splitting protocol; baseline model configurations; evaluation metrics (RMSE, MAE, NASA Asymmetric Score, Silhouette).
7. **VI. Results and Discussion**:
   - Representation quality analysis; signature discovery and characterization; signature transition graphs; downstream RUL regression performance.
8. **VII. Ablation Studies & Sensitivity Analysis**:
   - Influence of contrastive objective, augmentation composition, window size, and embedding dimension.
9. **VIII. Model Explainability & Industrial Deployment**:
   - SHAP feature importance; signature prototype analysis; containerized microservice architecture.
10. **IX. Conclusion & Future Directions**:
    - Summary of research answers; industrial implications; open challenges in fleet transferability.
11. **References**:
    - Comprehensive IEEE-style citations covering foundational literature.
