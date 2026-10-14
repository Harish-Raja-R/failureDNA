# FailureDNA — Phase 5.1: Official Benchmark Integrity Audit Report

**Audit Target**: NASA C-MAPSS FD004 Official Benchmark Evaluation (Phase 5)  
**Evaluated Cohort**: 248 Official Test Engines (`test_FD004.txt`, `RUL_FD004.txt`)  
**Audit Date**: October 14, 2026  
**Auditor**: Antigravity Autonomous Research Core  
**Final Verdict**: **PASS WITH QUALIFICATIONS**

---

## Executive Summary

This audit constitutes a formal, reproducible verification of the completed Phase 5 official benchmark evaluation of the FailureDNA prognostic framework on NASA C-MAPSS FD004. In accordance with strict audit constraints, no models were retrained, no hyperparameters were altered, no thresholds were modified, and the original Phase 5 benchmark artifacts were preserved in place without modification.

### Core Audit Outcomes:
1. **Frozen-Artifact & Data-Leakage Audit**: **PASS**. All representation encoders evaluated on the official test set were verified to have been loaded from frozen checkpoints trained during Phases 3 and 4 strictly on the 174 training engines. All preprocessors and downstream regression probes were fitted exclusively on the 174 training engines. Official test telemetry and ground-truth RUL targets were accessed solely at final inference time.
2. **NASA Metric Implementation Audit**: **PASS**. The mathematical formulation, residual sign convention, and asymmetric penalty logic were audited against hand-calculated toy cases with 100% exact numerical agreement. The ~87.5M NASA score for the Matched Random Encoder was traced mathematically to unclipped linear extrapolation on out-of-distribution embeddings for 5 short engines, not a software defect.
3. **Paired Uncertainty Analysis**: **QUALIFICATION REQUIRED**. Paired engine-level bootstrapping ($B=2,000$ resamples) reveals that **XGBoost achieves a statistically significant advantage over FailureDNA (Regime-Aware) in MAE** ($\Delta \text{MAE} = +2.8571$ cycles, $95\%$ CI: $[+1.000, +4.614]$, $p = 0.0010$). While point estimates favor FailureDNA over XGBoost in NASA score (8,049.31 vs 8,186.42; $\Delta = -137.12$), the paired $95\%$ confidence interval spans zero ($[-3,759.86, +3,088.25]$, $p = 0.9220$). The claim that FailureDNA demonstrates a statistically verified prognostic advantage over tuned gradient-boosted trees on FD004 is rejected.
4. **Signature & NMI Audit**: **QUALIFICATION REQUIRED**. The reported NMI of $0.000143$ was independently recomputed across $29,436$ sliding test windows ($W=50, s=1$). The sampling units are sliding windows with $98\%$ temporal autocorrelation and unequal trajectory weighting. While confirming low empirical association between discrete latent clusters and instantaneous regimes, this metric does not prove causal separation, complete confounding removal, or verified physical fault identification.
5. **Configuration & Table Audit**: **PASS**. The naming distinction between "Regime-Aware + Ridge" (24 input features) and "Regime-Conditioned + Ridge" (30 input features) was confirmed. All reported metrics across all 11 model configurations match the saved prediction file to $<10^{-4}$ precision.

---

## 1. Frozen-Artifact and Data-Leakage Audit

### 1.1 Artifact Manifest

Table 1 details the provenance, cryptographic SHA-256 hashes, file sizes, configurations, and split training lineage for every representation model evaluated in Phase 5.

#### Table 1: Official Frozen Artifact Manifest

| Checkpoint Name | Relative Path | SHA-256 Hash | Size (Bytes) | Training Split Lineage | Loaded vs. Fitted Status | Input / Latent Dim |
| :--- | :--- | :--- | :---: | :--- | :---: | :---: |
| `failuredna_regime_aware` | `models/checkpoints/failuredna_regime_aware/best_failuredna_model.pt` | `d18b6cd0abc2fadf12be18f6f1962707f3626e0a5054970147f50eb3620eb248` | 2,369,441 | Phase 3/4 Train Split (174 engines, `seed=42`) | **LOADED FROZEN** | $24 \to 128$ |
| `failuredna_pooled` | `models/checkpoints/failuredna_pooled/best_failuredna_model.pt` | `1e93ab9cbf31185838020c03c557b593f018b2b4aa2420d5b8c2732d288878ef` | 2,369,441 | Phase 4 Train Split (174 engines, `seed=42`) | **LOADED FROZEN** | $24 \to 128$ |
| `failuredna_regime_conditioned` | `models/checkpoints/failuredna_regime_conditioned/best_failuredna_model.pt` | `67b379d2d31a1cabe99cf5809cd5bcf7d606f3224ea9566da716f9c75c851fbb` | 2,372,513 | Phase 4 Train Split (174 engines, `seed=42`) | **LOADED FROZEN** | $30 \to 128$ |
| `failuredna_w30` | `models/checkpoints/failuredna_w30/best_failuredna_model.pt` | `8aa450d5612542b0827d7f9d6b1d295ab7e06014175e1f7f0c779c68d431b791` | 2,369,441 | Phase 4 Train Split (174 engines, `seed=42`) | **LOADED FROZEN** | $24 \to 128$ |
| `Matched Random Encoder` | *In-memory only* (`torch.manual_seed(42)`) | N/A (Dynamic Untrained) | N/A | None (Untrained Negative Control) | **INITIALIZED FRESH** | $24 \to 128$ |

### 1.2 Downstream Predictor and Baseline Fitting Protocol

During Phase 5 execution (`experiments/run_phase5_official_benchmark.py`), the following components were fitted or loaded:
- **Preprocessors**:
  - `RegimeAwarePreprocessor(n_regimes=6)`: `fit_transform` was called strictly on the 174 training engines (`train_raw`). Only `transform` was called on validation (`val_raw`), internal-test (`itest_raw`), and official test (`raw_test`).
  - `CMAPSSPreprocessor(scaler_type='robust')`: Fitted strictly on `train_raw`.
- **Downstream Probes on Frozen Embeddings**:
  - Linear Ridge probes (`StandardScaler + Ridge(alpha=10.0)`): Fitted strictly on `(emb_train, y_train)` derived from 174 training engines ($N=33,966$ windows with capped RUL $125.0$).
  - Random Forest and XGBoost probes: Fitted strictly on training set embeddings and targets.
- **Tabular Baselines**:
  - `OperatingConditionOnlyBaseline`: Fitted on training window settings mean.
  - `LinearWindowBaseline(alpha=1.0)`: Fitted on 96 handcrafted rolling statistics from 174 training engines.
  - `RandomForestBaseline(n_estimators=100, max_depth=10)`: Fitted on training window statistics.
  - `XGBoostBaseline(n_estimators=150, max_depth=6, lr=0.05)`: Fitted on training window statistics; early stopping monitored strictly on the 37 validation engines (`X_val_tab`, `y_val`).
- **Signature Discovery**:
  - `FailureSignatureClusterer(n_clusters=4)`: `fit` called strictly on `emb_tr_ra`. Predicted cluster assignments were evaluated on test telemetry post-hoc.

### 1.3 Data Leakage Audit Findings

1. **Test Telemetry Withholding**: Telemetry records in `test_FD004.txt` were never referenced during the fitting of KMeans regime estimators, feature standardizers, contrastive encoder pretraining, baseline fitting, or probe fitting.
2. **Ground-Truth RUL Withholding**: Ground-truth target values from `RUL_FD004.txt` ($N=248$) were accessed exclusively in Section 10 of `experiments/run_phase5_official_benchmark.py` through `compute_regression_metrics(y_test_official, y_pred)`.
3. **Hyperparameter Selection**: Ridge regularization ($\alpha=10.0$), XGBoost hyperparameters, window length ($W=50$), and RUL capping ($125.0$) were frozen identically to Phase 3/4 validation protocols prior to test evaluation.

### 1.4 Scope and Limitations of the Preflight Audit

The Phase 5 preflight audit verified that the on-disk checkpoint files matched known cryptographic checksums and that data paths were accessible.
> [!IMPORTANT]
> **Preflight Checksum Limitation**: SHA-256 verification of checkpoints confirms only that stored weights were not altered on disk after saving. It does **not** prove zero data leakage by itself. Zero data leakage is established by **code provenance and architectural isolation**: verifying that preprocessors execute `fit` exclusively on the training partition, that dataset loaders maintain disjoint engine sets, and that test labels are absent from loss functions and parameter tuning routines.

---

## 2. NASA Score Implementation Audit

### 2.1 Formula and Residual Sign Convention

The NASA asymmetric scoring function standard for C-MAPSS benchmarks is implemented in `src/failuredna/evaluation/metrics.py`:
$$d_i = \hat{y}_i - y_i^*$$
$$s_i = \begin{cases} \exp\left(-\frac{d_i}{13}\right) - 1 & \text{if } d_i < 0 \quad (\text{early prediction: underestimating RUL}) \\ \exp\left(\frac{d_i}{10}\right) - 1 & \text{if } d_i \ge 0 \quad (\text{late prediction: overestimating RUL, dangerous}) \end{cases}$$
$$S = \sum_{i=1}^N s_i$$

### 2.2 Independent Verification Against Hand-Calculated Toy Examples

To verify the piecewise residual logic and sign convention, three synthetic cases were calculated by hand and compared against the output of `nasa_scoring_function`:

#### Table 2: NASA Scoring Function Toy Case Verification

| Scenario | True RUL ($y^*$) | Pred RUL ($\hat{y}$) | Residual ($d = \hat{y} - y^*$) | Hand-Calculated Formula & Exact Math | Code Output | Audit Match |
| :--- | :---: | :---: | :---: | :--- | :---: | :---: |
| **Early Prediction** | $100.0$ | $80.0$ | $-20.0$ | $\exp(-(-20)/13) - 1 = \exp(1.538462) - 1 = 3.657419$ | $3.657419$ | **EXACT MATCH** |
| **Late Prediction** | $100.0$ | $120.0$ | $+20.0$ | $\exp(20/10) - 1 = \exp(2.0) - 1 = 6.389056$ | $6.389056$ | **EXACT MATCH** |
| **Exact Prediction** | $50.0$ | $50.0$ | $0.0$ | $\exp(0/10) - 1 = 1.0 - 1 = 0.000000$ | $0.000000$ | **EXACT MATCH** |
| **Composite Vector** | $[100, 100, 50]$ | $[80, 120, 50]$ | $[-20, +20, 0]$ | $3.657419 + 6.389056 + 0.0 = 10.046476$ | $10.046476$ | **EXACT MATCH** |

The asymmetry ratio for equal error magnitude $|d|=20$ is $6.389 / 3.657 \approx 1.747$. Late predictions (overestimating remaining life, risking catastrophic in-flight failure) are penalized $75\%$ more severely than early predictions of the same magnitude.

### 2.3 Mathematical Decomposition of the Random Encoder's 87.5M NASA Score

In Phase 5, the Matched Random Encoder achieved an RMSE of $48.86$, yet produced an astronomical NASA score of $87,465,483.74$. Table 3 isolates the top 10 contributing test engines to this score.

#### Table 3: Top 10 Engine Contributors to Random Encoder NASA Score

| Test Unit ID | Trajectory Length ($T_{\text{obs}}$) | True RUL ($y^*$) | Pred RUL ($\hat{y}$) | Residual ($d = \hat{y} - y^*$) | Engine NASA Penalty ($s_i$) | % of Fleet Total Score | Cumulative % |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **125** | 21 cycles | $184.0$ | $-44.02$ | $-228.02$ | $41,441,830.41$ | $47.38\%$ | $47.38\%$ |
| **10** | 23 cycles | $162.0$ | $-59.19$ | $-221.19$ | $24,504,727.87$ | $28.02\%$ | $75.40\%$ |
| **204** | 19 cycles | $151.0$ | $-63.55$ | $-214.55$ | $14,708,709.82$ | $16.82\%$ | $92.22\%$ |
| **19** | 24 cycles | $194.0$ | $-4.19$ | $-198.19$ | $4,179,264.44$ | $4.78\%$ | $97.00\%$ |
| **141** | 19 cycles | $130.0$ | $-56.73$ | $-186.73$ | $1,730,191.02$ | $1.98\%$ | **98.97%** |
| **28** | 20 cycles | $124.0$ | $-46.23$ | $-170.23$ | $486,317.03$ | $0.56\%$ | $99.53\%$ |
| **246** | 29 cycles | $194.0$ | $+41.86$ | $-152.14$ | $120,967.82$ | $0.14\%$ | $99.67\%$ |
| **164** | 29 cycles | $174.0$ | $+23.46$ | $-150.54$ | $106,959.66$ | $0.12\%$ | $99.79\%$ |
| **239** | 28 cycles | $166.0$ | $+18.46$ | $-147.54$ | $84,868.24$ | $0.10\%$ | $99.89\%$ |
| **229** | 20 cycles | $120.0$ | $-19.37$ | $-139.37$ | $45,285.66$ | $0.05\%$ | $99.94\%$ |

#### Root Cause Analysis:
1. **Short Trajectories with Early Censoring**: All top 10 engines are short trajectories ($T_{\text{obs}} \in [19, 29]$ cycles) requiring edge padding to reach the $W=50$ window threshold. These engines were stopped very early in life and have high true RUL ($120$ to $194$ cycles).
2. **Out-of-Distribution Extrapolation**: Untrained random projections from a randomly initialized transformer encoder map repeated-edge padded inputs to vectors far outside the manifold of full 50-cycle sequences.
3. **Linear Extrapolation into Negative Values**: The downstream linear Ridge model, fitted on positive training targets without output clipping, extrapolated these out-of-distribution embeddings into negative predictions (e.g., $\hat{y} = -63.55$ for Unit 204).
4. **Exponential Penalty Amplification**: For Unit 125, $d = -228.02$. The penalty is:
   $$s_{125} = \exp\left(-\frac{-228.02}{13}\right) - 1 = \exp(17.54) - 1 = 41,441,830.41$$
   This single engine accounts for **$47.38\%$** of the fleet score. The top 3 engines account for **$92.22\%$**, and the top 5 engines account for **$98.97\%$**.
5. **Contrast with FailureDNA**: For those identical 5 short engines, FailureDNA (Regime-Aware) predicted $148.51$ (Unit 10), $168.12$ (Unit 19), $148.53$ (Unit 125), and $181.71$ (Unit 204). Because contrastive temporal pretraining organizes the latent space into an invariant degradation trajectory, edge-padded early sequences mapped smoothly to the healthy cluster, preventing negative extrapolation and producing an overall NASA score of $8,049.31$.

**Audit Conclusion on NASA Metric**: The scoring function implementation is completely correct. The 87.5M score is a mathematically verified property of the exponential penalty acting upon negative linear extrapolation on short engines.

---

## 3. Paired Uncertainty Analysis

To determine whether FailureDNA provides a statistically reliable advantage over baseline architectures on the official test set, we executed paired engine-level bootstrapping ($B=2,000$ resamples, fixed random seed `42`).

### 3.1 Paired Engine-Level Bootstrap Results

In Table 4, the difference is defined as $\Delta = \text{Metric}(\text{Model A}) - \text{Metric}(\text{Model B})$. For error metrics (MAE, RMSE, NASA score), $\Delta < 0$ denotes superior performance by Model A.

#### Table 4: Paired Bootstrap Comparison on Official Test Engines ($N=248, B=2000$)

| Comparison ($\text{Model A vs. Model B}$) | Metric | Point Diff ($\Delta$) | 95% Bootstrap CI | Empirical $p$-value | Statistical Inference |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **FailureDNA (Regime-Aware) vs. XGBoost** | **MAE** | $\mathbf{+2.8571}$ | $[+1.000, +4.614]$ | $\mathbf{0.0010}$ | **XGBoost advantage is statistically significant** ($p < 0.01$) |
| | **RMSE** | $+1.2827$ | $[-1.147, +3.588]$ | $0.2860$ | Not statistically significant (CI spans 0) |
| | **NASA** | $-137.12$ | $[-3759.86, +3088.25]$ | $0.9220$ | Not statistically significant (CI spans 0) |
| **FailureDNA (Regime-Conditioned) vs. XGBoost** | **MAE** | $\mathbf{+1.8756}$ | $[+0.244, +3.427]$ | $\mathbf{0.0270}$ | **XGBoost advantage is statistically significant** ($p < 0.05$) |
| | **RMSE** | $+1.0816$ | $[-0.808, +2.814]$ | $0.2540$ | Not statistically significant (CI spans 0) |
| | **NASA** | $-495.43$ | $[-3459.38, +1917.91]$ | $0.7090$ | Not statistically significant (CI spans 0) |
| **FailureDNA (Regime-Aware) vs. Random Forest** | **MAE** | $\mathbf{+2.1385}$ | $[+0.196, +3.931]$ | $\mathbf{0.0340}$ | **Random Forest advantage is statistically significant** ($p < 0.05$) |
| | **RMSE** | $+0.5922$ | $[-1.805, +2.825]$ | $0.6250$ | Not statistically significant (CI spans 0) |
| | **NASA** | $-333.46$ | $[-4276.87, +2865.81]$ | $0.8560$ | Not statistically significant (CI spans 0) |
| **FailureDNA (Regime-Aware) vs. Linear Ridge** | **MAE** | $-1.5522$ | $[-3.148, +0.029]$ | $0.0560$ | Marginal; not significant at $\alpha = 0.05$ |
| | **RMSE** | $-1.7414$ | $[-3.512, +0.149]$ | $0.0650$ | Marginal; not significant at $\alpha = 0.05$ |
| | **NASA** | $-2038.87$ | $[-6039.50, +1337.33]$ | $0.2670$ | Not statistically significant (CI spans 0) |
| **FailureDNA (Regime-Conditioned) vs. FailureDNA (Regime-Aware)** | **MAE** | $-0.9815$ | $[-2.362, +0.376]$ | $0.1560$ | Not statistically significant (CI spans 0) |
| | **RMSE** | $-0.2011$ | $[-1.581, +1.272]$ | $0.7790$ | Not statistically significant (CI spans 0) |
| | **NASA** | $-358.32$ | $[-2038.13, +1255.87]$ | $0.6820$ | Not statistically significant (CI spans 0) |

### 3.2 Scientific Interpretation and Reporting Corrections

1. **Rejection of Overclaiming**: Prior reporting in Phase 5 suggested that FailureDNA "outperformed" XGBoost based on point estimates in the NASA score. The paired uncertainty audit disproves this claim:
   - For NASA score, the paired difference between FailureDNA (Regime-Aware) and XGBoost has a $95\%$ CI of $[-3,759.86, +3,088.25]$ ($p = 0.9220$). Zero lies in the center of the bootstrap distribution. The observed point difference ($-137.12$) is well within random fleet sampling variation.
   - For MAE, XGBoost is **statistically significantly superior** to FailureDNA (Regime-Aware) by $2.86$ cycles ($p = 0.0010$) and to FailureDNA (Regime-Conditioned) by $1.88$ cycles ($p = 0.0270$).
   - For RMSE, differences are not statistically distinguishable from zero.
2. **Defensible Scientific Positioning**: FailureDNA does not establish prognostic superiority over gradient-boosted trees on FD004. Rather, FailureDNA matches gradient-boosted trees in RMSE ($30.36$ vs $29.08$) and NASA score ($8,049$ vs $8,186$) using a simple linear readout probe on self-supervised embeddings, while providing structured temporal representations and unsupervised degradation state discovery that tabular baselines cannot offer.

---

## 4. Signature and NMI Audit

### 4.1 Independent NMI Recomputation

Using the saved test telemetry embeddings from the frozen `failuredna_regime_aware` model and the fitted KMeans clusterers, the mutual information and normalized mutual information were recalculated:

- **Total Test Telemetry Windows Analyzed**: $N = 29,436$ sliding windows ($W=50, s=1$) across all 248 test engines.
- **Mutual Information**: $I(\text{Signature}; \text{Regime}) = 0.00020461\text{ nats}$ ($0.0002952\text{ bits}$).
- **Signature Marginal Shannon Entropy**: $H(\text{Signature}) = 1.098495\text{ nats}$ (across 4 clusters).
- **Regime Marginal Shannon Entropy**: $H(\text{Regime}) = 1.768782\text{ nats}$ (across 6 operating regimes).
- **Normalized Mutual Information (Arithmetic)**:
  $$\text{NMI}_{\text{arith}} = \frac{2 \cdot I(\text{Signature}; \text{Regime})}{H(\text{Signature}) + H(\text{Regime})} = \frac{2 \cdot 0.00020461}{1.098495 + 1.768782} = \mathbf{0.00014272}$$
- **Normalized Mutual Information (Geometric)**:
  $$\text{NMI}_{\text{geom}} = \frac{I(\text{Signature}; \text{Regime})}{\sqrt{H(\text{Signature}) \cdot H(\text{Regime})}} = \mathbf{0.00014679}$$

The reported NMI in `phase5_official_benchmark_summary.json` ($0.00014272$) is exactly reproduced.

### 4.2 Sampling Units and Methodological Limitations

1. **Sampling Unit Definition**: The sampling units for this calculation are **sliding temporal windows** of length 50 cycles with stride 1.
2. **Severe Temporal Autocorrelation**: Because windows have a stride of 1, consecutive windows for the same engine share $49/50$ ($98\%$) of their sensor observations. These observations are strongly non-independent, which deflates standard errors and makes empirical entropy estimates sensitive to sequence length.
3. **Trajectory Length Imbalance**: In `test_FD004.txt`, engine trajectory lengths range from $19$ cycles (Unit 204) to $303$ cycles (Unit 223). A long-lived engine contributes up to 254 windows, whereas a short engine contributes only 1 window. The resulting contingency table is heavily weighted by engines operating in healthy states.
4. **Instantaneous vs. Aggregated Regimes**: The regime label corresponds strictly to the cluster ID at the terminal cycle (`end_cycle`) of the window, while the latent embedding aggregates 50 cycles of normalized dynamics across multiple regime shifts.

### 4.3 Mandatory Replacement of Unsupported Claims

> [!CAUTION]
> **Claim Correction**: The assertion in earlier draft text that the test NMI of $0.000143$ "proves that FailureDNA removed confounding" or "proves physical failure mode separation" is unscientific and must be replaced.

**Calibrated Replacement Statement**:
> "On official test telemetry ($29,436$ sliding windows), the empirical normalized mutual information between discrete latent signature cluster IDs and operating regime cluster IDs is very low ($\text{NMI} \approx 1.43 \times 10^{-4}$). This confirms that contrastive representation pretraining does not trivially organize its latent geometry by operating regime. However, because sliding windows are strongly autocorrelated, trajectory lengths vary widely across engines, and true physical failure modes (e.g., Fan vs. HPC degradation) are unobserved in the ground-truth data, low NMI cannot be interpreted as formal proof of confounding removal, causal independence, or verified physical fault isolation."

---

## 5. Configuration-Name and Table Audit

### 5.1 Disambiguation of Model Configurations

During Phase 5, two closely named representation ablations were evaluated:
1. **`FailureDNA (Regime-Aware) + Ridge`**:
   - Preprocessor: `RegimeAwarePreprocessor(n_regimes=6, include_settings=True, include_regime_onehot=False)`.
   - Feature Dimension: $24$ ($3$ operating settings + $21$ regime-normalized sensor channels).
   - Checkpoint: `models/checkpoints/failuredna_regime_aware/best_failuredna_model.pt`.
   - Prediction Column: `pred_failuredna_ridge` in `reports/results/phase5_engine_predictions.csv`.
   - Model Key in Summary: `failuredna_regime_aware_ridge`.
2. **`FailureDNA (Regime-Conditioned) + Ridge`**:
   - Preprocessor: `RegimeAwarePreprocessor(n_regimes=6, include_settings=True, include_regime_onehot=True)`.
   - Feature Dimension: $30$ ($3$ operating settings + $21$ regime-normalized sensors + $6$ one-hot regime indicator channels).
   - Checkpoint: `models/checkpoints/failuredna_regime_conditioned/best_failuredna_model.pt`.
   - Prediction Column: `pred_failuredna_regime_conditioned_ridge`.
   - Model Key in Summary: `failuredna_regime_conditioned_ridge`.

### 5.2 Metric Reproduction Cross-Reference

Table 5 confirms that recalculating MAE, RMSE, $R^2$, and NASA score directly from the 248 saved engine predictions in `reports/results/phase5_engine_predictions.csv` reproduces the values reported in `phase5_model_comparison.csv` and `phase5_official_benchmark_summary.json` with zero error.

#### Table 5: Comprehensive Numerical Reproduction Audit

| Model Configuration | Saved Prediction Column | CSV Reported MAE | Recalculated MAE | CSV Reported RMSE | Recalculated RMSE | CSV Reported NASA Score | Recalculated NASA Score | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FailureDNA (Regime-Conditioned) + Ridge** | `pred_failuredna_regime_conditioned_ridge` | $22.4389$ | $22.4389$ | $30.1612$ | $30.1612$ | $7,690.99$ | $7,690.99$ | **PERFECT MATCH** |
| **FailureDNA (Regime-Aware) + Ridge** | `pred_failuredna_ridge` | $23.4204$ | $23.4204$ | $30.3623$ | $30.3623$ | $8,049.31$ | $8,049.31$ | **PERFECT MATCH** |
| **XGBoost Baseline** | `pred_xgboost` | $20.5633$ | $20.5633$ | $29.0796$ | $29.0796$ | $8,186.42$ | $8,186.42$ | **PERFECT MATCH** |
| **Random Forest Baseline** | `pred_random_forest` | $21.2819$ | $21.2819$ | $29.7701$ | $29.7701$ | $8,382.76$ | $8,382.76$ | **PERFECT MATCH** |
| **Linear / Ridge Baseline** | `pred_linear_ridge` | $24.9726$ | $24.9726$ | $32.1038$ | $32.1038$ | $10,088.17$ | $10,088.17$ | **PERFECT MATCH** |
| **FailureDNA (W=30) + Ridge** | `pred_failuredna_w30_ridge` | $26.0953$ | $26.0953$ | $32.9133$ | $32.9133$ | $10,209.21$ | $10,209.21$ | **PERFECT MATCH** |
| **FailureDNA + XGBoost Probe** | `pred_failuredna_xgboost` | $26.2606$ | $26.2606$ | $33.6535$ | $33.6535$ | $12,777.35$ | $12,777.35$ | **PERFECT MATCH** |
| **FailureDNA + RF Probe** | `pred_failuredna_random_forest` | $28.0127$ | $28.0127$ | $35.1669$ | $35.1669$ | $18,640.09$ | $18,640.09$ | **PERFECT MATCH** |
| **FailureDNA (Pooled) + Ridge** | `pred_failuredna_pooled_ridge` | $33.4109$ | $33.4109$ | $41.4596$ | $41.4596$ | $35,253.88$ | $35,253.88$ | **PERFECT MATCH** |
| **Operating Condition Only** | `pred_operating_condition_only` | $46.4206$ | $46.4206$ | $54.5485$ | $54.5485$ | $139,041.98$ | $139,041.98$ | **PERFECT MATCH** |
| **Matched Random Encoder** | `pred_random_encoder_ridge` | $29.8761$ | $29.8761$ | $48.8564$ | $48.8564$ | $87,465,483.74$ | $87,465,483.74$ | **PERFECT MATCH** |

---

## 6. Audit Verdict and Actionable Deliverables

### Final Audit Determination: **PASS WITH QUALIFICATIONS**

The Phase 5 evaluation satisfies all core criteria for data hygiene, frozen checkpoint isolation, mathematical correctness of evaluation metrics, and complete numerical reproducibility. However, acceptance of the benchmark requires calibrated reporting adjustments.

### Summary of Passing Checks:
- [x] Zero data leakage verified across preprocessing, encoder weights, downstream regressors, and hyperparameter selections.
- [x] All frozen checkpoint SHA-256 hashes verified on disk against preflight audit records.
- [x] NASA asymmetric scoring function matches documented formulas and hand-calculated toy cases.
- [x] Extreme Random Encoder NASA score ($87.5\text{M}$) mathematically accounted for by short-engine extrapolation.
- [x] Complete consistency across CSV, JSON, Markdown tables, and raw engine predictions.

### Mandatory Qualifications to Be Enforced in Versioned Report (`v2`):
1. **Acknowledge XGBoost Superiority in MAE**: Replace statements claiming overall FailureDNA superiority with clear documentation that XGBoost achieves a statistically significant advantage in MAE ($p = 0.0010$).
2. **Qualify NASA Score Advantage**: State explicitly that while FailureDNA achieved a lower point estimate on NASA asymmetric penalty than XGBoost and Random Forest, the difference is not statistically significant under paired bootstrapping ($p = 0.9220$).
3. **Calibrate NMI Claims**: Reframe the test NMI of $0.000143$ as low empirical association across sliding windows, removing all claims of proven causal separation or physical fault identification.
4. **Preserve Versioning**: Leave `phase5_official_benchmark_report.md` intact and issue `phase5_official_benchmark_report_v2.md` containing these calibrated findings.
