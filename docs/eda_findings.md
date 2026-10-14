# FailureDNA — Phase 2 Comprehensive Exploratory Data Analysis (EDA) Report
**Target Dataset**: NASA C-MAPSS Sub-Dataset FD004  
**Date**: October 2026  
**Status**: Phase 2 Scientific Validation Complete (Pre-modeling Gate)

---

## 1. Executive Summary & Verification Review

Phase 1 established the end-to-end data pipeline, strict validation schemas, disjoint engine splitting, and unit testing on synthetic fixtures. With the placement and verification of the official NASA C-MAPSS FD004 raw telemetry files under `data/raw/`, Phase 2 executed an empirical exploratory data analysis (EDA) across all 61,249 training observations and 41,214 test observations.

Following an independent scientific audit, feature semantics were verified against authoritative NASA documentation (Saxena et al., PHM 2008), operating regime clustering dynamics were investigated, sensor variance was evaluated on a within-regime basis to audit Simpson's paradox, prospective evaluation boundaries were reinforced, and short-trajectory handling was formally validated.

### 1.1 Phase 1 Raw Data Audit Cross-Check
The statistics recorded in [`reports/results/dataset_audit_FD004.json`](file:///c:/Users/Harish/Downloads/FailureDNA/reports/results/dataset_audit_FD004.json) were verified directly against the raw source files:
- **`train_FD004.txt`**: 61,249 rows, 249 unique engines (`unit_id` 1–249), minimum lifespan 128 cycles, maximum lifespan 543 cycles, median 234 cycles.
- **`test_FD004.txt`**: 41,214 rows, 248 unique engines (`unit_id` 1–248), minimum lifespan 19 cycles, maximum lifespan 486 cycles, median 153.5 cycles.
- **`RUL_FD004.txt`**: Exactly 248 ground-truth entries, non-negative, ranging from 6.0 to 195.0 cycles (median 88.0, mean 86.55).
- **Integrity**: Exactly 0 nulls, 0 NaNs, 0 non-finite values, and 0 duplicate `(unit_id, cycle)` pairs.
- **Discrepancy Log**: **Zero discrepancies** were identified between `dataset_audit_FD004.json` and the raw data.

All synthetic test results remain isolated under [`reports/results/synthetic_audit_report.json`](file:///c:/Users/Harish/Downloads/FailureDNA/reports/results/synthetic_audit_report.json), maintaining strict separation between synthetic testing fixtures and empirical dataset findings.

---

## 2. Trajectory Lifespans & Operational Cycles

Turbofan engines in C-MAPSS undergo degradation until an operational failure threshold is reached. Because FD004 models two simultaneous failure modes (High-Pressure Compressor degradation and Fan degradation) across 6 flight conditions, engine lifespans exhibit substantial operational variance.

| Partition | Engine Count | Total Cycles | Min Lifespan | 10th Pct | 25th Pct (Q1) | Median | Mean | 75th Pct (Q3) | 90th Pct | Max Lifespan | Std Dev |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 249 | 61,249 | 128 | 165.6 | 190.0 | 234.0 | 245.98 | 290.0 | 347.8 | 543 | 73.11 |
| **Test** | 248 | 41,214 | 19 | 58.7 | 102.0 | 153.5 | 166.19 | 218.5 | 285.0 | 486 | 91.61 |

### Key Observations:
1. **Lifespan Span**: Training engine lifespans range from 128 cycles (Engine 70) to 543 cycles (Engine 171). The median engine operates for 234 cycles before failure.
2. **Right-Skewed Distribution**: The training trajectory distribution exhibits positive skewness ($+0.98$), with a long right tail of unusually durable engines (>350 cycles).
3. **Test Truncation**: Official test trajectories are deliberately truncated at random times prior to failure. The minimum test length is **19 cycles** (Engines 141 and 204), representing early operational inspections.

*Reference Visualization*: [`reports/figures/01_trajectory_length_distribution.png`](file:///c:/Users/Harish/Downloads/FailureDNA/reports/figures/01_trajectory_length_distribution.png)

---

## 3. Sensor Dynamics, Feature Semantics, and Within-Regime Variance

### 3.1 Verified Feature Semantics
All 21 numbered sensors and 3 operational settings were audited against authoritative NASA C-MAPSS literature (Saxena et al., *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*, PHM 2008, Table 2). The following corrections have been enforced:
- **$s_{19}$**: Canonical physical description is **Demanded corrected fan speed ($PCNfR\_dmd$) [rpm]**, *not* core speed.
- **$s_{17}$**: Authoritative unit for Bleed Enthalpy ($htBleed$) is **dimensionless (`--`)**, *not* BTU/lbm.
- **$s_{12}$**: Canonical engineering parameter is the **Ratio of fuel flow to static pressure ($W_f / P_{s30}$) [pps/psi]**.
- **Settings**: `setting_1` is Altitude ($0 - 42.0\times 10^3$ ft), `setting_2` is Mach Number ($0.0 - 0.84$), and `setting_3` is Throttle Resolver Angle ($TRA \in \{60, 100\}^\circ$).

### 3.2 Raw Variance vs. Within-Regime Variance
In single-regime datasets (FD001), 6 sensors are constant. In FD004, all 21 sensors exhibit raw pooled variance. However, our scientific audit proves that **raw pooled variance alone does not prove that a sensor is informative of degradation**. When conditioning on operating regime, sensor variance reveals critical distinctions:

| Sensor | Canonical Description | Authoritative Units | Raw Pooled Variance | Mean Within-Regime Variance | Within / Raw Variance Ratio | Coeff. of Variation (CV) | Diagnostic Category |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **$s_9$** | Physical core speed ($N_c$) | rpm | 113,520.17 | 104.91 | $9.24\times 10^{-4}$ | 0.0402 | Active Degradation Indicator |
| **$s_7$** | HPC outlet total pressure ($P_{30}$) | psia | 21,573.80 | 4.37 | $2.02\times 10^{-4}$ | 0.5287 | Active Degradation Indicator |
| **$s_{18}$** | Demanded fan speed ($N_{f\_dmd}$) | rpm | 21,162.25 | **0.0000** | **0.0000** | 0.0673 | Fixed Setpoint Channel (Constant within Regime) |
| **$s_8$** | Physical fan speed ($N_f$) | rpm | 21,126.11 | 2.50 | $1.18\times 10^{-4}$ | 0.0674 | Active Degradation Indicator |
| **$s_{12}$** | Fuel flow to static pressure ratio ($W_f/P_{s30}$) | pps/psi | 19,176.46 | 3.51 | $1.83\times 10^{-4}$ | 0.5243 | Active Degradation Indicator |
| **$s_{13}$** | Corrected fan speed ($NR_f$) | rpm | 16,434.69 | 5.28 | $3.21\times 10^{-4}$ | 0.0549 | Active Degradation Indicator |
| **$s_4$** | LPT outlet total temp ($T_{50}$) | °R | 14,239.07 | 34.02 | $2.39\times 10^{-3}$ | 0.1009 | Strong Degradation Indicator |
| **$s_3$** | HPC outlet total temp ($T_{30}$) | °R | 11,271.56 | 37.07 | $3.29\times 10^{-3}$ | 0.0768 | Strong Degradation Indicator |
| **$s_{14}$** | Corrected core speed ($NR_c$) | rpm | 7,339.44 | 14.16 | $1.93\times 10^{-3}$ | 0.0106 | Active Degradation Indicator |
| **$s_2$** | LPC outlet total temp ($T_{24}$) | °R | 1,394.47 | 3.53 | $2.53\times 10^{-3}$ | 0.0654 | Active Degradation Indicator |
| **$s_{17}$** | Bleed Enthalpy ($htBleed$) | -- | 773.30 | 1.83 | $2.37\times 10^{-3}$ | 0.0818 | Active Degradation Indicator |
| **$s_1$** | Fan inlet total temp ($T_2$) | °R | 698.91 | **0.0000** | **0.0000** | 0.0558 | Fixed Setpoint Channel (Constant within Regime) |
| **$s_{20}$** | HPT coolant bleed flow ($W_{31}$) | lbm/s | 98.73 | 0.0195 | $1.97\times 10^{-4}$ | 0.5369 | Active Degradation Indicator |
| **$s_{21}$** | LPT coolant bleed flow ($W_{32}$) | lbm/s | 35.55 | 0.0071 | $1.99\times 10^{-4}$ | 0.5372 | Active Degradation Indicator |
| **$s_6$** | Bypass duct total pressure ($P_{15}$) | psia | 29.64 | 0.0035 | $1.17\times 10^{-4}$ | 0.5358 | Active Degradation Indicator |
| **$s_{19}$** | Demanded corrected fan speed ($PCNfR\_dmd$) | rpm | 28.83 | **0.0000** | **0.0000** | 0.0549 | Fixed Setpoint Channel (Constant within Regime) |
| **$s_5$** | Fan inlet total pressure ($P_2$) | psia | 13.13 | **0.0000** | **0.0000** | 0.4439 | Fixed Setpoint Channel (Constant within Regime) |
| **$s_{11}$** | Static pressure at HPC outlet ($P_{s30}$) | psia | 10.52 | 0.0215 | $2.04\times 10^{-3}$ | 0.0766 | Active Degradation Indicator |
| **$s_{15}$** | Bypass ratio ($BPR$) | -- | 0.56 | $1.26\times 10^{-4}$ | $2.27\times 10^{-4}$ | 0.0792 | Active Degradation Indicator |
| **$s_{10}$** | Engine pressure ratio ($P_{50}/P_2$) | -- | 0.0163 | **0.0000** | **0.0000** | 0.1171 | Fixed Setpoint Channel (Constant within Regime) |
| **$s_{16}$** | Burner fuel-air ratio ($farB$) | -- | $2.19\times 10^{-5}$ | **0.0000** | **0.0000** | 0.2033 | Fixed Setpoint Channel (Constant within Regime) |

### Key Scientific Findings:
1. **Setpoints vs. Dynamic Wear**: Five sensors ($s_1, s_5, s_{10}, s_{16}, s_{18}, s_{19}$) have **strictly zero variance within individual regimes** ($\sigma^2_{\text{within}} = 0.0$). Their high raw variance (e.g., $698.9$ for $s_1$ and $21,162$ for $s_{18}$) is entirely an artifact of transitions between flight setpoints. They contain zero cycle-by-cycle degradation information.
2. **Environmental Masking**: For sensors tracking real wear (such as $s_7$, HPC outlet pressure), raw variance is $21,573.8$, while within-regime variance is only $4.37$. Over $99.98\%$ of the raw variation is attributable to flight regime changes. This demonstrates why regime-aware normalization is mandatory before modeling degradation.

*Reference Visualization*: [`reports/figures/02_sensor_variance_ranking.png`](file:///c:/Users/Harish/Downloads/FailureDNA/reports/figures/02_sensor_variance_ranking.png)

---

## 4. Sensor Correlations, Redundancy, and Simpson's Paradox

To determine whether sensor correlations represent genuine physical wear coupling or spurious environmental confounding, we compared the **pooled Pearson correlation matrix** against the **mean within-regime correlation matrix**.

### 4.1 Discovery of Simpson's Paradox
The analysis revealed widespread occurrences of Simpson's Paradox across the telemetry channels:

| Sensor Pair | Descriptions | Pooled Pearson $r$ | Within-Regime Mean $r$ | Discrepancy $|\Delta r|$ | Physical Interpretation |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **$s_3$ vs. $s_{15}$** | HPC Outlet Temp vs. Bypass Ratio | **$-0.875$** | **$+0.089$** | **$0.964$** | **Spurious Reversal**: Pooled correlation suggests strong negative coupling, but within any fixed flight regime, the correlation virtually disappears ($+0.09$). The pooled correlation is entirely driven by altitude/Mach shifts. |
| **$s_4$ vs. $s_{15}$** | LPT Outlet Temp vs. Bypass Ratio | **$-0.852$** | **$+0.046$** | **$0.898$** | **Environmental Artifact**: Strong negative correlation in pooled data vanishes inside regimes. |
| **$s_1$ vs. $s_2$** | Fan Inlet Temp vs. LPC Outlet Temp | **$+0.991$** | **$0.000$** | **$0.991$** | **Setpoint Confounding**: Apparent collinearity is an artifact of atmospheric temperature changes across altitudes. Within regimes, $s_1$ is constant ($r=0$). |
| **$s_2$ vs. $s_7$** | LPC Outlet Temp vs. HPC Outlet Pressure | **$+0.923$** | **$+0.142$** | **$0.781$** | **Atmospheric Scaling**: Pooled correlation reflects high ambient pressure/temperature at sea level vs low at 42k ft, not compressor wear. |

### 4.2 Genuine Physical Degradation Clusters
Within individual operating regimes, true physical degradation manifests in targeted channels:
1. **Compressor-Turbine Thermal Coupling**: Within-regime correlation between HPC outlet temperature ($s_3$) and LPT outlet temperature ($s_4$) remains moderately positive ($r \approx +0.48$), reflecting genuine thermal deterioration across the gas path.
2. **Spool Speed Association**: Physical fan speed ($s_8$) and core speed ($s_9$) maintain positive within-regime correlation ($r \approx +0.52$).

*Reference Visualization*: [`reports/figures/03_sensor_correlation_matrix.png`](file:///c:/Users/Harish/Downloads/FailureDNA/reports/figures/03_sensor_correlation_matrix.png) (Dual-panel: Panel A Pooled vs. Panel B Within-Regime).

---

## 5. Operating Regimes: Empirical Audit and Diagnostic Evaluation

### 5.1 Discrete Nature of Operating Settings
An independent empirical audit was conducted to investigate why the six-cluster Silhouette score reaches **0.9971** and Davies–Bouldin index drops to **0.0045**.

Inspection of `train_FD004.txt` across all 61,249 observations reveals that the 3 setting variables (`setting_1`: Altitude, `setting_2`: Mach, `setting_3`: TRA) do **not** form a continuous flight envelope. Instead, they represent **six discrete flight simulation setpoints** perturbed only by minor floating-point simulation noise:

| Regime ID | Cluster Center (Orig Units) | Within-Cluster Std Dev ($\sigma$) | Range in Cluster | Train Observations | Proportion |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **0** | Alt: $42.00$ kft, Mach: $0.8400$, TRA: $100.0^\circ$ | $\sigma_{\text{alt}} = 0.0029, \sigma_{\text{mach}} = 0.00065, \sigma_{\text{TRA}} = 0.0$ | Alt: $[41.99, 42.01]$, Mach: $[0.839, 0.841]$, TRA: $[100, 100]$ | 15,395 | $25.14\%$ |
| **1** | Alt: $10.00$ kft, Mach: $0.2504$, TRA: $100.0^\circ$ | $\sigma_{\text{alt}} = 0.0029, \sigma_{\text{mach}} = 0.00065, \sigma_{\text{TRA}} = 0.0$ | Alt: $[9.99, 10.01]$, Mach: $[0.249, 0.251]$, TRA: $[100, 100]$ | 9,224 | $15.06\%$ |
| **2** | Alt: $25.00$ kft, Mach: $0.6204$, TRA: $60.0^\circ$ | $\sigma_{\text{alt}} = 0.0029, \sigma_{\text{mach}} = 0.00065, \sigma_{\text{TRA}} = 0.0$ | Alt: $[24.99, 25.01]$, Mach: $[0.619, 0.621]$, TRA: $[60, 60]$ | 9,139 | $14.92\%$ |
| **3** | Alt: $20.00$ kft, Mach: $0.7004$, TRA: $100.0^\circ$ | $\sigma_{\text{alt}} = 0.0029, \sigma_{\text{mach}} = 0.00065, \sigma_{\text{TRA}} = 0.0$ | Alt: $[19.99, 20.01]$, Mach: $[0.699, 0.701]$, TRA: $[100, 100]$ | 9,091 | $14.84\%$ |
| **4** | Alt: $35.00$ kft, Mach: $0.8404$, TRA: $100.0^\circ$ | $\sigma_{\text{alt}} = 0.0029, \sigma_{\text{mach}} = 0.00065, \sigma_{\text{TRA}} = 0.0$ | Alt: $[34.99, 35.01]$, Mach: $[0.839, 0.841]$, TRA: $[100, 100]$ | 9,162 | $14.96\%$ |
| **5** | Alt: $0.00$ kft, Mach: $0.0004$, TRA: $100.0^\circ$ | $\sigma_{\text{alt}} = 0.0028, \sigma_{\text{mach}} = 0.00065, \sigma_{\text{TRA}} = 0.0$ | Alt: $[-0.01, 0.01]$, Mach: $[-0.001, 0.001]$, TRA: $[100, 100]$ | 9,238 | $15.08\%$ |

### 5.2 Mathematical Explanation for Near-Perfect Metric Values
In standardized feature space, the pairwise Euclidean distance between cluster centroids is $D_{\text{inter}} \approx 1.5 - 3.8$, while the within-cluster dispersion is $\sigma_{\text{within}} \approx 0.0003$.
The ratio $D_{\text{inter}} / \sigma_{\text{within}} > 3,000$. Consequently:
- Every observation is thousands of standard deviations closer to its own cluster centroid than to any neighboring centroid.
- The Silhouette score of **0.9971** and Davies-Bouldin index of **0.0045** are the direct mathematical outcome of clustering **virtually discrete point masses**.
- **Important Scientific Limit**: These metrics validate that the 3 operational settings fall into six discrete setpoints. They **do not** validate degradation discovery, nor do they prove that six regimes are optimal for wear modeling.

### 5.3 GMM BIC Evaluation
Diagnostics were evaluated across $k \in [2, 10]$:

| $k$ | Silhouette Score ($\uparrow$) | Davies-Bouldin Index ($\downarrow$) | Calinski-Harabasz Index ($\uparrow$) | KMeans Inertia ($\downarrow$) | GMM BIC ($\downarrow$) | Size Balance Ratio |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 2 | 0.6157 | 0.6086 | $7.77\times 10^4$ | 54,672.1 | -229,125.2 | 0.175 |
| 3 | 0.8019 | 0.3099 | $2.63\times 10^5$ | 21,390.8 | -998,066.4 | 0.370 |
| 4 | 0.8503 | 0.2197 | $5.67\times 10^5$ | 8,912.4 | -1,362,033.0 | 0.370 |
| 5 | 0.9301 | 0.1071 | $2.17\times 10^6$ | 1,288.5 | -1,570,363.4 | 0.370 |
| **6** | **0.9971** | **0.0045** | **$8.43\times 10^9$** | **0.2669** | **-1,827,423.6** | **0.5905** |
| 7 | 0.9367 | 0.1131 | $8.84\times 10^9$ | 0.2123 | -1,832,542.4 | 0.4436 |
| 8 | 0.9011 | 0.1938 | $9.00\times 10^9$ | 0.1788 | -1,835,646.0 | 0.2687 |
| 9 | 0.8662 | 0.2522 | $9.67\times 10^9$ | 0.1457 | -1,838,412.2 | 0.2451 |
| 10 | 0.8303 | 0.3046 | $1.12\times 10^{10}$ | 0.1125 | -1,841,367.1 | 0.2653 |

- **BIC Monotonicity**: Because $N = 61,249$, Gaussian Mixture BIC decreases monotonically across the entire range ($k=2$ to $k=10$). We explicitly clarify that **a monotonically improving BIC is not an elbow**.
- **Rate of Change ($\Delta\text{BIC}$)**: The marginal improvement $\Delta\text{BIC}$ drops by **$98\%$** after $k=6$ (from $>250,000$ per component for $k \le 6$ to $<5,000$ for $k > 6$). BIC alone does not select $k=6$; rather, $k=6$ is supported by the discrete setpoints, simulation physics, and the peak in Silhouette score.

### 5.4 Clustering Stability and Degeneracy Audit
- **Seed Invariance**: KMeans was fitted across 5 random seeds (`[0, 1, 7, 42, 123]`). The mean pairwise Adjusted Rand Index is **$1.0000$**.
- **Subsampling Stability**: Subsampling $80\%$ of training observations yielded an Adjusted Rand Index of **$1.0000$**.
- **Degeneracy Check**: Cluster sizes range from 9,091 to 15,395 observations ($14.8\%$ to $25.1\%$). There are **zero small, empty, or degenerate clusters**.

*Reference Visualizations*:
- Settings Distribution: [`reports/figures/04_operating_settings_distributions.png`](file:///c:/Users/Harish/Downloads/FailureDNA/reports/figures/04_operating_settings_distributions.png)
- Clustering Diagnostics: [`reports/figures/05_operating_regime_clustering_diagnostics.png`](file:///c:/Users/Harish/Downloads/FailureDNA/reports/figures/05_operating_regime_clustering_diagnostics.png)

---

## 6. Regime Profiles & Environmental Confounding

The empirical discovery of 6 operating regimes enables characterization of environmental confounding:

| Regime ID | Altitude ($10^3$ ft) | Mach Number | TRA (°) | Train Rows | Proportion | Inferred Physical Flight Condition |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | $42.00$ | $0.840$ | $100.0$ | 15,395 | $25.14\%$ | High-Altitude Subsonic Cruise (Ceiling) |
| **1** | $10.00$ | $0.251$ | $100.0$ | 9,224 | $15.06\%$ | Low-Altitude Climb / Departure |
| **2** | $25.00$ | $0.621$ | $60.0$ | 9,139 | $14.92\%$ | Medium-Altitude Loiter / Low Throttle |
| **3** | $20.00$ | $0.701$ | $100.0$ | 9,091 | $14.84\%$ | Medium-Altitude High-Speed Cruise |
| **4** | $35.00$ | $0.841$ | $100.0$ | 9,162 | $14.96\%$ | Standard Cruise (Subsonic En Route) |
| **5** | $0.00$ | $0.001$ | $100.0$ | 9,238 | $15.08\%$ | Sea-Level Takeoff / Static Test |

### Environmental Confounding Mechanism:
In turbofan engines, ambient atmospheric conditions exert dominant control over thermodynamic states:
- **Pressure Swings**: Ambient inlet pressure ($s_5, P_2$) drops from **$14.62$ psia** at Sea Level (Regime 5) to **$3.91$ psia** at 42k ft (Regime 0). HPC outlet pressure ($s_7, P_{30}$) drops proportionally from **$553.4$ psia** to **$138.9$ psia**.
- **Temperature Swings**: Fan inlet temperature ($s_1, T_2$) drops from **$518.67$ °R** (Sea Level) to **$445.00$ °R** (42k ft).
- **The Confounding Threat**: When an engine transitions from Regime 5 to Regime 0 within a single flight cycle, $P_{30}$ drops by **$414.5$ psia**. Physical mechanical degradation, by comparison, induces a gradual drift of only **$5 - 15$ psia** over 200 cycles. **Without regime normalization, operational flight condition transitions will swamp and obscure subtle failure signatures.**

*Reference Visualization*: [`reports/figures/06_regime_wise_sensor_profiles.png`](file:///c:/Users/Harish/Downloads/FailureDNA/reports/figures/06_regime_wise_sensor_profiles.png)

---

## 7. Retrospective Sensor Evolution Over Normalized Lifespan

> [!NOTE]
> **Methodological Disclaimer: Retrospective Analysis Only**  
> Visualizations displaying sensor evolution over normalized engine life ($t / T_{\text{max}} \in [0, 100\%]$) are **strictly retrospective**. Complete engine lifespan $T_{\text{max}}$ is never known during real-time prospective in-flight monitoring. This visualization is intended solely for historical post-mortem understanding of degradative trends. Future Phase 3 modeling pipelines will never utilize $T_{\text{max}}$, future sensor observations, or future-derived quantities.

By tracking sensor trajectories over normalized engine life ($t / T_{\text{max}} \in [0, 1]$), distinct degradative trends emerge beneath the flight condition noise:

1. **HPC Outlet Temperature ($s_3, T_{30}$)**:
   - Early life ($0\% - 50\%$): Fluctuates within regime-specific bands with zero monotonic drift.
   - Late life ($75\% - 100\%$): Increases monotonically across all regimes by $+20$ to $+35$ °R as compressor aerodynamic efficiency deteriorates, requiring higher turbine inlet temperatures to maintain demanded thrust.
2. **LPT Outlet Temperature ($s_4, T_{50}$)**:
   - Exhibits an upward drift of $+15$ to $+25$ °R in the final $20\%$ of engine lifespan.
3. **Corrected Core Speed ($s_{14}, NR_c$)**:
   - Drifts downward slightly in late life due to blade rub and friction losses before climbing sharply at terminal failure.
4. **Static Pressure ($s_{11}, P_{s30}$)**:
   - Exhibits a consistent downward drift of $-0.8$ to $-1.5$ psia in the terminal degradation phase.

*Reference Visualization*: [`reports/figures/07_sensor_evolution_normalized_life.png`](file:///c:/Users/Harish/Downloads/FailureDNA/reports/figures/07_sensor_evolution_normalized_life.png)

---

## 8. Exploratory Analysis of Derived RUL Horizons

> [!IMPORTANT]
> **Evaluation Boundary: Exploratory Stratification Only**  
> Stratifying sensor readings by Remaining Useful Life horizons ($>200$ down to $\le 25$ cycles) is an **exploratory analysis**. Remaining Useful Life (RUL) will strictly **never** enter self-supervised representation learning, feature extraction, or operating-regime clustering. The official test set ([`test_FD004.txt`](file:///c:/Users/Harish/Downloads/FailureDNA/data/raw/test_FD004.txt) and [`RUL_FD004.txt`](file:///c:/Users/Harish/Downloads/FailureDNA/data/raw/RUL_FD004.txt)) remains completely quarantined for final benchmark evaluation.

### Findings:
- In the "Near Failure" horizon ($\le 25$ cycles), sensor dispersion expands significantly. Temperature sensors ($s_3, s_4$) show clear upward median shifts, and fuel flow ratio ($s_{12}$) exhibits heightened volatility.
- The non-linear acceleration of degradation as RUL approaches 0 supports the piecewise-linear RUL clipping heuristic commonly applied in RUL estimation (e.g., $RUL_{\text{max}} = 125$ or $130$).

*Reference Visualization*: [`reports/figures/08_sensor_behavior_by_rul_interval.png`](file:///c:/Users/Harish/Downloads/FailureDNA/reports/figures/08_sensor_behavior_by_rul_interval.png)

---

## 9. Official Test Trajectory Truncation & Short Trajectory Validation

### 9.1 Test Truncation Audit
In [`test_FD004.txt`](file:///c:/Users/Harish/Downloads/FailureDNA/data/raw/test_FD004.txt), engine runs are intentionally truncated prior to failure. Analysis of trajectory lengths revealed that **20 of the 248 official test engines ($8.1\%$) have fewer than 50 recorded cycles**:

| Trajectory Length ($T_{\text{max}}$) | Affected Engine IDs | Engine Count | % of Test Engines |
| :---: | :--- | :---: | :---: |
| **$19$ cycles** | Engine 141, Engine 204 | 2 | $0.81\%$ |
| **$20 - 24$ cycles** | Engine 10 (23), Engine 19 (24), Engine 28 (20), Engine 125 (21), Engine 229 (20) | 5 | $2.02\%$ |
| **$25 - 29$ cycles** | Engine 156 (28), Engine 164 (29), Engine 239 (28), Engine 246 (29) | 4 | $1.61\%$ |
| **$30 - 39$ cycles** | Engine 46 (37), Engine 59 (37), Engine 144 (34), Engine 169 (34), Engine 225 (39) | 5 | $2.02\%$ |
| **$40 - 49$ cycles** | Engine 76 (48), Engine 170 (48), Engine 186 (47), Engine 241 (42) | 4 | $1.61\%$ |
| **Total $< 50$ cycles** | **All 20 engines listed above** | **20** | **$8.06\%$** |

### 9.2 Validation of `extract_last_window`
Official C-MAPSS scoring requires generating RUL predictions for **all 248 test engines**. Silently dropping short engines is prohibited.
We audited the actual implementation of [`extract_last_window`](file:///c:/Users/Harish/Downloads/FailureDNA/src/failuredna/data/windowing.py):
- **Completeness**: Executed on `test_FD004.txt`, the function extracts exactly **248 windows of shape $(W, D)$** for all 248 engines under both $W=50$ and $W=30$. **Zero engines are dropped**.
- **Determinism**: Every window is strictly aligned to the final recorded cycle of each test trajectory.
- **Edge Padding Mechanics**: For an engine with $T < W$, exactly $W - T$ copies of the earliest observed cycle ($t=1$) are prepended to the window.
- **Padding Assumption Disclaimer**: Left-padding with edge repetition is an **architectural modeling assumption** required to format inputs into fixed tensor dimensions $(B, W, D)$ without future leakage. It is **not** a physical reconstruction of unobserved history.
- **Sequence Mask Support**: The windowing pipeline supports binary validity masks ($M \in \{0, 1\}^W$), allowing attention layers in Phase 3 to mask out padded early positions if desired.

### 9.3 Validation Plan: $W=30$ vs. $W=50$
To select the optimal context window length without leaking or tuning on the official test set:
1. **Quarantine Test Set**: Official test data and ground-truth RUL remain quarantined.
2. **Internal Validation Comparison**: Candidate window lengths $W=30$ and $W=50$ will be evaluated strictly on the 37 engines of the internal validation partition.
3. **Decision Criteria**: The winning window size will be selected based on validation RMSE, NASA scoring function, and attention convergence stability on the internal validation split.

*Reference Visualization*: [`reports/figures/09_short_test_trajectories_and_padding_analysis.png`](file:///c:/Users/Harish/Downloads/FailureDNA/reports/figures/09_short_test_trajectories_and_padding_analysis.png)

---

## 10. Summary of Generated Figures

All 9 publication-grade figures have been generated at 300 DPI and saved under [`reports/figures/`](file:///c:/Users/Harish/Downloads/FailureDNA/reports/figures):

| # | Figure Filename | Core Demonstration / Scientific Insight |
| :---: | :--- | :--- |
| **01** | [`01_trajectory_length_distribution.png`](file:///c:/Users/Harish/Downloads/FailureDNA/reports/figures/01_trajectory_length_distribution.png) | Trajectory length histograms (Train vs Truncated Test), showing test truncation and the 20 engines shorter than $W=50$. |
| **02** | [`02_sensor_variance_ranking.png`](file:///c:/Users/Harish/Downloads/FailureDNA/reports/figures/02_sensor_variance_ranking.png) | Log-scale variance ranking demonstrating raw vs within-regime variance across all 21 sensors. |
| **03** | [`03_sensor_correlation_matrix.png`](file:///c:/Users/Harish/Downloads/FailureDNA/reports/figures/03_sensor_correlation_matrix.png) | Dual-panel correlation comparison (Panel A Pooled vs Panel B Mean Within-Regime), proving Simpson's paradox. |
| **04** | [`04_operating_settings_distributions.png`](file:///c:/Users/Harish/Downloads/FailureDNA/reports/figures/04_operating_settings_distributions.png) | Tri-panel distribution of Altitude, Mach, and TRA illustrating the discrete flight setpoints. |
| **05** | [`05_operating_regime_clustering_diagnostics.png`](file:///c:/Users/Harish/Downloads/FailureDNA/reports/figures/05_operating_regime_clustering_diagnostics.png) | Empirical diagnostics (Silhouette, Davies-Bouldin, Calinski-Harabasz, GMM BIC) across $k=2..10$. |
| **06** | [`06_regime_wise_sensor_profiles.png`](file:///c:/Users/Harish/Downloads/FailureDNA/reports/figures/06_regime_wise_sensor_profiles.png) | Heatmap comparing normalized sensor means across the 6 regimes, proving environmental confounding. |
| **07** | [`07_sensor_evolution_normalized_life.png`](file:///c:/Users/Harish/Downloads/FailureDNA/reports/figures/07_sensor_evolution_normalized_life.png) | Retrospective sensor drift over $0\% - 100\%$ normalized life for $s_3, s_4, s_7, s_{11}$ (with explicit disclaimer). |
| **08** | [`08_sensor_behavior_by_rul_interval.png`](file:///c:/Users/Harish/Downloads/FailureDNA/reports/figures/08_sensor_behavior_by_rul_interval.png) | Exploratory boxplots of key sensor readings stratified across Remaining Useful Life horizons. |
| **09** | [`09_short_test_trajectories_and_padding_analysis.png`](file:///c:/Users/Harish/Downloads/FailureDNA/reports/figures/09_short_test_trajectories_and_padding_analysis.png) | Ranked test trajectory lengths and left-padding overhead comparison for candidate windows $W=50$ vs $W=30$. |

---

## 11. Leakage Safeguards & Integrity Audits

To ensure absolute research integrity:
1. **Operating Regime Discovery**: Clustered exclusively on `setting_1, setting_2, setting_3` of the training set. No RUL, cycle index, or failure labels were accessible to the clustering model.
2. **Transformation Isolation**: Scaling and clustering estimators were fitted on training data; validation and test sets were transformed using frozen parameters.
3. **Internal Validation Reservation**: The official test set ([`test_FD004.txt`](file:///c:/Users/Harish/Downloads/FailureDNA/data/raw/test_FD004.txt) and [`RUL_FD004.txt`](file:///c:/Users/Harish/Downloads/FailureDNA/data/raw/RUL_FD004.txt)) is strictly quarantined. All modeling choices rely solely on the disjoint internal validation partition (37 engines).
4. **Causal Temporal Integrity**: Left-padding repeats strictly the earliest observed cycle ($t=1$), ensuring zero future information is injected.

---

## 12. Verification & Test Suite Status

The test suite was audited and expanded to verify within-regime variance and correlation calculations alongside all pipeline components:

```powershell
$env:PYTHONPATH="src"; python -m pytest -v
```

**Results**:
```
tests/test_data.py (13 tests) ............. PASSED
tests/test_eda.py (4 tests) ............... PASSED
tests/test_regimes.py (8 tests) ........... PASSED
tests/test_windowing.py (5 tests) ......... PASSED
tests/test_windowing_padding.py (4 tests) . PASSED

============================= 34 passed in 5.26s ==============================
```
**Outcome**: 34 of 34 unit tests passing ($100\%$ success rate). Zero regressions.
