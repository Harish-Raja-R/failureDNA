# NASA C-MAPSS Sub-Dataset FD004: Comprehensive Data Dictionary

This document provides the authoritative telemetry specification and physical definitions for the **NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) Sub-Dataset FD004**.

---

## 1. Dataset Overview & Operational Scope

- **Simulation Engine Model**: 90,000 lb thrust class turbofan engine simulator.
- **Sub-Dataset Identifier**: FD004.
- **Flight Operating Envelope**: 6 distinct flight conditions across the full flight envelope.
- **Fault Regimes Simulated**: 2 distinct failure modes:
  1. High-Pressure Compressor (HPC) degradation (flow capacity reduction and efficiency drop).
  2. Fan degradation (efficiency drop).
- **Nominal Lifespan Range**: 128 to 543 flight cycles per engine.

---

## 2. File Schema & Layout

Raw files (`train_FD004.txt`, `test_FD004.txt`) are whitespace-delimited without header rows, structured across 26 numeric columns:

| Column Index (1-based) | Variable Name | Data Type | Physical Description (NASA PHM08 Table 2) | Symbol | Units |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **1** | `unit_id` | Integer | Unique engine identifier ($1, 2, \dots$) | `unit_id` | None |
| **2** | `cycle` | Integer | Operational time index ($1, 2, \dots, T$) | `cycle` | Flight Cycles |
| **3** | `setting_1` | Float | Operational Setting 1: Flight Altitude | Altitude | $10^3$ ft ($0 - 42$) |
| **4** | `setting_2` | Float | Operational Setting 2: Flight Mach Number | Mach | Dimensionless ($0.0 - 0.84$) |
| **5** | `setting_3` | Float | Operational Setting 3: Throttle Resolver Angle | TRA | Degrees ($20 - 100$) |
| **6** | `s_1` | Float | Total temperature at fan inlet | $T_2$ | $^\circ\text{R}$ |
| **7** | `s_2` | Float | Total temperature at LPC outlet | $T_{24}$ | $^\circ\text{R}$ |
| **8** | `s_3` | Float | Total temperature at HPC outlet | $T_{30}$ | $^\circ\text{R}$ |
| **9** | `s_4` | Float | Total temperature at LPT outlet | $T_{50}$ | $^\circ\text{R}$ |
| **10** | `s_5` | Float | Pressure at fan inlet | $P_2$ | psia |
| **11** | `s_6` | Float | Total pressure in bypass duct | $P_{15}$ | psia |
| **12** | `s_7` | Float | Total pressure at HPC outlet | $P_{30}$ | psia |
| **13** | `s_8` | Float | Physical fan rotational speed | $N_f$ | rpm |
| **14** | `s_9` | Float | Physical core rotational speed | $N_c$ | rpm |
| **15** | `s_10` | Float | Engine pressure ratio | $epr$ ($P_{50} / P_2$) | Dimensionless |
| **16** | `s_11` | Float | Static pressure at HPC outlet | $P_{s30}$ | psia |
| **17** | `s_12` | Float | Ratio of fuel flow to static pressure | $\phi$ ($W_f / P_{s30}$) | pps / psi |
| **18** | `s_13` | Float | Corrected fan rotational speed | $NR_f$ | rpm |
| **19** | `s_14` | Float | Corrected core rotational speed | $NR_c$ | rpm |
| **20** | `s_15` | Float | Bypass ratio | $BPR$ | Dimensionless |
| **21** | `s_16` | Float | Burner fuel-air ratio | $farB$ | Dimensionless |
| **22** | `s_17` | Float | Bleed enthalpy | $htBleed$ | Dimensionless ($--$) |
| **23** | `s_18` | Float | Demanded fan rotational speed | $N_{f\_dmd}$ | rpm |
| **24** | `s_19` | Float | Demanded corrected fan speed | $PCNfR\_dmd$ | rpm |
| **25** | `s_20` | Float | High-pressure turbine coolant bleed flow | $W_{31}$ | lbm / s |
| **26** | `s_21` | Float | Low-pressure turbine coolant bleed flow | $W_{32}$ | lbm / s |

---

## 3. RUL Ground-Truth Schema (`RUL_FD004.txt`)

Contains 1 column:
- `rul` (Float / Integer): Remaining Useful Life in operational flight cycles for each test engine at the final recorded truncated observation cycle. Row $i$ maps to test engine $i$.

---

## 4. Operating Conditions & Environmental Confounding

The 3 operating settings (`setting_1`, `setting_2`, `setting_3`) define 6 distinct flight regimes:
1. Sea-level climb / takeoff
2. Low-altitude cruise
3. Medium-altitude cruise
4. High-altitude subsonic cruise
5. Descent / deceleration
6. Low-throttle loiter

In raw sensor telemetry, operating regime transitions induce sudden shifts that can mask gradual mechanical wear. Preprocessing isolates physical degradation through robust scaling and regime normalization.

---

## 5. Sensor Selection & Variance Audit Protocol

In single-condition datasets (FD001), sensors 1, 5, 10, 16, 18, and 19 are constant. In FD004, operating condition shifts introduce variance across settings. The `CMAPSSPreprocessor` evaluates variance exclusively on the training partition:
$$\sigma_{s_j}^2 = \frac{1}{N_{train} - 1} \sum_{k=1}^{N_{train}} (s_{j, k} - \mu_{s_j})^2$$
Any channel satisfying $\sigma_{s_j}^2 \le 1.0\times 10^{-5}$ is pruned automatically to eliminate zero-information inputs.

---

## 6. Discrete Operating Setpoints & Clustering Structure

Empirical analysis demonstrates that the 3 operational settings are not continuously distributed across the flight envelope, but are concentrated at **6 discrete operating condition setpoints** with tiny sensor perturbation noise ($\sigma_{\text{within}} \approx 0.0029$ in Altitude, $0.00065$ in Mach, $0.0$ in TRA):
1. **Regime 0**: Altitude $\approx 42.0$k ft, Mach $\approx 0.840$, TRA $= 100^\circ$ (High-Altitude Cruise Ceiling)
2. **Regime 1**: Altitude $\approx 10.0$k ft, Mach $\approx 0.251$, TRA $= 100^\circ$ (Low-Altitude Departure / Climb)
3. **Regime 2**: Altitude $\approx 25.0$k ft, Mach $\approx 0.621$, TRA $= 60^\circ$ (Medium-Altitude Low-Throttle Loiter)
4. **Regime 3**: Altitude $\approx 20.0$k ft, Mach $\approx 0.701$, TRA $= 100^\circ$ (Medium-Altitude High-Speed Cruise)
5. **Regime 4**: Altitude $\approx 35.0$k ft, Mach $\approx 0.841$, TRA $= 100^\circ$ (Subsonic En Route Cruise)
6. **Regime 5**: Altitude $\approx 0.0$k ft, Mach $\approx 0.001$, TRA $= 100^\circ$ (Sea-Level Takeoff)

Because inter-cluster distances in standardized space exceed intra-cluster dispersion by several orders of magnitude ($D_{\text{inter}} / \sigma_{\text{within}} > 3,000$), clustering algorithms achieve a near-perfect Silhouette score ($0.9971$) and Davies-Bouldin index ($0.0045$).

---

## 7. Pooled vs. Within-Regime Variance & Simpson's Paradox

Crucially, **a non-zero raw pooled variance does not imply that a sensor channel is informative regarding physical degradation**:
- Sensors $s_1$ (Fan inlet temp), $s_5$ (Fan inlet pressure), $s_{18}$ (Demanded fan speed), and $s_{19}$ (Demanded corrected fan speed) have **$0.0$ within-regime variance**. Their raw pooled variance is purely a mathematical artifact of switching discrete operating regimes.
- Pooled sensor correlations can reflect Simpson's paradox: for instance, $s_3$ and $s_{15}$ have a pooled correlation of $r = -0.875$ across the dataset, but an average within-regime correlation of only $r = +0.089$. In Phase 3, representation models must decouple regime transitions from internal degradation.
