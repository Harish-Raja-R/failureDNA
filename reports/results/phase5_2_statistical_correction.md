# FailureDNA — Phase 5.2: Statistical Inference and Reporting Correction Addendum

**Document Version**: 1.0 (Phase 5.2 Formal Statistical Addendum)  
**Parent Reports**: 
- [`reports/results/phase5_1_integrity_audit.md`](file:///c:/Users/Harish/Downloads/FailureDNA/reports/results/phase5_1_integrity_audit.md)
- [`reports/results/phase5_official_benchmark_report_v2.md`](file:///c:/Users/Harish/Downloads/FailureDNA/reports/results/phase5_official_benchmark_report_v2.md)  
**Machine-Readable Summary**: [`reports/results/phase5_2_statistical_summary.json`](file:///c:/Users/Harish/Downloads/FailureDNA/reports/results/phase5_2_statistical_summary.json)  
**Adjusted Bootstrap Data**: [`reports/results/phase5_2_adjusted_bootstrap.csv`](file:///c:/Users/Harish/Downloads/FailureDNA/reports/results/phase5_2_adjusted_bootstrap.csv)  
**Execution Timestamp**: 2026-10-14  
**Audit Scope**: Statistical inference correction, family-wise multiple comparisons adjustment, bootstrap verification, and reporting calibration.

---

## 1. Overview and Purpose

This addendum reviews the Phase 5.1 integrity audit and implements necessary statistical corrections without retraining models, changing prediction files, tuning hyperparameters, or altering decision thresholds. 

While Phase 5.1 correctly identified that overlapping individual confidence intervals do not substitute for formal hypothesis testing and computed paired bootstrap differences, it reported nominal (unadjusted) $p$-values and characterized the NASA score confidence interval as "covering zero symmetrically." This addendum provides:
1. **Multiple-Comparisons Adjustments**: Formal specification of hypothesis testing families and Holm's step-down procedure (Holm-Bonferroni) to control the Family-Wise Error Rate (FWER).
2. **Paired Bootstrap Protocol Verification**: Methodological audit confirming that resampling was conducted strictly at the engine level ($N=248$) with fixed seed and proper difference conventions.
3. **NASA Score Uncertainty Diagnostics**: Investigation of the heavy-tailed, exponential penalty distribution that causes wide confidence intervals and unstable uncertainty estimates.
4. **Report Wording & Taxonomy Calibration**: Replacement of imprecise language (replacing "covers zero symmetrically" with "the confidence interval includes zero"), retraction of unsupported significance claims, and a rigorous taxonomy separating point rankings, nominal evidence, and multiplicity-adjusted evidence.

---

## 2. Multiple-Comparisons Analysis and Family-Wise Error Rate Control

### 2.1 Specification of Hypothesis Families

When evaluating multiple models across multiple prognostic metrics, performing multiple unadjusted significance tests inflates the family-wise type I error rate. To establish rigorous statistical inference, we define three explicit comparison families:

1. **Per-Metric Comparison Families ($m = 5$ tests per metric)**:
   - Evaluates the 5 model contrasts separately within each distinct metric family: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the NASA Asymmetric Score.
2. **Primary Baseline Benchmark Family ($m = 3$ tests per metric)**:
   - Pre-specified, confirmatory comparisons evaluating the primary proposed architecture, **FailureDNA (Regime-Aware)**, against the three established external baselines: **XGBoost**, **Random Forest**, and **Linear Ridge**.
3. **Omnibus Comparison Family ($m = 15$ tests)**:
   - Comprehensive exploratory family comprising all 5 model contrasts across all 3 performance metrics simultaneously.

### 2.2 Holm's Step-Down Procedure (Holm-Bonferroni)

To control the Family-Wise Error Rate at $\alpha = 0.05$ without the excessive conservatism of a single-step Bonferroni correction, we apply **Holm's step-down procedure**. 

For a family of $m$ hypotheses with ordered raw $p$-values $p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$, the adjusted $p$-values are computed sequentially:
$$\tilde{p}_{(k)} = \max_{j \le k} \left( \min\left(1.0, (m - j + 1) \cdot p_{(j)}\right) \right)$$
A hypothesis $H_{(k)}$ is rejected at FWER $\alpha = 0.05$ if and only if $\tilde{p}_{(k)} < 0.05$.

### 2.3 Paired Bootstrap Hypothesis Tests with Raw and Adjusted $p$-Values

Table 1 presents the full paired-bootstrap analysis ($B=2,000$ resamples, seed `42`), reporting point differences ($\Delta = \text{Model A} - \text{Model B}$), $95\%$ percentile confidence intervals, raw nominal $p$-values, and Holm-adjusted $p$-values across all defined families.

#### Table 1: Comprehensive Paired Bootstrap Statistical Inference ($N=248, B=2000$)

| Contrast ($\text{Model A vs. Model B}$) | Metric | Point Diff ($\Delta$) | 95% Bootstrap CI | Raw Nominal $p$-val | Holm Adjusted (Per-Metric $m=5$) | Holm Adjusted (Primary $m=3$) | Holm Adjusted (Omnibus $m=15$) | FWER Status ($\alpha = 0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **FailureDNA (Regime-Aware) vs. XGBoost** | **MAE** | $\mathbf{+2.8571}$ | $[+1.000, +4.614]$ | $\mathbf{0.0010}$ | $\mathbf{0.0050}$ | $\mathbf{0.0030}$ | $\mathbf{0.0150}$ | **STATISTICALLY SIGNIFICANT** |
| | **RMSE** | $+1.2827$ | $[-1.147, +3.588]$ | $0.2860$ | $1.0000$ | $0.5720$ | $1.0000$ | Not Significant (CI includes 0) |
| | **NASA** | $-137.12$ | $[-3759.86, +3088.25]$ | $0.9220$ | $1.0000$ | $1.0000$ | $1.0000$ | Not Significant (CI includes 0) |
| **FailureDNA (Regime-Conditioned) vs. XGBoost** | **MAE** | $+1.8756$ | $[+0.244, +3.427]$ | $0.0270$ | $0.1080$ | — | $0.3780$ | **NOT SIGNIFICANT AFTER ADJ.** |
| | **RMSE** | $+1.0816$ | $[-0.808, +2.814]$ | $0.2540$ | $1.0000$ | — | $1.0000$ | Not Significant (CI includes 0) |
| | **NASA** | $-495.43$ | $[-3459.38, +1917.91]$ | $0.7090$ | $1.0000$ | — | $1.0000$ | Not Significant (CI includes 0) |
| **FailureDNA (Regime-Aware) vs. Random Forest** | **MAE** | $+2.1385$ | $[+0.196, +3.931]$ | $0.0340$ | $0.1080$ | $0.0680$ | $0.4420$ | **NOT SIGNIFICANT AFTER ADJ.** |
| | **RMSE** | $+0.5922$ | $[-1.805, +2.825]$ | $0.6250$ | $1.0000$ | $0.6250$ | $1.0000$ | Not Significant (CI includes 0) |
| | **NASA** | $-333.46$ | $[-4276.87, +2865.81]$ | $0.8560$ | $1.0000$ | $1.0000$ | $1.0000$ | Not Significant (CI includes 0) |
| **FailureDNA (Regime-Aware) vs. Linear Ridge** | **MAE** | $-1.5522$ | $[-3.148, +0.029]$ | $0.0560$ | $0.1120$ | $0.0680$ | $0.6720$ | Not Significant (CI includes 0) |
| | **RMSE** | $-1.7414$ | $[-3.512, +0.149]$ | $0.0650$ | $0.3250$ | $0.1950$ | $0.7150$ | Not Significant (CI includes 0) |
| | **NASA** | $-2038.87$ | $[-6039.50, +1337.33]$ | $0.2670$ | $1.0000$ | $0.8010$ | $1.0000$ | Not Significant (CI includes 0) |
| **FailureDNA (Regime-Conditioned) vs. FailureDNA (Regime-Aware)** | **MAE** | $-0.9815$ | $[-2.362, +0.376]$ | $0.1560$ | $0.1560$ | — | $1.0000$ | Not Significant (CI includes 0) |
| | **RMSE** | $-0.2011$ | $[-1.581, +1.272]$ | $0.7790$ | $1.0000$ | — | $1.0000$ | Not Significant (CI includes 0) |
| | **NASA** | $-358.32$ | $[-2038.13, +1255.87]$ | $0.6820$ | $1.0000$ | — | $1.0000$ | Not Significant (CI includes 0) |

*Note on Testing Type*: Comparisons involving FailureDNA (Regime-Aware) vs. XGBoost and Linear Ridge represent confirmatory baseline benchmarking. Contrasts involving FailureDNA (Regime-Conditioned) and Random Forest are exploratory ablation analyses.

### 2.4 Critical Multiplicity Findings

1. **Only One Test Survives Family-Wise Error Rate Control**:
   - The superiority of **XGBoost over FailureDNA (Regime-Aware) in MAE** ($\Delta = +2.8571$ cycles) remains statistically significant across all family definitions: raw $p = 0.0010$, Holm metric $\tilde{p} = 0.0050$, Holm primary $\tilde{p} = 0.0030$, and Holm omnibus $\tilde{p} = 0.0150$.
   - The confidence interval $[+1.000, +4.614]$ strictly excludes zero.
2. **Retraction of Random Forest and Regime-Conditioned Significance Claims**:
   - In Phase 5.1, the difference between FailureDNA (Regime-Aware) and Random Forest in MAE was described as statistically significant based on an unadjusted $p$-value ($p = 0.0340$).
   - Under Holm's step-down adjustment, this contrast yields $\tilde{p} = 0.1080$ ($m=5$) and $\tilde{p} = 0.0680$ ($m=3$).
   - Similarly, FailureDNA (Regime-Conditioned) vs. XGBoost in MAE had an unadjusted $p = 0.0270$, which adjusts to $\tilde{p} = 0.1080$ ($m=5$).
   - **Correction**: Because both adjusted $p$-values exceed $\alpha = 0.05$, neither comparison meets the standard for statistical significance under multiple testing control. Claims of statistically significant differences for these contrasts are formally retracted.

---

## 3. Paired Bootstrap Verification

The paired bootstrap execution was verified against the codebase and saved predictions:

1. **Difference Convention**: $\Delta = \text{Metric}(\text{Model A}) - \text{Metric}(\text{Model B})$. For loss/error metrics (MAE, RMSE, NASA score), $\Delta > 0$ indicates that Model B achieved lower error than Model A.
2. **Resampling Unit**: Resampling was performed strictly at the **engine level** ($N=248$ clusters). Each bootstrap sample resamples 248 engine units with replacement. Sliding temporal windows within each engine trajectory were not resampled independently, preserving within-engine temporal autocorrelation.
3. **Random Seed & Iteration Count**: Fixed seed `42` using `numpy.random.default_rng(42)` with $B = 2,000$ paired resamples.
4. **Confidence Interval Formulation**: Formulated using empirical percentiles ($2.5\text{th}$ and $97.5\text{th}$ percentiles) of the paired difference distribution across the 2,000 resamples.
5. **Exact Reproduction**: All point estimates and confidence bounds reproduced from [`reports/results/phase5_engine_predictions.csv`](file:///c:/Users/Harish/Downloads/FailureDNA/reports/results/phase5_engine_predictions.csv) with $<10^{-4}$ discrepancy.

---

## 4. NASA Score Uncertainty and Metric Instability Analysis

### 4.1 Retaining the Documented Scoring Formula

The Phase 5 evaluation correctly implemented the standard C-MAPSS asymmetric scoring function:
$$d_i = \hat{y}_i - y_i^*$$
$$s_i = \begin{cases} \exp(-d_i / 13) - 1 & \text{if } d_i < 0 \quad (\text{early prediction}) \\ \exp(d_i / 10) - 1 & \text{if } d_i \ge 0 \quad (\text{late prediction}) \end{cases}$$
$$S = \sum_{i=1}^N s_i$$

### 4.2 Per-Engine Penalty Distribution and Outlier Sensitivity

While the formula is correct, the audit investigated why the paired bootstrap confidence interval for the NASA score difference between FailureDNA and XGBoost is remarkably wide ($[-3,759.86, +3,088.25]$) despite a small point difference ($-137.12$ points).

We analyzed the distribution of per-engine penalty differences $\delta s_i = s_i(\text{FailureDNA}) - s_i(\text{XGBoost})$ across the 248 test engines:
- **Total Fleet Score**: FailureDNA $= 8,049.31$, XGBoost $= 8,186.42$ ($\Delta S = -137.12$).
- **Per-Engine Mean Difference**: $\mu(\delta s) = -0.55$ points.
- **Per-Engine Median Difference**: $\text{Median}(\delta s) = +0.64$ points (favoring XGBoost on the majority of engines).
- **Per-Engine Standard Deviation**: $\sigma(\delta s) = 108.16$ points ($197\times$ larger than the mean difference!).
- **Skewness & Kurtosis**: Skewness $= -0.47$, Kurtosis $= 33.72$ (extremely heavy-tailed, leptokurtic distribution).

#### Table 2: Top Individual Engine Penalty Swings Between FailureDNA and XGBoost

| Engine Unit ID | Observed Cycles ($T_{\text{obs}}$) | True RUL ($y^*$) | FailureDNA Pred | XGBoost Pred | FailureDNA Penalty | XGBoost Penalty | Difference ($\delta s_i$) | Influence on Fleet Total |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **54** | $176$ | $176.0$ | $88.43$ | $78.92$ | $841.54$ | $1,749.92$ | $\mathbf{-908.38}$ | **Favors FailureDNA** (Single engine swings fleet score by $908$ pts) |
| **153** | $153$ | $173.0$ | $84.40$ | $119.81$ | $910.81$ | $58.83$ | $\mathbf{+851.99}$ | **Favors XGBoost** (Single engine swings fleet score by $852$ pts) |
| **125** | $21$ | $184.0$ | $148.53$ | $102.27$ | $14.31$ | $536.38$ | $-522.07$ | Favors FailureDNA |
| **246** | $29$ | $194.0$ | $153.38$ | $112.15$ | $21.75$ | $541.35$ | $-519.60$ | Favors FailureDNA |
| **175** | $175$ | $123.0$ | $84.51$ | $43.90$ | $18.31$ | $437.99$ | $-419.68$ | Favors FailureDNA |

### 4.3 Methodological Limitations of the Exponential NASA Score

1. **Heavy-Tailed Instability**: Because the NASA penalty grows exponentially ($\exp(|d|/10)$ or $\exp(|d|/13)$), an error discrepancy of just 30 cycles on a single engine creates an individual penalty swing of hundreds to thousands of points.
2. **Bootstrap Vulnerability**: In paired engine-level resampling, whether Unit 54 ($\delta s = -908$) or Unit 153 ($\delta s = +852$) is selected 0, 1, 2, or 3 times in a given draw causes the resampled fleet difference to fluctuate wildly.
3. **Implication for Claims**: 
   > [!WARNING]
   > The confidence interval $[-3,759.86, +3,088.25]$ **includes zero**. Because the exponential metric produces highly unstable uncertainty estimates on a finite test fleet of $N=248$ engines, point-estimate rankings on the NASA score cannot support any claim of statistical superiority or reliable advantage.

---

## 5. Report Wording Corrections

In accordance with Phase 5.2 requirements, the following language corrections are formally established:

1. **Replacement of "Covers Zero Symmetrically"**:
   - *Previous Imprecise Phrasing*: "the paired 95% bootstrap CI covers zero symmetrically."
   - *Corrected Phrasing*: **"the confidence interval includes zero."**
   - *Technical Rationale*: The bootstrap distribution of the NASA score difference is skewed and heavy-tailed, ranging from $-3,759.86$ to $+3,088.25$ around a point estimate of $-137.12$. It is not symmetric. The proper, mathematically accurate statement is simply that the confidence interval includes zero.
2. **Prohibition of Unadjusted Significance Claims**:
   - Authors and downstream reports must not claim statistical significance for nominal $p$-values ($p < 0.05$) when those contrasts fail to reach significance under Holm-Bonferroni correction.
   - Specifically, Random Forest vs. FailureDNA in MAE ($p_{\text{raw}} = 0.0340$, $\tilde{p}_{\text{holm}} = 0.1080$) and FailureDNA (Regime-Conditioned) vs. XGBoost in MAE ($p_{\text{raw}} = 0.0270$, $\tilde{p}_{\text{holm}} = 0.1080$) must be described as **not statistically significant**.

---

## 6. Scientific Conclusion and Taxonomy of Evidence

To prevent misinterpretation in academic reporting and engineering deployment, benchmark findings must be presented using a four-tier taxonomy:

### 1. Point-Estimate Ranking
- On point estimates alone:
  - **NASA Score**: FailureDNA (Regime-Conditioned) ranks 1st ($7,690.99$), FailureDNA (Regime-Aware) ranks 2nd ($8,049.31$), XGBoost ranks 3rd ($8,186.42$), and Random Forest ranks 4th ($8,382.76$).
  - **MAE**: XGBoost ranks 1st ($20.56$), Random Forest ranks 2nd ($21.28$), FailureDNA (Regime-Conditioned) ranks 3rd ($22.44$), FailureDNA (Regime-Aware) ranks 4th ($23.42$).
  - **RMSE**: XGBoost ranks 1st ($29.08$), Random Forest ranks 2nd ($29.77$), FailureDNA (Regime-Conditioned) ranks 3rd ($30.16$), FailureDNA (Regime-Aware) ranks 4th ($30.36$).

### 2. Nominal (Unadjusted) Statistical Evidence
- Unadjusted paired tests indicate that XGBoost and Random Forest achieve nominally lower MAE than FailureDNA (Regime-Aware) ($p = 0.0010$ and $p = 0.0340$).
- All RMSE differences and NASA score differences have unadjusted $p > 0.05$ (e.g., FDNA vs. XGBoost in NASA score has raw $p = 0.9220$).

### 3. Multiplicity-Adjusted Evidence (FWER Controlled at $\alpha = 0.05$)
- Under Holm's step-down procedure across the family of benchmark comparisons:
  - **XGBoost achieves a statistically significant advantage over FailureDNA (Regime-Aware) in MAE** ($\Delta = +2.8571$ cycles, $95\%$ CI: $[+1.000, +4.614]$, Holm adjusted $\tilde{p} = 0.0050$).
  - **No other pairwise comparison achieves statistical significance.** Random Forest vs. FailureDNA in MAE adjusts to $\tilde{p} = 0.1080$.

### 4. Uncertainty and Methodological Limitations
- The NASA asymmetric scoring function is inherently unstable under resampling due to its exponential formulation, producing heavy-tailed penalty distributions where single engines contribute swings of over $800$ points.
- The $95\%$ bootstrap confidence interval for the NASA score difference between FailureDNA and XGBoost spans from $-3,759.86$ to $+3,088.25$ and includes zero.
- **Definitive Finding**: The data do **not** support the claim that FailureDNA is superior to XGBoost on NASA C-MAPSS FD004. XGBoost is statistically superior in MAE, while FailureDNA achieves competitive RMSE and NASA scores via linear probing on self-supervised representations without manual rolling feature engineering.

---

## 7. Phase 5.2.1 Statistical Reproducibility Audit Addendum

**Audit Date**: October 14, 2026  
**Audit Target**: Phase 5.2 Statistical Correction Artifacts  
**Machine-Readable Summary**: [`reports/results/phase5_2_1_reproducibility_audit.json`](file:///c:/Users/Harish/Downloads/FailureDNA/reports/results/phase5_2_1_reproducibility_audit.json)  

### 7.1 Exact Raw P-Value Calculation Details
The paired-bootstrap hypothesis tests evaluate the null hypothesis:
$$H_0: \Delta = \text{Metric}(\text{Model A}) - \text{Metric}(\text{Model B}) = 0$$
- **Tail Convention**: Two-sided empirical bootstrap $p$-value:
  $$p_{\text{emp}} = \min\left(1.0, 2 \cdot \min\left(\frac{1}{B}\sum_{b=1}^B \mathbf{1}_{\Delta^{*(b)} \ge 0}, \frac{1}{B}\sum_{b=1}^B \mathbf{1}_{\Delta^{*(b)} \le 0}\right)\right)$$
- **Treatment of Ties at Zero**: Evaluated via weak inequalities ($\ge 0$ and $\le 0$). Audit inspection of all $2,000$ bootstrap resamples across all 15 comparisons confirmed that **exact ties with zero occurred exactly 0 times** ($\Delta^* = 0.000000$ count $= 0$ in all cases).
- **Finite-Resampling Pseudo-Count Analysis**:
  In addition to the empirical proportion $p_{\text{emp}}$, we evaluated the Davison-Hinkley (1997) finite-sample formulation:
  $$p_{\text{dh}} = \min\left(1.0, 2 \cdot \frac{1 + \min(k_{\text{pos}}, k_{\text{neg}})}{B + 1}\right)$$
  - For FailureDNA (Regime-Aware) vs. XGBoost in MAE ($k_{\text{neg}} = 1$ resample $\le 0$ out of $2,000$):
    - $p_{\text{emp}} = 2 \cdot (1 / 2000) = \mathbf{0.0010} \implies \tilde{p}_{\text{holm}} = \mathbf{0.0050}$ ($m=5$), $\tilde{p}_{\text{holm}} = \mathbf{0.0150}$ ($m=15$).
    - $p_{\text{dh}} = 2 \cdot (2 / 2001) = \mathbf{0.0020} \implies \tilde{p}_{\text{holm}} = \mathbf{0.0100}$ ($m=5$), $\tilde{p}_{\text{holm}} = \mathbf{0.0300}$ ($m=15$).
  - **Invariance of Inferences**: Both formulations yield identical statistical decisions. Under both methods, XGBoost's MAE advantage is statistically significant ($p_{\text{adj}} < 0.05$), and all other 14 comparisons remain not statistically significant ($p_{\text{adj}} \ge 0.108$).

### 7.2 Independent Numerical Reproduction
Using the frozen engine predictions in [`reports/results/phase5_engine_predictions.csv`](file:///c:/Users/Harish/Downloads/FailureDNA/reports/results/phase5_engine_predictions.csv) and random seed `42`, all point differences, confidence intervals, and empirical $p$-values reported in `phase5_1_paired_bootstrap.csv` and `phase5_2_adjusted_bootstrap.csv` were independently recomputed. The maximum absolute discrepancy across all 15 comparisons was $<10^{-4}$, confirming $100\%$ numerical reproducibility.

### 7.3 Direction Convention Audit
- **Declared Convention**: $\Delta = \text{Metric}(\text{Model A}) - \text{Metric}(\text{Model B})$.
- **Verification**:
  - FailureDNA (Regime-Aware) MAE $= 23.4204$ cycles; XGBoost MAE $= 20.5633$ cycles.
  - $\Delta = 23.4204 - 20.5633 = \mathbf{+2.8571 \text{ cycles}}$.
  - Because $\Delta > 0$ for an error metric, Model A has higher error than Model B. This confirms that XGBoost achieved superior (lower) MAE, exactly matching the reported text.

### 7.4 Preregistration Status Qualification
> [!IMPORTANT]
> **Preregistration Status**: Review of project records confirms that the $m=3$ primary baseline family (evaluating FailureDNA Regime-Aware vs. XGBoost, Random Forest, and Linear Ridge) was **not preregistered** prior to inspecting test outcomes. It represents a retrospective, post-hoc baseline subset.  
> Consequently, confirmatory statistical inference must rely upon the per-metric family ($m=5$) or the omnibus family ($m=15$). The $m=3$ analysis must be interpreted strictly as an exploratory baseline grouping. Under both confirmatory families ($m=5$ and $m=15$), the statistical conclusions remain identical: only XGBoost's MAE advantage achieves statistical significance.

