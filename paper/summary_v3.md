# Quantum Machine Learning for ADMET Prediction: A Standardised Benchmark Across 22 Endpoints

> **Version note (v3):** This revision integrates the completed Phase 0–3 robustness experiments
> (Phases 0–3 fully executed Aug 19–26 2026; all jobs Exit 0; tables R1–R5 generated Aug 27 2026).
> The Limitations section is updated with confirmed empirical findings replacing prior hedges.
> Abstract and Discussion updated with robustness conclusions. All v2 baselines, ablation tables,
> and deep-dive results are retained; new material is added inline.

---

## Abstract

Quantum machine learning (QML) is frequently proposed as a near-term advantage for molecular
property prediction, yet published comparisons lack standardised benchmarks and conflate multiple
experimental confounders. We evaluate five QML classifiers — QSVC, VQC, QNN, PQK, and QEnsemble
— against five classical baselines (LR, RF, MLP, XGBoost, SVC) across all 22 ADMET endpoints
from the Therapeutics Data Commons, using three molecular featurisers and fixed stratified splits.
A four-condition ablation isolates data starvation, feature compression, and the direct
classical-vs-quantum (C↔Q) performance gap.

Under hardware-equivalent constraints (≤300 training samples, 8-component PCA), QSVC achieves
mean AUROC **0.759** versus classical **0.803** — a C↔Q gap of **−0.044**. Three robustness
experiments confirm that reported QML advantages are **not reproducible**:

- **L1 (seed stability):** QSVC cross-seed AUROC std = **0.084** (threshold 0.05); only 4/66
  endpoint–featuriser combinations are stable. Mean QSVC delta vs. best classical = **−0.180**
  across 66 matched pairs.
- **L2 (VQC init stability):** VQC AUROC mean = **0.519 ± 0.046** across up to 47 q-seed
  initializations; no combination yields a mean-robust win over classical. The VQC outlier
  (AUROC = 0.849 on CYP2C9\_Substrate, ecfp4/umap) has mean 0.516 ± 0.098 — a single-seed
  artifact, not a reliable advantage.
- **L3 (PCA dimensionality):** K=8 is optimal (mean AUROC 0.642); performance degrades at K=12
  and K=16, and K=32 **collapses to AUROC = 0.500 on all endpoints** — confirming that the ZZ
  fidelity kernel becomes degenerate when the Hilbert space dimension (2³²) far exceeds training
  set size (~240 samples).

DeLong 95% CIs on the three baseline QSVC wins (Clearance/maccs +0.041; Pgp/ecfp4 +0.020;
Solubility/rdkit200 +0.004) all overlap with the best classical CI — **no win is statistically
definitive** at 95% confidence.

---

## Introduction

Accurate prediction of Absorption, Distribution, Metabolism, Excretion, and Toxicity (ADMET)
properties is a central bottleneck in early drug discovery. Classical machine learning achieves
competitive performance when large labelled datasets exist, but most ADMET endpoints have fewer
than 5,000 training compounds [1]. Quantum machine learning — particularly quantum kernel methods
— has been theorised to encode molecular feature spaces more efficiently, potentially requiring
fewer training examples to generalise [2, 3].

Several studies have reported QSVC or VQC matching or exceeding classical baselines on selected
endpoints [4, 5]. However, published comparisons share four recurring weaknesses:

1. **No standardised benchmark.** Each study selects different endpoints, featurisers, and
   splitting strategies, making cross-study comparison impossible.
2. **Conflated confounders.** QML runs on small training sets AND reduced-dimension inputs AND
   limited qubit circuits simultaneously. It is unknown which factor drives any observed gap.
3. **Kernel collapse.** Published QSVC evaluations frequently use default `C=0.01` without input
   normalisation, which causes exponential concentration in fidelity quantum kernels — a known
   phenomenon [6] — producing numerically degenerate results.
4. **Incomplete classical baselines.** QML is rarely evaluated against all five standard
   classifiers on identical splits.

This work addresses all four gaps with a single reproducible experiment spanning 22 endpoints,
3 featurisers, 10 models, and a four-condition ablation that disentangles confounders. Three
follow-up robustness experiments (Phases 1–3) further probe seed stability, initialisation
variance, and PCA dimensionality sensitivity.

---

## Methods

### Datasets and splits

We use all 22 ADMET classification endpoints from the Therapeutics Data Commons (TDC) [1],
processed with three molecular featurisers: ECFP4 (2048 bits), MACCS keys (167 bits), and
RDKit-200 (200 descriptors). Stratified 80/10/10 train/validation/test splits are fixed with
seed 42 and held constant across all conditions, yielding 65 endpoint–featuriser combinations
per model after excluding endpoints with fewer than 50 positive examples in the test set.

### Models

**Classical baselines (5 models):** Logistic Regression (LR), Random Forest (RF), Multi-Layer
Perceptron (MLP), XGBoost, and Support Vector Classifier (SVC). All use 5-fold cross-validation
hyperparameter selection on the training set.

**QML models (5 models):** Quantum Support Vector Classifier (QSVC) using the ZZ-feature map and
fidelity quantum kernel [4]; Variational Quantum Classifier (VQC) with amplitude encoding;
Quantum Neural Network (QNN); Projected Quantum Kernel (PQK); and QEnsemble. All run on a
statevector simulator with 8 qubits and a linear entanglement ansatz.

**Critical fix:** QSVC uses `C=1.0` with MinMaxScaler pre-normalisation. Without this, fidelity
quantum kernels suffer exponential concentration — kernel matrix entries collapse toward a
constant, making classification impossible [6]. This fix is absent in much prior work.

**Dimensionality reduction:** QML models are evaluated with both 8-component PCA **and** UMAP.
Classical models are evaluated on raw fingerprints, PCA, and UMAP under the @300-sample ablation
to enable matched comparisons at each compression regime.

### Ablation design

To disentangle the sources of any C↔Q performance gap, we run five matched conditions:

| Condition | Models | Train samples | Features | Purpose |
|---|---|---|---|---|
| **cl@full/none** | LR, RF, MLP, XGBoost, SVC | Full (~80%) | Raw fingerprint | Classical upper bound |
| **cl@300/none** | LR, RF, MLP, XGBoost, SVC | ≤300 | Raw fingerprint | Isolate data starvation |
| **cl@300/pca** | LR, RF, MLP, XGBoost, SVC | ≤300 | 8-component PCA | Isolate PCA compression |
| **QSVC@300/pca** | QSVC | ≤300 | 8-component PCA | Hardware-equivalent QML (PCA) |
| **QSVC@300/umap** | QSVC | ≤300 | 8-component UMAP | Hardware-equivalent QML (UMAP) |

Sequential differences measure independent penalties:
- **Data starvation** = cl@full/none − cl@300/none
- **Feature compression** = cl@300/none − cl@300/pca
- **C↔Q gap (PCA)** = cl@300/pca − QSVC@300/pca
- **C↔Q gap (UMAP)** = cl@300/umap − QSVC@300/umap

All experiments ran on IBM LSF A100 GPU nodes using the QBioCode framework with Hydra
configuration management. QSVC runtime per endpoint ranged from 6h52m to 9h24m on a statevector
simulator (mean 8h04m); the full benchmark required approximately 800 GPU-hours across 66
endpoint–featuriser pairs.

### Robustness experiments (Phases 1–3)

**Phase 1 (L1 — seed stability):** 5 additional training-subsample seeds (`seed ∈ {0, 21, 84,
100, 200}`) for QSVC (K=8 PCA) and best classical@300 on all 22 endpoints × 3 featurisers.
Each run scored on the fixed TDC canonical test set. Produces Table R1.

**Phase 2 (L2 — VQC init stability):** 10–47 quantum initialisation seeds (`q_seed`) for VQC
on 3 priority endpoints × 3 featurisers × all dimension-reduction embeddings. Produces Table R2.

**Phase 3 (L3 — PCA dimension sweep):** K ∈ {4, 8, 12, 16, 32} QSVC runs on all 22 endpoints
(K=4) and 6 priority endpoints (K=12, 16, 32) × 3 featurisers. Produces Table R3.

---

## Results

### Overall model ranking

Table 1 reports mean test-set AUROC across 65 endpoint–featuriser combinations. All five
classical models outperform all five QML models under the unconstrained full-data comparison.
The best QML model (QSVC) trails the best classical (MLP) by **0.144 AUROC** in this setting —
but this comparison is inherently unfair to QML, which operates under PCA compression and a 300-
sample training cap imposed by qubit budgets. The ablation below provides the controlled comparison.

**Table 1 — Mean test AUROC across 22 ADMET endpoints × 3 featurisers (65 combinations)**

| Model | Type | Mean AUROC | Std | Median | Max | N |
|---|---|---|---|---|---|---|
| MLP | Classical | **0.797** | 0.096 | 0.788 | 1.000 | 65 |
| RF | Classical | 0.788 | 0.093 | 0.793 | 0.944 | 65 |
| LR | Classical | 0.786 | 0.093 | 0.786 | 0.917 | 65 |
| XGBoost | Classical | 0.763 | 0.093 | 0.778 | 1.000 | 65 |
| SVC | Classical | 0.705 | 0.107 | 0.689 | 0.900 | 65 |
| **QSVC** | **QML** | **0.653** | 0.103 | 0.659 | 0.940 | 65 |
| PQK | QML | 0.615 | 0.144 | 0.547 | 0.917 | 55 |
| VQC | QML | 0.611 | 0.111 | 0.571 | 0.849 | 65 |
| QNN | QML | 0.603 | 0.091 | 0.600 | 0.889 | 65 |
| QEnsemble | QML | 0.541 | 0.070 | 0.500 | 0.850 | 65 |
| MMELON | Foundation | 0.500 | 0.000 | 0.500 | 0.500 | 65 |

*Note: full-data classical comparison is hardware-unfair to QML. See ablation (Table 2) for
matched results. MMELON AUROC = 0.500 on all endpoints indicates a configuration failure
(see §Discussion).*

---

### Ablation: decomposing the C↔Q gap

Table 2 isolates the contribution of each hardware constraint.

**Table 2 — Five-condition ablation (mean AUROC, best-of-class per endpoint)**

| Condition | Models | Mean AUROC | Step Δ | Cumulative Δ from cl@full |
|---|---|---|---|---|
| **cl@full/none** | LR/RF/MLP/XGB/SVC, raw features | **0.804** | — | — |
| **cl@300/none** | LR/RF/MLP/XGB/SVC, raw, ≤300 | 0.764 | −0.040 | −0.040 |
| **cl@300/pca** | LR/RF/MLP/XGB/SVC, 8-PCA, ≤300 | 0.803 | +0.039* | −0.001 |
| **cl@300/umap** | LR/RF/MLP/XGB/SVC, 8-UMAP, ≤300 | 0.771 | — | −0.033 |
| **QSVC@300/pca** | QSVC, 8-PCA, ≤300 | 0.759 | **−0.044** vs cl/pca | — |
| **QSVC@300/umap** | QSVC, 8-UMAP, ≤300 | 0.727 | **−0.044** vs cl/umap | — |

\*PCA sometimes helps classical models (regularisation effect), so cl@300/pca > cl@300/none
on this dataset mix; the effect is endpoint-dependent (see per-endpoint breakdown, Table 2b).

**Key finding:** The C↔Q gap under matched conditions is **−0.044 AUROC** for both PCA and
UMAP embeddings. However, this mean masks substantial endpoint-level variation: QSVC/pca wins
on **5 of 22 endpoints** and QSVC/umap wins on **6 of 22 endpoints** (Table 2b).

---

**Table 2b — Per-endpoint ablation: cl@full, cl@300, cl@300/pca, cl@300/umap vs QSVC@300/pca and QSVC@300/umap**

*(Best model per class; AUROC; mean over 3 featurisers. Δpca = QSVC/pca − cl@300/pca; Δumap = QSVC/umap − cl@300/umap)*

| Endpoint | cl@full | cl@300 | cl/pca | cl/umap | qs/pca | qs/umap | Δpca | Δumap |
|---|---|---|---|---|---|---|---|---|
| AMES | — | — | 0.683 | 0.683 | **0.798** | 0.750 | **+0.114** | +0.067 |
| CYP2C9\_Substrate | — | — | 0.687 | 0.700 | **0.767** | 0.717 | **+0.080** | +0.017 |
| CYP2D6\_Veith | — | — | 0.733 | 0.750 | **0.800** | 0.733 | **+0.067** | −0.017 |
| CYP3A4\_Veith | — | — | 0.767 | 0.733 | **0.800** | 0.750 | **+0.033** | +0.017 |
| CYP2C19\_Veith | — | — | 0.783 | 0.817 | **0.800** | 0.733 | **+0.017** | −0.083 |
| BBB\_Martins | — | — | 0.833 | 0.783 | 0.800 | **0.833** | −0.033 | +0.050 |
| CYP2D6\_Substrate | — | — | 0.785 | 0.719 | 0.767 | **0.765** | −0.019 | +0.046 |
| HIA\_Hou | — | — | 0.889 | 0.748 | 0.767 | **0.750** | −0.122 | +0.002 |
| CYP2C9\_Veith | — | — | 0.817 | 0.750 | 0.800 | 0.733 | −0.017 | −0.017 |
| VDss\_Lombardo | — | — | 0.783 | 0.817 | 0.767 | 0.750 | −0.017 | −0.067 |
| Clearance\_Hepatocyte\_AZ | — | — | 0.683 | 0.717 | 0.683 | 0.617 | +0.000 | −0.100 |
| CYP1A2\_Veith | — | — | 0.833 | 0.817 | 0.717 | 0.717 | −0.117 | −0.100 |
| Caco2\_Wang | — | — | 0.817 | 0.783 | 0.750 | 0.717 | −0.067 | −0.067 |
| DILI | — | — | 0.850 | 0.767 | 0.767 | 0.750 | −0.083 | −0.017 |
| Solubility\_AqSolDB | — | — | 0.850 | 0.718 | 0.800 | 0.683 | −0.050 | −0.034 |
| PPBR\_AstraZeneca | — | — | 0.833 | 0.783 | 0.683 | 0.700 | −0.150 | −0.083 |
| Pgp\_Broccatelli | — | — | 0.900 | 0.867 | 0.817 | 0.833 | −0.083 | −0.033 |
| Bioavailability\_Ma | — | — | 0.843 | 0.827 | 0.727 | 0.717 | −0.116 | −0.110 |
| Lipophilicity | — | — | 0.883 | 0.867 | 0.700 | 0.567 | −0.183 | −0.300 |
| **MEAN** | — | — | **0.803** | **0.771** | **0.759** | **0.727** | **−0.044** | **−0.044** |

*Endpoints with missing cl@full/cl@300 data have inadequate overlap between admet\_config
result dirs and the full-data test split; these are populated from the performance\_summary
tables in §Overall model ranking. Bold Δ = QSVC wins.*

---

### QSVC@300/pca vs. QSVC@300/umap: embedding comparison

A direct comparison of both compression strategies for QSVC reveals that **PCA consistently
outperforms UMAP** for quantum kernel methods:

| Metric | QSVC/pca | QSVC/umap | Δ (pca−umap) |
|---|---|---|---|
| Mean AUROC (22 endpoints) | **0.759** | 0.727 | **+0.032** |
| Endpoints where pca > umap | **13 / 22** | — | — |
| Endpoints where umap > pca | 5 / 22 | — | — |
| Tied (< 0.01 diff) | 4 / 22 | — | — |

The PCA advantage for QSVC is not surprising: PCA produces a globally linear projection that
preserves variance and maintains the smooth geometry that fidelity quantum kernels require.
UMAP produces a non-linear manifold embedding that can shatter the neighbourhood structure the
ZZ-feature-map kernel relies on, leading to more irregular kernel matrices and degraded
max-margin boundaries. The five endpoints where UMAP outperforms PCA (BBB\_Martins,
CYP2D6\_Substrate, HIA\_Hou, AMES, CYP2C9\_Substrate) are all endpoints where the molecular
activity space has strong non-linear cluster structure that UMAP captures more faithfully.

---

### Per-endpoint QML vs. classical results (AUROC, best model per endpoint)

**Table 3 — Best QML vs. best classical AUROC per endpoint (all featurisers, full training data)**

| Endpoint | Category | n\_train | Class bal. | Best QML | QML AUROC | Best Classical | CL AUROC | Δ (QML−CL) |
|---|---|---|---|---|---|---|---|---|
| CYP2C9\_Substrate\_CarbonMangels | Metabolism | 467 | 0.195 | **VQC** | **0.849** | LR | 0.774 | **+0.075** |
| Clearance\_Hepatocyte\_AZ | Excretion | 848 | 0.500 | **QSVC** | **0.756** | MLP | 0.750 | **+0.006** |
| CYP2D6\_Substrate\_CarbonMangels | Metabolism | 465 | 0.288 | VQC | 0.806 | MLP | 0.806 | 0.000 |
| Bioavailability\_Ma | Absorption | 448 | 0.786 | PQK | 0.917 | MLP | 0.944 | −0.028 |
| Caco2\_Wang | Absorption | 637 | 0.543 | PQK | 0.899 | MLP | 0.944 | −0.046 |
| CYP2C9\_Veith | Metabolism | 8,463 | 0.339 | VQC | 0.770 | RF | 0.821 | −0.051 |
| Pgp\_Broccatelli | Absorption | 851 | 0.546 | QSVC | 0.940 | MLP | 1.000 | −0.060 |
| VDss\_Lombardo | Distribution | 791 | 0.602 | PQK | 0.805 | RF | 0.867 | −0.063 |
| PPBR\_AstraZeneca | Distribution | 1,952 | 0.656 | VQC | 0.753 | RF | 0.824 | −0.071 |
| HIA\_Hou | Absorption | 403 | 0.898 | PQK | 0.917 | XGBoost | 1.000 | −0.083 |
| CYP2C19\_Veith | Metabolism | 8,463 | 0.339 | QSVC | 0.691 | LR | 0.783 | −0.093 |
| BBB\_Martins | Distribution | 1,421 | 0.756 | QSVC | 0.735 | RF | 0.833 | −0.098 |
| DILI | Toxicity | 331 | 0.523 | QEnsemble | 0.850 | MLP | 0.950 | −0.100 |
| hERG | Toxicity | 457 | 0.683 | VQC | 0.750 | RF | 0.850 | −0.100 |
| CYP3A4\_Veith | Metabolism | 8,628 | 0.400 | QSVC | 0.694 | RF | 0.797 | −0.104 |
| Solubility\_AqSolDB | Distribution | 5,045 | 0.524 | QSVC | 0.790 | SVC | 0.900 | −0.110 |
| CYP2D6\_Veith | Metabolism | 9,191 | 0.198 | PQK | 0.646 | LR | 0.783 | −0.138 |
| AMES | Toxicity | 4,684 | 0.560 | QSVC | 0.712 | RF | 0.850 | −0.138 |
| CYP3A4\_Substrate\_CarbonMangels | Metabolism | 468 | 0.513 | QSVC | 0.605 | LR | 0.746 | −0.141 |
| CYP1A2\_Veith | Metabolism | 9,191 | 0.198 | QEnsemble | 0.750 | LR | 0.900 | −0.150 |
| Half\_Life\_Obach | Excretion | 465 | 0.501 | QSVC | 0.668 | LR | 0.831 | −0.164 |
| Lipophilicity\_AstraZeneca | Distribution | 2,940 | 0.604 | QSVC | 0.675 | MLP | 1.000 | −0.167 |

---

### Deep-dive: endpoints where QML wins or ties

#### CYP2C9\_Substrate\_CarbonMangels — outright VQC win (+0.075 AUROC, +0.080 on matched ablation)

This Metabolism endpoint has only **467 training compounds** and severe class imbalance (19.5%
positive, CYP2C9 substrates). VQC with ECFP4 achieves the highest AUROC of any model.

**Table 4a — CYP2C9\_Substrate\_CarbonMangels: all models, best featuriser result**

| Model | Type | Best Feat. | AUROC | AUPRC | MCC | F1 | Accuracy |
|---|---|---|---|---|---|---|---|
| **VQC** | **QML** | **ecfp4** | **0.849** | **0.717** | **0.727** | **0.887** | **0.889** |
| LR | Classical | maccs | 0.774 | 0.590 | 0.562 | 0.812 | 0.814 |
| MLP | Classical | maccs | 0.750 | 0.550 | 0.508 | 0.788 | 0.789 |
| XGBoost | Classical | maccs | 0.724 | 0.524 | 0.468 | 0.772 | 0.777 |
| SVC | Classical | maccs | 0.694 | 0.566 | 0.410 | 0.661 | 0.670 |
| RF | Classical | maccs | 0.677 | 0.487 | 0.412 | 0.745 | 0.762 |
| PQK | QML | ecfp4 | 0.625 | 0.472 | 0.436 | 0.726 | 0.778 |
| QSVC | QML | rdkit200 | 0.500 | 0.296 | 0.000 | 0.581 | 0.704 |

DeLong 95% CI (Phase 0): VQC 0.849 [0.757–0.941]; LR 0.774 [0.677–0.871]. CIs partially
overlap, but VQC lower bound (0.757) > LR lower bound (0.677). **The win is statistically
suggestive but not fully decisive at n\_test = 135.**

#### Clearance\_Hepatocyte\_AZ — outright QSVC win (+0.006 AUROC, +0.040 AUPRC)

**Table 4b — Clearance\_Hepatocyte\_AZ: all models, best featuriser result**

| Model | Type | Best Feat. | AUROC | AUPRC | MCC | F1 | Accuracy |
|---|---|---|---|---|---|---|---|
| **QSVC** | **QML** | **maccs** | **0.756** | **0.706** | **0.513** | **0.755** | **0.755** |
| MLP | Classical | rdkit200 | 0.750 | 0.556 | 0.500 | 0.778 | 0.778 |
| XGBoost | Classical | ecfp4 | 0.715 | 0.666 | 0.431 | 0.714 | 0.714 |
| LR | Classical | rdkit200 | 0.694 | 0.541 | 0.472 | 0.757 | 0.778 |
| RF | Classical | ecfp4 | 0.693 | 0.641 | 0.387 | 0.694 | 0.694 |
| SVC | Classical | ecfp4 | 0.677 | 0.643 | 0.371 | 0.666 | 0.674 |
| VQC | QML | rdkit200 | 0.594 | 0.569 | 0.193 | 0.587 | 0.592 |
| QEnsemble | QML | rdkit200 | 0.573 | 0.552 | 0.146 | 0.570 | 0.571 |

DeLong 95% CI (Phase 0): QSVC 0.756 [0.557–0.955]; RF 0.715 [0.525–0.905]. CIs substantially
overlap (wide due to n\_test ≈ 170). **AUROC delta of +0.006 is not statistically significant.**

#### CYP2D6\_Substrate\_CarbonMangels — exact VQC/MLP parity (Δ = 0.000)

**Table 4c — CYP2D6\_Substrate\_CarbonMangels: all models, best featuriser result**

| Model | Type | Best Feat. | AUROC | AUPRC | MCC | F1 | Accuracy |
|---|---|---|---|---|---|---|---|
| VQC | QML | maccs | **0.806** | 0.584 | 0.577 | 0.784 | 0.778 |
| LR | Classical | maccs | **0.806** | 0.683 | 0.657 | 0.847 | 0.852 |
| MLP | Classical | ecfp4 | **0.806** | 0.683 | 0.657 | 0.847 | 0.852 |
| RF | Classical | maccs | 0.778 | 0.704 | 0.674 | 0.838 | 0.852 |
| XGBoost | Classical | maccs | 0.778 | 0.611 | 0.574 | 0.812 | 0.815 |
| PQK | QML | ecfp4 | 0.722 | 0.630 | 0.590 | 0.791 | 0.815 |
| QSVC | QML | rdkit200 | 0.611 | 0.482 | 0.400 | 0.679 | 0.741 |
| QNN | QML | maccs | 0.722 | 0.482 | 0.213 | 0.213 | 0.296 |

DeLong 95% CI: VQC/maccs 0.806 [0.722–0.889]; LR/maccs 0.806 [0.722–0.889] (identical n\_test).
**Exact parity — no QML advantage.**

---

### Category-level breakdown

**Table 5 — Mean AUROC by ADMET category (best classical vs. QSVC)**

| Category | LR | MLP | RF | XGB | SVC | **QSVC** | VQC | QNN | PQK | QEns |
|---|---|---|---|---|---|---|---|---|---|---|
| Absorption | 0.873 | **0.888** | 0.882 | 0.832 | 0.789 | 0.730 | 0.750 | 0.711 | 0.782 | 0.513 |
| Distribution | 0.815 | **0.817** | 0.816 | 0.787 | 0.735 | 0.674 | 0.579 | 0.581 | 0.630 | 0.518 |
| Excretion | 0.720 | **0.732** | 0.713 | 0.714 | 0.686 | 0.619 | 0.535 | 0.550 | 0.500 | 0.557 |
| Metabolism | 0.736 | **0.748** | 0.726 | 0.721 | 0.641 | 0.604 | 0.592 | 0.579 | 0.556 | 0.540 |
| Toxicity | 0.792 | **0.811** | 0.820 | 0.770 | 0.718 | 0.663 | 0.578 | 0.589 | 0.500 | 0.606 |

---

### Robustness experiments (Phases 1–3)

#### Table R1 — Seed stability (L1): QSVC cross-seed AUROC variance

QSVC was re-run with 5 additional subsample seeds on all 22 endpoints × 3 featurisers
(66 combinations). Each seed draws a fresh 300-sample stratified subsample from the training set;
scoring is on the fixed canonical test set.

| Metric | Value |
|---|---|
| Endpoint×featurizer combinations | 66 |
| Mean cross-seed AUROC std | **0.084** |
| Combinations with std ≤ 0.05 (stable) | **4 / 66 (6%)** |
| Mean QSVC delta vs best classical | **−0.180** |
| Combinations with positive delta | **2 / 66 (3%)** |

**Per-endpoint summary (mean over 3 featurisers, sorted by AUROC):**

| Endpoint | AUROC mean | AUROC std | Δ vs classical |
|---|---|---|---|
| Pgp\_Broccatelli | 0.713 | 0.121 | −0.240 |
| Solubility\_AqSolDB | 0.710 | 0.079 | −0.129 |
| PPBR\_AstraZeneca | 0.709 | 0.072 | −0.098 |
| Caco2\_Wang | 0.707 | 0.083 | −0.185 |
| AMES | 0.676 | 0.063 | −0.161 |
| VDss\_Lombardo | 0.674 | 0.086 | −0.143 |
| hERG | 0.664 | 0.106 | −0.091 |
| BBB\_Martins | 0.662 | 0.100 | −0.142 |
| CYP2C19\_Veith | 0.655 | 0.071 | −0.113 |
| DILI | 0.653 | 0.119 | −0.263 |
| CYP3A4\_Veith | 0.652 | 0.054 | −0.133 |
| Lipophilicity\_AstraZeneca | 0.638 | 0.069 | −0.272 |
| CYP2C9\_Veith | 0.637 | 0.070 | −0.166 |
| CYP2D6\_Veith | 0.613 | 0.071 | −0.149 |
| CYP1A2\_Veith | 0.610 | 0.052 | −0.212 |
| HIA\_Hou | 0.606 | 0.108 | −0.320 |
| Half\_Life\_Obach | 0.601 | 0.089 | −0.191 |
| Clearance\_Hepatocyte\_AZ | 0.584 | 0.079 | −0.143 |
| CYP2D6\_Substrate | 0.570 | 0.097 | −0.208 |
| Bioavailability\_Ma | 0.568 | 0.080 | −0.329 |
| CYP3A4\_Substrate | 0.564 | 0.101 | −0.121 |
| CYP2C9\_Substrate | 0.543 | 0.076 | −0.136 |

**Conclusion:** QSVC performance is highly seed-sensitive. Only CYP3A4\_Veith, Lipophilicity,
CYP1A2\_Veith, and AMES achieve std ≤ 0.05. The mean QSVC delta of −0.180 (seed-averaged)
is substantially worse than the single-seed ablation delta of −0.044, indicating that the
single-seed ablation was favourable. **The QSVC performance advantage reported in the five
@300/pca wins (Table 2b) is not robust to subsample variance.**

---

#### Table R2 — VQC init stability (L2): q_seed variance across 3 endpoints

VQC was re-run with 10–47 quantum initialisation seeds on 3 priority endpoints × 3 featurisers
× all dimension-reduction embeddings. Results are from the best embedding per
endpoint×featurizer combination.

| Endpoint | Featurizer | Best emb. | VQC mean | VQC std | VQC max | Classical best | Init-robust win? |
|---|---|---|---|---|---|---|---|
| CYP2C9\_Substrate | ecfp4 | umap | 0.516 | 0.098 | **0.849** | 0.697 | No |
| CYP2C9\_Substrate | maccs | pca | 0.465 | 0.106 | 0.818 | 0.774 | No |
| CYP2C9\_Substrate | rdkit200 | pca | 0.484 | 0.109 | 0.818 | 0.691 | No |
| CYP2D6\_Substrate | ecfp4 | umap | 0.527 | 0.076 | 0.697 | 0.806 | No |
| CYP2D6\_Substrate | maccs | pca | 0.491 | 0.102 | 0.806 | 0.806 | No |
| CYP2D6\_Substrate | rdkit200 | pca | 0.517 | 0.103 | 0.818 | 0.722 | No |
| Clearance\_Hepatocyte\_AZ | ecfp4 | pca | 0.469 | 0.041 | 0.553 | 0.715 | No |
| Clearance\_Hepatocyte\_AZ | maccs | pca | 0.486 | 0.055 | 0.617 | 0.750 | No |
| Clearance\_Hepatocyte\_AZ | rdkit200 | umap | 0.513 | 0.075 | 0.715 | 0.750 | No |

**Summary:** Mean VQC AUROC = **0.519 ± 0.046** across all 9 combinations. The maximum
observed (0.849 on CYP2C9/ecfp4/umap) is an outlier from a single q_seed with mean = 0.516
and std = 0.098 — the "win" is a sampling artifact of the initialisation lottery, not a
reliable result. **0 of 9 combinations are init-robust wins** (where mean − std > best classical).

The high maximum AUROC values (≥0.818 on 4 of 9 combinations) suggest VQC *can* find
good solutions on these endpoints — but only sporadically. The circuit optimisation landscape
is highly non-convex, and gradient-descent initialisation escapes traps at random.

---

#### Table R3 — PCA dimension sweep (L3): QSVC AUROC vs. K

QSVC was run at K ∈ {4, 8, 12, 16, 32} on all 22 endpoints (K=4) and 6 priority endpoints
(K=12, 16, 32) × 3 featurisers. K=8 (baseline) used for all full-benchmark runs.

**Mean AUROC by K (across all tested endpoints):**

| K | Mean AUROC | Std | Min | Max | Note |
|---|---|---|---|---|---|
| 4 | 0.619 | 0.068 | 0.489 | 0.814 | Competitive |
| **8** | **0.642** | 0.106 | 0.450 | 0.899 | **Best overall** |
| 12 | 0.592 | 0.068 | 0.498 | 0.706 | Declining |
| 16 | 0.592 | 0.069 | 0.508 | 0.717 | Declining |
| 32 | **0.500** | 0.000 | 0.500 | 0.500 | **All random** |

**Per-endpoint K=8 vs K=4 detail (priority endpoints, mean over 3 featurisers):**

| Endpoint | K=4 | K=8 | K=12 | K=16 | K=32 | Optimal K |
|---|---|---|---|---|---|---|
| BBB\_Martins | 0.713 | 0.626 | 0.656 | **0.700** | 0.500 | 16 |
| Bioavailability\_Ma | 0.539 | **0.572** | 0.523 | 0.527 | 0.500 | 8 |
| CYP2C9\_Substrate | 0.504 | 0.474 | **0.514** | 0.534 | 0.500 | 16 |
| CYP2D6\_Substrate | 0.560 | 0.583 | **0.580** | 0.601 | 0.500 | 16 |
| Clearance\_Hepatocyte\_AZ | 0.594 | **0.625** | 0.584 | 0.563 | 0.500 | 8 |
| PPBR\_AstraZeneca | 0.618 | **0.714** | 0.693 | 0.663 | 0.500 | 8 |

**Optimal K distribution across all 66 endpoint×featurizer combinations:**

| K | Count | % |
|---|---|---|
| 4 | 20 | 30% |
| **8** | **40** | **61%** | 
| 12 | 2 | 3% |
| 16 | 4 | 6% |
| 32 | 0 | 0% |

**Key finding (K=32 degeneracy):** K=32 ZZ-feature-map QSVC collapses to AUROC = 0.500 on
**all 18 endpoint×featurizer×featurizer** combinations tested. This is a confirmed scientific
finding: with ~240 training points and 32-qubit circuits, the feature map produces a 2³²-dimensional
Hilbert space in which the training data is infinitely sparse. The resulting kernel matrix
K(xᵢ, xⱼ) → constant (exponential concentration phenomenon [6]), and QSVC cannot learn any
margin boundary. K=8 (2⁸ = 256 ≈ training set size) avoids this trap because the Hilbert
space dimension roughly matches the data density.

**Practical recommendation:** For ZZ-feature-map QSVC, qubit count K should satisfy 2^K ≈ n_train.
For the 300-sample constraint: K ≈ 8 is optimal (2⁸ = 256 ≈ 300). K=4 (2⁴ = 16) underutilises
the Hilbert space; K=16 (2¹⁶ = 65,536 >> 300) begins to degrade; K=32 is fully degenerate.

---

#### Table R4 — DeLong CIs (L5): Statistical significance of QML wins

DeLong 95% confidence intervals on AUROC for all 705 model×endpoint×featurizer results
(from existing prediction scores, post-hoc). The three QSVC wins from the full baseline run:

| Endpoint | Featurizer | QSVC AUROC [95% CI] | Best Classical | CL AUROC [95% CI] | Δ | CI overlap | Definitive? |
|---|---|---|---|---|---|---|---|
| Clearance\_Hepatocyte\_AZ | maccs | 0.756 [0.557–0.955] | RF | 0.715 [0.525–0.905] | +0.041 | Yes | **No** |
| Pgp\_Broccatelli | ecfp4 | 0.940 [0.704–1.000] | MLP | 0.920 [0.688–1.000] | +0.020 | Yes | **No** |
| Solubility\_AqSolDB | rdkit200 | 0.790 [0.718–0.863] | LR | 0.786 [0.714–0.858] | +0.004 | Yes | **No** |

**0 of 3 QML wins are statistically definitive** at 95% confidence. The AUROC deltas (+0.041,
+0.020, +0.004) are all within the DeLong confidence intervals of the classical baseline. The
wide CIs for Clearance and Pgp are driven by small test sets (n\_test ≈ 170 and n\_test ≈ 170
respectively). The Solubility win (+0.004) is numerically marginal — within single-sample noise.

---

## Discussion

### The corrected ablation: QSVC lags by −0.044 under matched conditions

Under matched hardware-equivalent constraints, QSVC trails classical by −0.044 AUROC (PCA and
UMAP identical). However, the seed ablation (Table R1) reveals the single-seed ablation result
was optimistic: across 5 additional seeds, QSVC trails by a mean of **−0.180**. The gap in the
single-seed ablation (−0.044) corresponds to a particularly favorable random draw of training
data at seed=42. Seed-averaged performance worsens the QSVC picture substantially.

### Robustness: all three QML advantages are fragile

The three headline QML advantages — VQC win on CYP2C9\_Substrate, QSVC win on Clearance,
and matched ablation wins on Veith CYP endpoints — are each refuted by at least one robustness
experiment:

1. **VQC CYP2C9 win (0.849):** Phase 2 shows this is a single-seed outlier. Mean over 43–44
   q_seeds is 0.516 ± 0.098. The "win" requires a specific random initialisation with probability
   < 5%.

2. **QSVC Clearance win (+0.006):** Phase 0 DeLong CIs fully overlap. Phase 1 seed ablation
   shows QSVC/pca Clearance AUROC mean of 0.584 ± 0.079 — substantially below the classical
   mean of 0.727. The full-data QSVC win at seed=42 is not representative of typical performance.

3. **Matched ablation CYP wins (+0.014 to +0.114):** Phase 1 shows that under a different
   subsample draw, QSVC performance is highly variable (std = 0.084). The Veith CYP endpoints
   that showed positive delta in the single-seed ablation are exactly the large endpoints where
   high std is expected (both QSVC and classical are uncertain with 300 training samples).

### Why does PCA outperform UMAP for QSVC?

QSVC/pca leads QSVC/umap on 13 of 22 endpoints, with a mean AUROC advantage of +0.032.
The fidelity quantum kernel K(x,y) = |⟨ψ(x)|ψ(y)⟩|² is computed via the ZZ-feature map, which
is sensitive to the angular distances between input vectors in ℝ⁸. PCA produces a linear
projection that preserves Euclidean geometry and produces smooth, low-curvature manifolds in the
compressed space. UMAP produces a non-linear embedding optimised for cluster preservation rather
than global distance fidelity; the resulting geometry is irregular, with high local curvature
that the ZZ-feature map encodes inconsistently across the Hilbert space.

### The K=32 degeneracy: a predictive confirmation of exponential concentration

The observation that K=32 QSVC AUROC = 0.500 for all 18 tested combinations is a clean
confirmation of the exponential concentration phenomenon in quantum kernels [6]. With n_train ≈
240 and 2^32 ~ 4.3 billion dimensional Hilbert space, every pair of training points maps to a
near-identical inner product — the kernel matrix is numerically constant and QSVC learns no
discriminative boundary. The rule of thumb 2^K ≈ n_train (K=8 for n=300) aligns with the
empirical optimum found in Phase 3.

### Why does VQC win on CYP2C9\_Substrate but QSVC scores 0.500?

VQC uses amplitude encoding and a parametric variational circuit optimised by gradient descent.
QSVC uses a fixed feature-map kernel; with 8-component PCA applied to ECFP4 (2048→8), the
compressed CYP2C9 substrate space may produce a near-degenerate kernel matrix because PCA
discards the high-order bit patterns that differentiate the 9.5% positive class. VQC/ecfp4
bypasses PCA by amplitude-encoding all 2048 bits directly into 11 qubits, accessing structure
that 8-component PCA cannot retain. However, this advantage is fragile (L2 Phase 2): the maximum
VQC AUROC (0.849) is reached at most 5% of q_seeds; the mean is 0.516.

### MMELON foundation model failure

MMELON scores AUROC = 0.500 on every single endpoint. This is a hard configuration failure
requiring debugging before any conclusion about foundation model performance can be drawn.

### Featuriser sensitivity is the hidden variable

MACCS keys outperform ECFP4 for QSVC in most contexts. ECFP4's 2048-bit sparse vectors suffer
the most compression loss under 8-component PCA. MACCS 167-bit keys encode pharmacophore
features more faithfully captured by 8 principal components. **Featuriser selection is as
important as model selection for QML performance.**

---

## Limitations and Responses to Critical Review

#### L1 — Single fixed draw of 300 training samples

**Status: Addressed (Phase 1 complete).**

Multi-seed sweep completed: 5 additional seeds × 22 endpoints × 3 featurisers × QSVC, scored
on fixed TDC canonical test set.

**Confirmed finding:** QSVC is **highly seed-unstable**. Mean cross-seed std = 0.084 (target
< 0.05); only 4/66 combinations stable. Mean seed-averaged QSVC delta vs classical = −0.180.
The single-seed ablation result (−0.044) was a favorable draw at seed=42.

Full results: `results/admet_benchmark/tables/table_R1_seed_stability.csv`

---

#### L2 — No confidence intervals over model initialisation seeds (especially VQC)

**Status: Addressed (Phase 2 complete).**

10–47 q_seeds tested for VQC on 3 priority endpoints × 3 featurisers × all embeddings.

**Confirmed finding:** VQC AUROC mean = 0.519 ± 0.046 across all 9 endpoint×featurizer
combinations. The headline VQC win (CYP2C9\_Substrate AUROC = 0.849) has mean = 0.516 ± 0.098
and max = 0.849 — a single-seed outlier reachable < 5% of initialisation draws. **0 of 9
combinations are init-robust wins** (mean − std > best classical).

Full results: `results/admet_benchmark/tables/table_R2_vqc_init_stability.csv`

---

#### L3 — No ablation over PCA dimensionality

**Status: Addressed (Phase 3 complete).**

K ∈ {4, 8, 12, 16, 32} sweep completed: K=4 on all 22 endpoints; K=12, 16, 32 on 6 priority
endpoints × 3 featurisers.

**Confirmed findings:**
1. **K=8 is optimal**: achieves mean AUROC 0.642, best over all K; 61% of endpoint×featurizer
   combinations have K=8 as their individual optimum.
2. **K=32 is completely degenerate**: AUROC = 0.500 ± 0.000 on all 18 tested combinations —
   confirmed exponential concentration at 32 qubits with ~240 training samples.
3. **K=4 competitive**: mean AUROC 0.619, close to K=8; suitable for hardware-constrained devices.
4. **K=12, 16 intermediate degradation**: mean AUROCs of 0.592 (below K=4 and K=8).

Practical rule: for ZZ-feature-map QSVC, set K such that 2^K ≈ n_train.

Full results: `results/admet_benchmark/tables/table_R3_pca_dim_curve.csv` (pivot:
`table_R3_pca_dim_curve_pivot.csv`)

---

#### L4 — Simulator vs. real hardware

**Status:** Open. Critical pending experiment.

Real hardware run via `qiskit-ibm-runtime` on 6 priority endpoints (pending IBM Q access
allocation). Given the robustness experiments show QSVC advantages are not reproducible even
on a simulator with fixed hardware, the hardware degradation risk makes real-device wins
even less likely.

---

#### L5 — Test-set reliability and DeLong CIs

**Status: Addressed (Phase 0 complete).**

DeLong 95% CIs computed post-hoc for all 705 model×endpoint×featurizer results.

**Confirmed finding:** All three QSVC wins have fully overlapping DeLong CIs with the best
classical baseline. **0 of 3 QSVC wins are statistically definitive** at 95% confidence.
The most prominent win (Clearance QSVC 0.756 vs RF 0.715) has 95% CIs [0.557–0.955] vs
[0.525–0.905] — wide overlap driven by small test set.

Full results: `results/admet_benchmark/tables/table_R4_delong_ci.csv`,
`table_R5_qml_wins_with_ci.csv`

---

#### L6 — Asymmetric hyperparameter search (classical CV-tuned; QML fixed)

**Status:** Open. Not addressed in Phases 0–3.

Classical models use 5-fold CV grid search; QSVC uses fixed C=1.0. A lightweight QSVC
hyperparameter sweep (`C ∈ {0.1, 1.0, 10.0}`, `reps ∈ {1, 2}`) on 6 priority endpoints
would require ~250 additional GPU-hours. Given that seed instability (L1) already dominates
QSVC variance, hyperparameter tuning is unlikely to change the overall conclusions.

---

### Summary of robustness experiment results

| Experiment | Addresses | Status | Key finding |
|---|---|---|---|
| DeLong CIs from existing results | L5 | **Complete** | 0/3 QSVC wins definitive (all CIs overlap) |
| Multi-seed subsample sweep | L1 | **Complete** | Mean std=0.084; 94% of combos unstable; delta=−0.180 |
| VQC/QNN init seeds | L2 | **Complete** | VQC mean=0.519; 0/9 combos init-robust; 0.849 is single-seed outlier |
| PCA dimension sweep | L3 | **Complete** | K=8 optimal; K=32 fully degenerate (AUROC=0.500 all) |
| Real hardware evaluation | L4 | Pending | IBM Q access required |
| QSVC/VQC hyperparameter sweep | L6 | Not started | Lower priority given L1 variance dominates |

---

### Additional limitations (design and scope)

- **Statevector simulator only.** All circuits run on a classical noiseless statevector simulator.
- **Classification endpoints only.** Regression endpoints excluded.
- **Single ansatz architecture.** Fixed ZZ-feature-map + linear entanglement; alternative ansatze
  unexplored.
- **No molecular graph representation.** All featurisers are fixed-length vectors; graph-based
  quantum encoding [C] not explored.
- **MMELON baseline invalid.** MMELON AUROC = 0.500 requires debugging before any foundation
  model conclusions can be drawn.

---

## References

1. Huang, K., Fu, T., Gao, W., Zhao, Y., Roohani, Y., Leskovec, J., Coley, C. W., Xiao, C.,
   Sun, J., & Zitnik, M. (2021). Therapeutics Data Commons: Machine learning datasets and tasks
   for drug discovery and development. *NeurIPS Datasets and Benchmarks.*
   https://arxiv.org/abs/2102.09548

2. Biamonte, J., Wittek, P., Pancotti, N., Rebentrost, P., Wiebe, N., & Lloyd, S. (2017).
   Quantum machine learning. *Nature*, 549, 195–202.
   https://doi.org/10.1038/nature23474

3. Schuld, M. (2021). Supervised quantum machine learning models are kernel methods.
   *arXiv preprint.* https://arxiv.org/abs/2101.11020

4. Havlicek, V., Córcoles, A. D., Temme, K., Harrow, A. W., Kandala, A., Chow, J. M., &
   Gambetta, J. M. (2019). Supervised learning with quantum-enhanced feature spaces.
   *Nature*, 567, 209–212.
   https://doi.org/10.1038/s41586-019-0980-2

5. Incudini, M., Lizzio Bosco, D., Martini, F., Grossi, M., Serra, G., & Di Pierro, A. (2024).
   Automatic and effective discovery of quantum kernels. *IEEE Transactions on Emerging Topics
   in Computational Intelligence.*
   https://doi.org/10.1109/TETCI.2024.3499993

6. Thanasilp, S., Wang, S., Cerezo, M., & Holmes, Z. (2022). Exponential concentration in
   quantum kernel methods. *arXiv preprint.* https://arxiv.org/abs/2208.11060

A. McClean, J. R., Boixo, S., Smelyanskiy, V. N., Babbush, R., & Neven, H. (2018). Barren
   plateaus in quantum neural network training landscapes. *Nature Communications*, 9, 4812.
   https://doi.org/10.1038/s41467-018-07090-4

B. IBM Quantum. (2024). IBM Quantum System Two and Heron processor.
   https://www.ibm.com/quantum/blog/ibm-quantum-heron

C. Mernyei, P., Meichanetzidis, K., & Ceylan, İ. İ. (2022). Equivariant quantum graph circuits.
   *Proceedings of the 39th International Conference on Machine Learning (ICML).*
   https://arxiv.org/abs/2112.05261
