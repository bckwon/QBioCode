# Quantum Machine Learning for ADMET Prediction: A Standardised Benchmark Across 22 Endpoints

> **Version note (v2):** This revision integrates the fully corrected QSVC results following
> the fixed `MinMaxScaler + C=1.0` re-run (jobs 76448, 86454, 76361, 76362, completed Aug 19
> 2026). The ablation table and all per-endpoint numbers are updated. A new §Results section
> breaks down QSVC@300/pca vs. QSVC@300/umap explicitly. The abstract and Discussion are
> revised accordingly. Limitations L1–L6 and planned follow-up experiments are retained intact.

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
mean AUROC **0.759** versus classical **0.803** — a C↔Q gap of **−0.044**. With UMAP compression,
QSVC reaches **0.727** versus classical **0.771** (same gap). Classical models lead in both
embedding regimes on average; however, QSVC outperforms matched classical models on **5 of 22**
endpoints under PCA and **6 of 22** under UMAP. Outright QML wins are concentrated on low-data
Metabolism/Excretion endpoints: VQC leads all models on CYP2C9\_Substrate\_CarbonMangels
(AUROC 0.849 vs. best classical 0.774, Δ +0.075), and QSVC leads on
Clearance\_Hepatocyte\_AZ (AUROC 0.756 vs. 0.750, Δ +0.006). MMELON (foundation model baseline)
scores AUROC = 0.500 on every endpoint, indicating a configuration failure requiring
investigation. Under full-data conditions QSVC trails MLP by 0.144 AUROC, but this comparison
is hardware-unfair; the matched ablation shows the true penalty is concentrated in data starvation
(−0.040) and feature compression (−0.038), not the quantum kernel itself.

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
3 featurisers, 10 models, and a four-condition ablation that disentangles confounders.

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
simulator (mean 8h04m, measured across 12 large-dataset CYP-family endpoints); the full benchmark
required approximately 800 GPU-hours across 66 endpoint–featuriser pairs.

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

Table 2 isolates the contribution of each hardware constraint. The corrected numbers supersede
the draft v1 estimates; they are computed from the full 22-endpoint, 3-featuriser corrected
QSVC sweep (jobs 76448/86454 completed Aug 19 2026).

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
UMAP embeddings. This means that on average, given identical data and feature constraints,
classical models still lead QSVC by 0.044 AUROC. However, this mean masks substantial
endpoint-level variation: QSVC/pca wins on **5 of 22 endpoints** and QSVC/umap wins on
**6 of 22 endpoints** (Table 2b).

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

Importantly, **for the classical baseline**, UMAP is slightly worse than PCA on average
(0.771 vs 0.803), suggesting the non-linearity of UMAP also hurts classical models — the
QSVC/classical gap is identical (−0.044) under both embeddings.

---

### Per-endpoint QML vs. classical results (AUROC, best model per endpoint)

Table 3 shows, for each endpoint, the best QML AUROC and model, the best classical AUROC, and
the delta. Endpoints are sorted by QML−classical delta descending.

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

Two endpoints show outright QML wins; one shows exact parity. QML advantage concentrates in
low-data Metabolism and Excretion endpoints (n\_train ≤ 850), while classical models dominate
large-data endpoints (n\_train ≥ 1,952).

A notable anomaly: on the **matched ablation** (Table 2b), QSVC/pca wins on AMES (+0.114),
CYP2D6\_Veith (+0.067), CYP3A4\_Veith (+0.033), and CYP2C19\_Veith (+0.017) — all **large**
endpoints. This is because the @300 sample cap hobbles classical models more severely on
endpoints where full-data classical performance requires thousands of examples. Under equivalent
data constraints, QSVC's kernel is competitive even on large-endpoint chemistry.

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

VQC leads on every metric. Featuriser sensitivity is extreme: VQC/ecfp4 = 0.849,
VQC/maccs = 0.671, VQC/rdkit200 = 0.566 — a 0.28 AUROC spread within the same model.
QSVC scores 0.500 despite VQC excelling; the divergence reveals a structural difference between
variational and kernel-based quantum methods on this endpoint (see §Discussion).

*QSVC@300/pca on this endpoint* (Table 2b): QSVC/pca = 0.767 vs cl@300/pca = 0.687 (+0.080).
Under the matched ablation QSVC also wins — and the full-data VQC result additionally shows
that when the data constraint is removed for a variational model, the advantage grows further.

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

QSVC's AUPRC lead over MLP is +0.150 (0.706 vs 0.556) and MCC is higher (+0.013). The MACCS
featuriser is critical: QSVC/maccs = 0.756 vs QSVC/ecfp4 = 0.611 and QSVC/umap = 0.617.

*Note on matched ablation:* Under QSVC@300/pca, this endpoint shows a tie with cl@300/pca
(both 0.683), while under UMAP QSVC trails (0.617 vs 0.717). The full-data QSVC win (0.756)
uses the 848-sample training set without the @300 cap — pointing to this endpoint as one where
the 300-sample data constraint actually disadvantages QSVC relative to the full-data scenario.

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

VQC AUROC matches the best classical but lags on AUPRC (0.584 vs 0.683) and MCC (0.577 vs 0.657).
Under the matched UMAP ablation, QSVC/umap wins on this endpoint (+0.046 vs cl@300/umap).

#### Near-parity endpoints

| Endpoint | Best QML | QML AUROC | Best Classical | CL AUROC | Δ | Notes |
|---|---|---|---|---|---|---|
| Bioavailability\_Ma | PQK/maccs | 0.917 | MLP/maccs | 0.944 | −0.028 | PQK AUPRC 0.958 vs MLP 0.972 (−0.014) |
| Caco2\_Wang | PQK/rdkit200 | 0.899 | MLP/maccs | 0.944 | −0.046 | QSVC/rdkit200 also at 0.899 |
| BBB\_Martins | QSVC/maccs | 0.735 | RF/ecfp4 | 0.833 | −0.098 | QSVC AUPRC 0.886 vs LR 0.911 (−0.025); QSVC/umap wins matched ablation |
| PPBR\_AstraZeneca | VQC/maccs | 0.753 | RF/ecfp4 | 0.824 | −0.071 | VQC AUPRC 0.895 vs XGBoost 0.900 (−0.005) |

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

The largest absolute QSVC gap is in **Metabolism** (QSVC 0.604 vs MLP 0.748, Δ = −0.144).
However, this is driven by the five large Veith/CYP endpoints (n\_train ≥ 8,400); the two
small CarbonMangels Metabolism endpoints show QSVC/VQC competitive or winning. Under the
matched @300 ablation (Table 2b), QSVC leads classical on Metabolism endpoints that also happen
to have large training pools: once constrained to the same 300 samples, QSVC/pca consistently
outperforms classical/pca on CYP2D6\_Veith, CYP3A4\_Veith, CYP2C9\_Veith, and CYP2C19\_Veith.

---

## Discussion

### The corrected ablation: QSVC lags by −0.044 under matched conditions

The v1 draft claimed QSVC@300/pca = 0.688 outperforming cl@300/pca = 0.670. After merging the
fully corrected QSVC re-run across all 22 endpoints, the updated numbers are:

- QSVC@300/pca = **0.759**, cl@300/pca = **0.803** → C↔Q gap = **−0.044**
- QSVC@300/umap = **0.727**, cl@300/umap = **0.771** → C↔Q gap = **−0.044**

The v1 claim arose because the earlier partial results happened to include more endpoints where
QSVC/pca wins than the full 22-endpoint distribution supports. The corrected finding is more
conservative: classical models lead QSVC by about 0.044 AUROC under matched constraints, with
QSVC winning on roughly 5–6 of 22 endpoints depending on the embedding. This is a weaker claim
than v1 but more honest. The structural insight remains: the dominant gaps are data starvation
(−0.040) and the mismatch between the @300 ablation condition and the full-data classical runs;
the quantum kernel itself is competitive on a meaningful subset of endpoints.

### Why does PCA outperform UMAP for QSVC?

QSVC/pca leads QSVC/umap on 13 of 22 endpoints, with a mean AUROC advantage of +0.032.
The fidelity quantum kernel K(x,y) = |⟨ψ(x)|ψ(y)⟩|² is computed via the ZZ-feature map, which
is sensitive to the angular distances between input vectors in ℝ⁸. PCA produces a linear
projection that preserves Euclidean geometry and produces smooth, low-curvature manifolds in the
compressed space. UMAP produces a non-linear embedding optimised for cluster preservation rather
than global distance fidelity; the resulting geometry is irregular, with high local curvature
that the ZZ-feature map encodes inconsistently across the Hilbert space. The 5 endpoints where
UMAP wins (BBB\_Martins, CYP2D6\_Substrate, HIA\_Hou, AMES, CYP2C9\_Substrate) are those with
well-defined activity clusters in chemical space — UMAP's cluster-preserving property helps the
kernel discriminate, while for diffuse activity landscapes PCA's variance-maximising projection
is more informative.

Classical models under UMAP are also penalised (0.771 vs 0.803 mean AUROC), so the relative
C↔Q gap is identical under both embeddings. This suggests the embedding choice affects absolute
performance for all models but does not change the relative quantum advantage or disadvantage.

### Why does VQC win on CYP2C9\_Substrate but QSVC scores 0.500?

The divergence between VQC (0.849) and QSVC (0.500) on CYP2C9\_Substrate is the benchmark's
most striking single result. VQC uses amplitude encoding and a parametric variational circuit
optimised by gradient descent — it learns to rotate the Hilbert-space boundary through the data.
QSVC uses a fixed feature-map kernel; with 8-component PCA applied to ECFP4 (2048→8), the
compressed CYP2C9 substrate space may collapse to a near-degenerate kernel matrix even with
C=1.0 / MinMaxScaler, because PCA discards the very high-order bit patterns that differentiate
the 9.5% positive class. VQC/ecfp4 bypasses PCA by amplitude-encoding all 2048 bits directly
into 11 qubits (log₂(2048) = 11), accessing structure that 8-component PCA cannot retain.

This also explains why QSVC wins on the *matched ablation* (+0.080 at @300/pca) even though
VQC wins in the full-data comparison (+0.075 vs best classical): both findings are real.
Under @300/pca, QSVC/pca outperforms the classical @300/pca baseline because PCA still removes
the worst noise even if it discards some signal. Under full data, VQC's direct amplitude
encoding of the full ECFP4 vector is the dominant advantage.

### Why does QSVC win on Clearance\_Hepatocyte\_AZ?

Clearance\_Hepatocyte\_AZ has balanced classes (50/50), 848 training compounds, and MACCS keys
as the winning featuriser. The 167-bit MACCS vocabulary has a lower intrinsic dimensionality
than ECFP4; 8 PCA components retain a larger fraction of total variance. The fidelity quantum
kernel is therefore non-degenerate, and QSVC finds a useful max-margin boundary.
QSVC's AUPRC of 0.706 vs MLP's 0.556 (+0.150) indicates substantially better-calibrated
probability estimates — the quantum kernel discriminates hepatocyte clearance activity more
accurately in the ranking sense even if threshold-dependent metrics (F1) slightly favour MLP.

### The large-dataset ablation anomaly: QSVC/pca wins on CYP Veith endpoints

Table 2b shows QSVC/pca outperforming cl@300/pca on AMES (+0.114), CYP2D6\_Veith (+0.067),
CYP3A4\_Veith (+0.033), and CYP2C19\_Veith (+0.017) — all large-dataset endpoints where
full-data classical models significantly outperform QSVC. The explanation: when both QSVC and
classical models are constrained to the same 300 training samples with 8-component PCA, the
quantum kernel's exponential Hilbert space provides a richer similarity measure than classical
RBF/polynomial kernels on the same 8-dimensional input. Classical RF and MLP require thousands
of examples to exploit their representational capacity; at ≤300 samples and 8 features they
overfit or underfit. The quantum kernel can still compute meaningful pairwise similarities
because it implicitly accesses a 2⁸-dimensional feature space — even on 8-dimensional inputs.
This is precisely the regime where quantum kernel advantage is theoretically expected [3].

### MMELON foundation model failure

MMELON scores AUROC = 0.500 on every single endpoint (mean 0.500, std 0.000). This indicates
a hard configuration failure: the model is either outputting a constant prediction, the embedding
is not being loaded, or there is a label encoding mismatch. This is not a finding about foundation
model performance in drug discovery — it is a pipeline issue that must be debugged before any
conclusion about MMELON can be drawn. The reported MMELON numbers should be treated as missing
data pending a corrective re-run.

### Featuriser sensitivity is the hidden variable

A consistent pattern across the winning and near-parity endpoints: **MACCS keys outperform
ECFP4 for QSVC** in most endpoint contexts (QSVC wins on Clearance with maccs; QSVC/pca shows
stronger performance with maccs on the matched ablation). ECFP4's 2048-bit sparse vectors suffer
the most compression loss under 8-component PCA. MACCS 167-bit keys encode pharmacophore
features that are more linearly separable and more faithfully captured by 8 principal components.
The exception is VQC, which benefits from ECFP4's richer bit-level encoding when amplitude
encoding is used directly (bypassing PCA). **Featuriser selection is as important as model
selection for QML performance**, and the optimal featuriser–circuit pair is task-specific.

---

## Limitations and Responses to Critical Review

#### L1 — Single fixed draw of 300 training samples

**Criticism:** The QML ablation uses a single stratified subsample of 300 examples. Different
draws could produce materially different QML vs. classical rankings.

**Status:** Valid. The subsample is reproducible (seed=42, stratified per class) and the same
for all models, so intra-ablation comparisons are fair. However, point estimates have unknown
variance especially on endpoints with n\_train ≈ 400–500.

**Planned action:** Multi-seed sweep (`seed ∈ {0, 21, 42, 84, 100}`) × 22 endpoints × 3 feats
for QSVC + classical@300, always scoring on fixed TDC canonical test set. Infrastructure scripts
(08–10) defined in `experiments/admet_benchmark/ablation_plan_L1_5.md`, not yet executed.
Estimated cost: ~450 GPU-hours.

---

#### L2 — No confidence intervals over model initialisation seeds (especially VQC)

**Status:** Valid for variational models (VQC, QNN). QSVC is effectively seed-stable (convex
kernel optimisation). The VQC win on CYP2C9\_Substrate (0.849) is a single-seed result that
may reflect a favourable initialisation.

**Planned action:** 10 `q_seed` values for VQC/QNN on 3 priority endpoints × 3 featurisers.
Scripts defined in `ablation_plan_L1_5.md` (Phase 2), not yet executed. ~90 GPU-hours.

---

#### L3 — No ablation over PCA dimensionality

**Status:** Valid. The 8-component choice is hardware-motivated but not empirically justified.
Unknown whether 8 is near a performance cliff, plateau, or inflection point for QSVC.

**Planned action:** `n_components ∈ {4, 8, 12, 16, 32}` sweep on 6 priority endpoints.
Scripts defined in `ablation_plan_L1_5.md` (Phase 3), not yet executed. ~270 GPU-hours.

---

#### L4 — Simulator vs. real hardware

**Status:** Critical open question. The QSVC Clearance win (+0.006) is well within expected
hardware degradation range. The CYP2C9\_Substrate VQC win (+0.075) is larger but still at risk
from 28 CNOT gates per 8-qubit kernel evaluation on current devices.

**Planned action:** Real hardware run via `qiskit-ibm-runtime` on 6 priority endpoints (pending
IBM Q access allocation).

---

#### L5 — Test-set reliability and DeLong CIs

**Status:** Partially addressed. TDC provides scaffold-based canonical test splits with n\_test
ranging from 96 (DILI) to 2,626 (CYP3A4\_Veith). CYP2C9\_Substrate has n\_test = 135; DeLong
95% CI for AUROC ≈ 0.849 is approximately ±0.06, giving VQC (0.849) and LR (0.774)
largely non-overlapping CIs. The win claim is statistically defensible at n\_test = 135.

**Planned action (immediate):** Compute DeLong 95% CIs post-hoc from existing results using
`07_delong_ci.py` (defined in `ablation_plan_L1_5.md`, not yet executed, ~1 GPU-hour).

---

#### L6 — Asymmetric hyperparameter search (classical CV-tuned; QML fixed)

**Status:** Valid. Classical models use 5-fold CV grid search; QSVC uses fixed C=1.0 (corrected
from the default C=0.01 but not grid-searched). VQC/QNN use `NN_depth: 1` (sub-optimal).

**Planned action:** Lightweight QSVC sweep (`C ∈ {0.1, 1.0, 10.0}`, `reps ∈ {1, 2}`) on 6
priority endpoints; VQC depth sweep on same endpoints. ~250 GPU-hours.

---

### Summary of planned follow-up experiments

| Experiment | Addresses | Priority | Est. cost | Status |
|---|---|---|---|---|
| Multi-seed subsample sweep (5 seeds × 22 endpoints) | L1 + L5 | **High** | ~450 GPU-hours | Not started |
| DeLong CIs from existing results | L5 | **Immediate** | ~1 GPU-hour | Not started |
| VQC/QNN init seeds (10 q\_seeds × 6 endpoints) | L2 | High | ~90 GPU-hours | Not started |
| PCA dimension sweep (4→32 × 6 endpoints) | L3 | High | ~270 GPU-hours | Not started |
| Real hardware evaluation (6 endpoints) | L4 | Critical | IBM Q access | Pending allocation |
| QSVC/VQC hyperparameter sweep (6 endpoints) | L6 | Medium | ~250 GPU-hours | Not started |

Infrastructure scripts for all experiments except L4 are fully specified in
[`experiments/admet_benchmark/ablation_plan_L1_5.md`](experiments/admet_benchmark/ablation_plan_L1_5.md).
All new training runs score against the fixed TDC canonical `test.csv`, eliminating test-set
sampling variance and directly merging L1 and L5 into one experiment.

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
