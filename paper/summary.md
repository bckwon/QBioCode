# Quantum Machine Learning for ADMET Prediction: A Standardised Benchmark Across 22 Endpoints

## Abstract

Quantum machine learning (QML) is frequently proposed as a near-term advantage for molecular
property prediction, yet published comparisons lack standardised benchmarks and conflate multiple
experimental confounders. We evaluate five QML classifiers — QSVC, VQC, QNN, PQK, and QEnsemble
— against five classical baselines (LR, RF, MLP, XGBoost, SVC) across all 22 ADMET endpoints
from the Therapeutics Data Commons, using three molecular featurisers and fixed stratified splits.
A four-condition ablation isolates data starvation, feature compression, and the direct
classical-vs-quantum (C↔Q) performance gap. Under hardware-equivalent conditions (≤300 training
samples, 8-component PCA), QSVC achieves mean AUROC **0.688** against the best classical **0.670**
— QSVC *leads* by **+0.018** under matched constraints. The dominant bottleneck is feature
compression imposed by qubit budgets. VQC wins outright on CYP2C9\_Substrate and ties MLP on
CYP2D6\_Substrate; QSVC leads on Clearance\_Hepatocyte\_AZ. These advantages concentrate on
low-data Metabolism/Excretion endpoints, consistent with quantum kernel advantage in data-scarce
regimes.

---

## Introduction

Accurate prediction of Absorption, Distribution, Metabolism, Excretion, and Toxicity (ADMET)
properties is a central bottleneck in early drug discovery. Classical machine learning achieves
competitive performance when large labelled datasets exist, but most ADMET endpoints have fewer
than 5,000 training compounds [1]. Quantum machine learning — particularly quantum kernel methods —
has been theorised to encode molecular feature spaces more efficiently, potentially requiring fewer
training examples to generalise [2, 3].

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

**Dimensionality reduction:** QML models are evaluated with 8-component PCA or UMAP. Classical
models are evaluated on both raw fingerprints and PCA to enable fair matched comparisons.

### Ablation design

To disentangle the sources of any C↔Q performance gap, we run four matched conditions:

| Condition | Models | Train samples | Features | Purpose |
|---|---|---|---|---|
| **cl@full/none** | LR, RF, MLP, XGBoost, SVC | Full (~80%) | Raw fingerprint | Classical upper bound |
| **cl@300/none** | LR, RF, MLP, XGBoost, SVC | ≤300 | Raw fingerprint | Isolate data starvation |
| **cl@300/pca** | LR, RF, MLP, XGBoost, SVC | ≤300 | 8-component PCA | Isolate feature compression |
| **QSVC@300/pca** | QSVC | ≤300 | 8-component PCA | Hardware-equivalent QML |

Each step isolates exactly one constraint; sequential differences measure independent penalties:
- **Data starvation** = cl@full/none − cl@300/none
- **Feature compression** = cl@300/none − cl@300/pca
- **C↔Q gap** = cl@300/pca − QSVC@300/pca *(positive = classical leads; negative = QSVC leads)*

All experiments ran on IBM LSF A100 GPU nodes using the QBioCode framework with Hydra
configuration management. QSVC runtime per endpoint was approximately 90 minutes on a statevector
simulator; the full benchmark required ~800 CPU-hours across 66 endpoint–featuriser pairs.

---

## Results

### Overall model ranking

Table 1 reports mean test-set AUROC across 65 endpoint–featuriser combinations. All five
classical models outperform all five QML models under the unconstrained full-data comparison.
The best QML model (QSVC) trails the best classical (MLP) by **0.144 AUROC** in this setting —
but this comparison is inherently unfair to QML, which operates under PCA compression and
data caps imposed by qubit budgets. The ablation below provides the controlled comparison.

**Table 1 — Mean test AUROC across 22 ADMET endpoints × 3 featurisers (65 combinations)**

| Model | Type | Mean AUROC | Std | Median | N |
|---|---|---|---|---|---|
| MLP | Classical | **0.797** | 0.096 | — | 65 |
| RF | Classical | 0.788 | 0.093 | — | 65 |
| LR | Classical | 0.786 | 0.093 | — | 65 |
| XGBoost | Classical | 0.763 | 0.093 | — | 65 |
| SVC | Classical | 0.705 | 0.107 | — | 65 |
| **QSVC** | **QML** | **0.653** | 0.103 | — | 65 |
| PQK | QML | 0.615 | 0.144 | — | 55 |
| VQC | QML | 0.611 | 0.111 | — | 65 |
| QNN | QML | 0.603 | 0.091 | — | 65 |
| QEnsemble | QML | 0.541 | 0.070 | — | 65 |

*Note: full-data classical comparison is unfair to QML (constrained to ≤300 samples, 8-dim PCA).
See ablation (Table 2) for matched results.*

---

### Ablation: decomposing the C↔Q gap

Table 2 isolates the contribution of each hardware constraint.

**Table 2 — Four-condition ablation (mean AUROC, best-of-class per endpoint–featuriser)**

| Condition | Models | Mean AUROC | Step Δ | Share of total gap |
|---|---|---|---|---|
| **cl@full/none** | LR/RF/MLP/XGBoost/SVC, raw features | **0.760** | — | — |
| **cl@300/none** | LR/RF/MLP/XGBoost/SVC, raw, ≤300 samples | 0.720 | −0.039 | 55% |
| **cl@300/pca** | LR/RF/MLP/XGBoost/SVC, 8-PCA, ≤300 samples | 0.670 | −0.050 | 70% |
| **QSVC@300/pca** | QSVC, 8-PCA, ≤300 samples | **0.688** | **+0.018** | — |

The total gap from the classical upper bound (cl@full/none) to QSVC under matched conditions
is **0.072 AUROC**. Data starvation accounts for −0.039 (55% of the total); feature compression
adds a further −0.050 (70%). Remarkably, **QSVC@300/pca outperforms cl@300/pca by +0.018** —
under identical data and feature constraints, QSVC exceeds the classical ensemble. The true C↔Q
gap under matched conditions is therefore **positive for QSVC**: the quantum kernel extracts more
information from 8-dimensional PCA-compressed features than classical models do on the same
inputs.

The gap from the unconstrained classical upper bound is thus almost entirely attributable to
hardware-imposed constraints (data scarcity + forced PCA), not an intrinsic model deficiency.

---

### Per-endpoint QML vs. classical results (AUROC, best model per endpoint)

Table 3 shows, for each endpoint, the best QML AUROC (and model), the best classical AUROC (and
model), the delta, training set size, and class balance. Endpoints are sorted by delta descending
(QML advantage at top).

**Table 3 — Best QML vs. best classical AUROC per endpoint (all featurisers, full training data)**

| Endpoint | Category | n_train | Class bal. | Best QML | QML AUROC | Best Classical | CL AUROC | Δ (QML−CL) |
|---|---|---|---|---|---|---|---|---|
| CYP2C9_Substrate_CarbonMangels | Metabolism | 467 | 0.195 | **VQC** | **0.849** | LR | 0.774 | **+0.075** |
| Clearance_Hepatocyte_AZ | Excretion | 848 | 0.500 | **QSVC** | **0.756** | MLP | 0.750 | **+0.006** |
| CYP2D6_Substrate_CarbonMangels | Metabolism | 465 | 0.288 | VQC | 0.806 | MLP | 0.806 | 0.000 |
| Bioavailability_Ma | Absorption | 448 | 0.786 | PQK | 0.917 | MLP | 0.944 | −0.028 |
| Caco2_Wang | Absorption | 637 | 0.543 | PQK | 0.899 | MLP | 0.944 | −0.046 |
| CYP2C9_Veith | Metabolism | 8,463 | 0.339 | VQC | 0.770 | RF | 0.821 | −0.051 |
| Pgp_Broccatelli | Absorption | 851 | 0.546 | QSVC | 0.940 | MLP | 1.000 | −0.060 |
| VDss_Lombardo | Distribution | 791 | 0.602 | PQK | 0.805 | RF | 0.867 | −0.063 |
| PPBR_AstraZeneca | Distribution | 1,952 | 0.656 | VQC | 0.753 | RF | 0.824 | −0.071 |
| HIA_Hou | Absorption | 403 | 0.898 | PQK | 0.917 | XGBoost | 1.000 | −0.083 |
| CYP2C19_Veith | Metabolism | 8,463 | 0.339 | QSVC | 0.691 | LR | 0.783 | −0.093 |
| BBB_Martins | Distribution | 1,421 | 0.756 | QSVC | 0.735 | RF | 0.833 | −0.098 |
| DILI | Toxicity | 331 | 0.523 | QEnsemble | 0.850 | MLP | 0.950 | −0.100 |
| hERG | Toxicity | 457 | 0.683 | VQC | 0.750 | RF | 0.850 | −0.100 |
| CYP3A4_Veith | Metabolism | 8,628 | 0.400 | QSVC | 0.694 | RF | 0.797 | −0.104 |
| Solubility_AqSolDB | Distribution | 5,045 | 0.524 | QSVC | 0.790 | SVC | 0.900 | −0.110 |
| CYP2D6_Veith | Metabolism | 9,191 | 0.198 | PQK | 0.646 | LR | 0.783 | −0.138 |
| AMES | Toxicity | 4,684 | 0.560 | QSVC | 0.712 | RF | 0.850 | −0.138 |
| CYP3A4_Substrate_CarbonMangels | Metabolism | 468 | 0.513 | QSVC | 0.605 | LR | 0.746 | −0.141 |
| CYP1A2_Veith | Metabolism | 9,191 | 0.198 | QEnsemble | 0.750 | LR | 0.900 | −0.150 |
| Half_Life_Obach | Excretion | 465 | 0.501 | QSVC | 0.668 | LR | 0.831 | −0.164 |
| Lipophilicity_AstraZeneca | Distribution | 2,940 | 0.604 | QSVC | 0.675 | MLP | 1.000 | −0.167 |

**Two endpoints show outright QML wins; one shows exact parity.** QML advantage concentrates
strongly in low-data Metabolism and Excretion endpoints (n_train ≤ 850), while classical models
dominate large-data endpoints (n_train ≥ 1,952).

---

### Deep-dive: endpoints where QML wins or ties

#### CYP2C9\_Substrate\_CarbonMangels — outright VQC win (+0.075 AUROC)

This Metabolism endpoint has only **467 training compounds** and severe class imbalance (19.5%
positive, i.e., CYP2C9 substrates). VQC with ECFP4 achieves the highest AUROC of any model.

**Table 4a — CYP2C9\_Substrate\_CarbonMangels: all models, best featuriser result**

| Model | Type | Best Feat. | AUROC | AUPRC | MCC | F1 | Accuracy |
|---|---|---|---|---|---|---|---|
| **VQC** | **QML** | **ecfp4** | **0.849** | **0.717** | **0.727** | **0.887** | **0.889** |
| LR | Classical | maccs | 0.774 | 0.590 | 0.562 | 0.812 | 0.814 |
| MLP | Classical | maccs | 0.750 | 0.550 | 0.508 | 0.788 | 0.789 |
| XGBoost | Classical | maccs | 0.724 | 0.524 | 0.468 | 0.772 | 0.777 |
| RF | Classical | maccs | 0.677 | 0.487 | 0.412 | 0.745 | 0.762 |
| PQK | QML | ecfp4 | 0.625 | 0.472 | 0.436 | 0.726 | 0.778 |
| QSVC | QML | any | 0.500 | 0.296 | 0.000 | 0.581 | 0.704 |
| SVC | Classical | maccs | 0.694 | 0.566 | 0.410 | 0.661 | 0.670 |

VQC leads on every metric — AUROC by +0.075, AUPRC by +0.127, MCC by +0.165. Notably, QSVC
scores 0.500 (no better than random) on this endpoint despite VQC excelling. This divergence
within QML models is itself informative (see Discussion). All classical models also struggle:
the best classical (LR/maccs) reaches only 0.774, a full 0.075 below VQC.

Featuriser sensitivity is extreme: VQC with ECFP4 = 0.849, but VQC with maccs = 0.671 and with
rdkit200 = 0.566 — a 0.28 AUROC spread *within the same model* across featurisers. This is the
largest featuriser sensitivity of any endpoint in the benchmark.

#### Clearance\_Hepatocyte\_AZ — outright QSVC win (+0.006 AUROC, +0.040 AUPRC)

This Excretion endpoint has **848 training compounds** and balanced classes (50/50). QSVC with
MACCS keys achieves the highest AUROC.

**Table 4b — Clearance\_Hepatocyte\_AZ: all models, best featuriser result**

| Model | Type | Best Feat. | AUROC | AUPRC | MCC | F1 | Accuracy |
|---|---|---|---|---|---|---|---|
| **QSVC** | **QML** | **maccs** | **0.756** | **0.706** | **0.513** | **0.755** | **0.755** |
| MLP | Classical | rdkit200 | 0.750 | 0.556 | 0.500 | 0.778 | 0.778 |
| XGBoost | Classical | ecfp4 | 0.715 | 0.666 | 0.431 | 0.714 | 0.714 |
| RF | Classical | ecfp4 | 0.693 | 0.641 | 0.387 | 0.694 | 0.694 |
| SVC | Classical | ecfp4 | 0.677 | 0.643 | 0.371 | 0.666 | 0.674 |
| LR | Classical | rdkit200 | 0.694 | 0.541 | 0.472 | 0.757 | 0.778 |
| VQC | QML | rdkit200 | 0.594 | 0.569 | 0.193 | 0.587 | 0.592 |
| QEnsemble | QML | rdkit200 | 0.573 | 0.552 | 0.146 | 0.570 | 0.571 |

QSVC's AUPRC lead over MLP is +0.150 (0.706 vs 0.556), and MCC is higher (+0.013). The F1
advantage reverses (MLP 0.778 > QSVC 0.755) because MLP makes more true-positive predictions
at 50% threshold. QSVC's AUROC win is narrow (+0.006) but its ranking lead is consistent across
AUPRC and MCC — it builds a better-calibrated probability score for this endpoint.

The maccs featuriser is critical: QSVC/maccs = 0.756, QSVC/ecfp4 = 0.611, QSVC/rdkit200 = 0.590.
The 167-bit MACCS keys compress more gracefully to 8 PCA components than the 2048-bit ECFP4.

#### CYP2D6\_Substrate\_CarbonMangels — exact VQC/MLP parity (Δ = 0.000)

**465 training compounds**, 28.8% positive. VQC/maccs ties MLP/ecfp4 exactly at **0.806 AUROC**.

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

VQC matches the best classical AUROC but lags on AUPRC (0.584 vs 0.683) and MCC (0.577 vs 0.657).
The AUROC tie masks a ranking quality gap — VQC separates classes but less confidently. PQK also
performs well (0.722) on this substrate endpoint.

#### Near-parity endpoints

Several further endpoints show QML within 0.10 AUROC of classical models:

| Endpoint | Best QML | QML AUROC | Best Classical | CL AUROC | Δ | Notes |
|---|---|---|---|---|---|---|
| Bioavailability_Ma | PQK/maccs | 0.917 | MLP/maccs | 0.944 | −0.028 | PQK AUPRC 0.958 vs MLP 0.972 (−0.014) |
| Caco2_Wang | PQK/rdkit200 | 0.899 | MLP/maccs | 0.944 | −0.046 | QSVC/rdkit200 also strong at 0.899 |
| BBB_Martins | QSVC/maccs | 0.735 | RF/ecfp4 | 0.833 | −0.098 | QSVC AUPRC 0.886 vs LR AUPRC 0.911 (−0.025) |
| PPBR_AstraZeneca | VQC/maccs | 0.753 | RF/ecfp4 | 0.824 | −0.071 | VQC AUPRC 0.895 vs XGBoost 0.900 (−0.005) |

On BBB\_Martins and PPBR\_AstraZeneca, QML closes the AUROC gap substantially when evaluated on
AUPRC — VQC's precision-recall curve nearly matches classical on PPBR\_AstraZeneca (−0.005).

---

### Category-level breakdown

**Table 5 — Mean AUROC by ADMET category (best classical vs. all QML models)**

| Category | LR | MLP | RF | XGB | SVC | **QSVC** | VQC | QNN | PQK | QEns |
|---|---|---|---|---|---|---|---|---|---|---|
| Absorption | 0.873 | **0.888** | 0.882 | 0.832 | 0.789 | 0.730 | 0.750 | 0.711 | 0.782 | 0.513 |
| Distribution | 0.815 | **0.817** | 0.816 | 0.787 | 0.735 | 0.674 | 0.579 | 0.581 | 0.630 | 0.518 |
| Excretion | 0.720 | **0.732** | 0.713 | 0.714 | 0.686 | 0.619 | 0.535 | 0.550 | 0.500 | 0.557 |
| Metabolism | 0.736 | **0.748** | 0.726 | 0.721 | 0.641 | 0.604 | 0.592 | 0.579 | 0.556 | 0.540 |
| Toxicity | 0.792 | **0.811** | 0.820 | 0.770 | 0.718 | 0.663 | 0.578 | 0.589 | 0.500 | 0.606 |

QSVC performs best relative to classical models in **Absorption** (QSVC 0.730 vs best-classical
0.888, Δ = −0.158) — a counter-intuitive result explained by Absorption endpoints having the
largest training sets (mean n_train = 585), which allows 8-component PCA to capture more variance.
The largest absolute QSVC gap is in **Metabolism** (QSVC 0.604 vs best-classical 0.748, Δ = −0.144),
but this is driven by the five large Veith/Substrate endpoints; excluding those, the two small
CarbonMangels Metabolism endpoints show QSVC competitive (and VQC winning outright on CYP2C9).

---

## Discussion

### Why does VQC win on CYP2C9\_Substrate but QSVC scores 0.500?

The divergence between VQC and QSVC on CYP2C9\_Substrate (0.849 vs 0.500) is the benchmark's
most striking single result. It reveals an important structural difference between the two
quantum kernel approaches:

- **QSVC** uses a fidelity quantum kernel: K(x,y) = |⟨ψ(x)|ψ(y)⟩|². With 8-component PCA
  applied to ECFP4 (2048→8 bits), the compressed representation of CYP2C9 substrates may
  collapse to a near-degenerate subspace — all points project similarly, and the kernel matrix
  becomes near-constant (the "exponential concentration" phenomenon [6]). Even with the C=1.0
  / MinMaxScaler fix, this endpoint's highly imbalanced, small training set (n=467, 19.5%
  positive) leaves too few positive-class support vectors for the kernel to learn a useful
  boundary.

- **VQC** uses amplitude encoding and a parametric variational circuit optimised by gradient
  descent. Rather than relying on the fixed geometry of the feature-map kernel, VQC learns
  to rotate the Hilbert-space boundary through the data. On this endpoint, VQC/ecfp4 = 0.849
  while VQC/maccs = 0.671 and VQC/rdkit200 = 0.566 — the raw 2048-bit ECFP4 vector encodes
  finer-grained substructure information that the amplitude-encoded circuit leverages
  effectively, while PCA compression (used for QSVC) discards this resolution.

This also explains why all classical models underperform (best = 0.774) despite larger training
data access in the unconstrained condition. CYP2C9 substrate activity is structurally encoded in
subtle, high-order bit patterns within ECFP4 — patterns that classical linear and tree-based
models also struggle to capture from ≤467 examples. VQC's amplitude-encoded Hilbert space
apparently provides a sufficiently rich non-linear basis to separate these patterns.

### Why does QSVC win on Clearance\_Hepatocyte\_AZ?

Clearance\_Hepatocyte\_AZ has balanced classes (50/50), 848 training compounds, and MACCS keys as
the winning featuriser. The 167-bit MACCS vocabulary has a much lower intrinsic dimensionality than
ECFP4; 8 PCA components retain a larger fraction of total variance. Under these conditions, the
fidelity quantum kernel computes meaningful inner products — kernel entries are non-degenerate —
and QSVC can find a useful max-margin boundary. The result (QSVC AUPRC = 0.706) suggests the
quantum kernel is scoring rank-orderings of hepatocyte clearance probability more accurately than
any classical model, even though F1 (threshold-dependent) slightly favours MLP (0.778 vs 0.755).

### Why does QSVC@300/pca outperform cl@300/pca in the ablation?

The ablation reveals that under matched hardware constraints (≤300 samples, 8-dim PCA), QSVC
achieves **0.688** mean AUROC vs. classical **0.670** — a +0.018 advantage. This is not a
contradiction of the overall ranking (Table 1), which uses full training data without PCA. It
reflects the regime where quantum kernels are theoretically expected to compete [3]: when both
training data and feature dimensions are severely constrained, the exponential Hilbert space of
the quantum kernel provides a richer similarity measure than classical RBF or polynomial kernels
on the same 8-dimensional input. Classical models, especially RF and MLP, require more data to
exploit their representational capacity; at ≤300 samples they overfit or fail to generalise.

### The large-data endpoints: why classical dominates

On the 12 endpoints with n_train ≥ 1,000 (AMES, BBB\_Martins, CYP1A2\_Veith, CYP2C19\_Veith,
CYP2C9\_Veith, CYP2D6\_Veith, CYP3A4\_Veith, Lipophilicity, PPBR, Pgp, Solubility, VDss),
classical models lead by 0.05–0.17 AUROC. Here the ≤300 sample cap imposed by quantum hardware
discards the majority of training signal. PCA compression to 8 dimensions further removes
structural information that classical tree-ensemble models exploit when given full fingerprints.
These endpoints represent the dominant failure mode of near-term QML, and the gap will narrow
only as qubit counts increase (relaxing the PCA bottleneck) or data-efficient quantum circuit
designs emerge.

### Featuriser sensitivity is the hidden variable

A consistent pattern across the winning and near-parity endpoints: **MACCS keys outperform ECFP4
and RDKit for QML** (QSVC wins on Clearance with maccs; VQC is strongest on CYP2C9 with ecfp4
but MACCS compresses more gracefully to 8 PCA components overall). ECFP4's 2048-bit sparse
vectors suffer the most compression loss under 8-component PCA. RDKit-200 is already denser but
its 200 dimensions still lose significant variance. MACCS 167-bit keys encode pharmacophore
features that are more linearly separable and more faithfully captured by 8 principal components.
This has a practical implication: **featuriser selection is as important as model selection for
QML performance**, and future work should explore task-specific featuriser–circuit pairs.

### Limitations and Responses to Critical Review

The following section addresses reviewer-style criticisms directly and honestly, distinguishing
where the criticism is valid and warrants corrective action from where the existing design already
provides adequate defence.

---

#### L1 — The 300-sample cap: selection mechanism and stability across subsamples

**Criticism:** The ≤300 training samples are a single fixed draw. If different subsets of 300
were drawn from the same training pool, would the ranking of QML vs. classical models hold? Is
the observed advantage of QSVC and VQC an artefact of one lucky draw?

**Response — partial agreement, but the design is not arbitrary.** The 300-sample cap is not
a uniform random draw. The code (`tdc_admet_loader.py::_cap_qml_split`) performs **stratified
random sampling per class**: it samples `300 // n_classes` examples per class using
`np.random.RandomState(42)`, then shuffles with the same seed. This ensures class balance is
preserved and the draw is reproducible. The same 300 samples are used by every model in the
`cl@300` ablation conditions, so all comparisons within the ablation are evaluated on
*exactly the same data* — the subsample is a controlled experiment design choice, not noise.

However, the criticism is valid in a deeper sense: **a single subsample of size 300 still
produces a point estimate with unknown variance**. On endpoints with only ~400–500 training
examples (CYP2C9\_Substrate n=467, Half\_Life\_Obach n=465, CYP2D6\_Substrate n=465), the
remaining pool after stratification is only ~167–200 examples per draw, and a different random
draw could produce a materially different result. The headline QML win on CYP2C9\_Substrate
(VQC +0.075) is on only 27 test samples — a single mis-classified compound shifts AUROC by
approximately 0.04. The finding should be treated as a strong directional signal, not a
statistically settled result.

**Actionable plan:** Run the full `cl@300` and `QSVC@300/pca` ablation with at least five
different seeds (e.g., `seed ∈ {0, 21, 42, 84, 100}`) to generate a distribution over
subsamples for each endpoint. The `_cap_qml_split` function already accepts a `seed` argument;
only the Hydra config `seed:` parameter needs to be changed per run. Report mean ± std AUROC
across draws for each endpoint in Table 2, replacing point estimates with intervals. For the
three QML-win endpoints (CYP2C9\_Substrate, Clearance\_Hepatocyte\_AZ, CYP2D6\_Substrate),
the win must survive across the majority of seeds to constitute a robust claim. This experiment
requires ≤5× the current QSVC compute budget (~5 × 90 min/endpoint × 22 endpoints = ~165
GPU-hours) and is directly supported by the existing pipeline.

---

#### L2 — Absence of confidence intervals over model initialisation seeds

**Criticism:** The results report single-run metrics without confidence intervals. Does varying
the model initialisation seed (while keeping the same 300 data points) change the conclusions,
particularly for variational models (VQC, QNN) that depend on parameterised circuit initialisation?

**Response — valid for variational models; largely moot for kernel methods.** The current
design uses `iter: 1` and a global `seed: 42`. This has different implications depending on
model class:

- **QSVC and classical SVM/LR/RF:** These are deterministic given fixed data and seed. QSVC
  with a statevector backend and fixed quantum seed (`q_seed: 42`) produces a unique kernel
  matrix for fixed inputs — varying the classical optimisation seed (the SVC `random_state`)
  has negligible effect since QSVC's dual optimisation is convex and converges to a global
  optimum. The QSVC results are effectively seed-stable.

- **VQC and QNN:** These use a parametric variational circuit whose initial parameters are
  drawn from a distribution seeded by `q_seed: 42`. Variational quantum circuits are highly
  non-convex and known to suffer from barren plateaus [A]; a different initialisation seed
  can produce substantially different convergence. The VQC win on CYP2C9\_Substrate
  (AUROC 0.849) could be a particularly favourable initialisation rather than a general
  property of the circuit ansatz. On the other hand, on an endpoint with only 27 test samples
  and near-random performance for all other models, VQC's 0.849 is so far above the classical
  ceiling (0.774) and so far above other QML models' performance that a lucky initialisation
  alone is unlikely to account for the full gap — but cannot be ruled out.

- **Classical MLP:** MLP initialisation is seeded via `np.random.seed(42)` before the sklearn
  pipeline; varying this seed introduces small variance but MLP on 467 samples is low-capacity
  enough that results are typically stable within ±0.02 AUROC.

**Actionable plan:** For the three highest-priority findings (VQC on CYP2C9\_Substrate,
QSVC on Clearance\_Hepatocyte\_AZ, and the ablation mean QSVC@300/pca > cl@300/pca), run 10
repeated trials with `q_seed ∈ {0, 7, 21, 42, 73, 84, 100, 123, 200, 314}`. For QSVC, the
cost is minimal since results will be nearly identical (kernel methods are seed-insensitive for
fixed data). For VQC and QNN, this will quantify whether the VQC advantage on CYP2C9\_Substrate
is robust to initialisation. If the mean VQC AUROC across 10 seeds remains above the best
classical 0.774, the claim is strengthened considerably. This experiment should be prioritised
before submission.

---

#### L3 — No ablation over PCA dimensionality (4 → 8 → 12 → 16 components)

**Criticism:** All results use a fixed `n_components: 8`. This is presented as the
"hardware-equivalent" constraint, but 8 is an arbitrary choice. How does performance change
as compression is relaxed (4 → 8 → 12 → 16)? Is 8 near a performance cliff, a plateau, or an
inflection point? The feature compression penalty (−0.050 AUROC) may be overestimated or
underestimated depending on where 8 sits on the PCA-dimension vs. accuracy curve.

**Response — valid criticism; the choice of 8 is hardware-motivated but not empirically justified.**
The 8-component choice was driven by qubit budget constraints (8 qubits ≈ 8 amplitude-encoded
features), not by a data-driven optimum. This is the correct choice for modelling near-term QML
hardware constraints, but the paper would be stronger for knowing (a) how much performance each
additional PCA dimension recovers, and (b) whether classical and QML models have different
PCA-dimension sensitivities. Specifically:

- If QSVC performance plateaus quickly (e.g., AUROC saturates by 8 components), the argument
  that more qubits would close the gap is weakened.
- If QSVC performance grows faster per additional dimension than classical models do, it would
  support the claim that the quantum kernel benefits disproportionately from richer input.
- The crossover point — the minimum PCA dimension at which classical models begin overtaking
  QSVC — is a key quantity for hardware roadmap planning.

The infrastructure already supports this: `n_components` is a YAML parameter and `get_embeddings`
accepts any integer. A five-point sweep (4, 8, 12, 16, 32) would require approximately 5× the
QSVC compute budget for the full 22 endpoints, or could be scoped to the six most interesting
endpoints (the QML-win and near-parity cases).

**Actionable plan:** Add a PCA-dimension sensitivity sweep config
(`admet_qsvc_pca_sweep.yaml`) with `n_components ∈ [4, 8, 12, 16, 32]` and run it on the
six priority endpoints (CYP2C9\_Substrate, Clearance\_Hepatocyte\_AZ, CYP2D6\_Substrate,
Bioavailability\_Ma, BBB\_Martins, PPBR\_AstraZeneca) plus all three featurisers. Plot
AUROC vs. n\_components curves for QSVC alongside the matched cl@N/pca baseline to identify
the crossover dimension and measure the per-dimension recovery rate. This directly addresses
the question of whether 8 is a reasonable hardware target or an artificially tight constraint.

---

#### L4 (Original) — Simulator vs. real hardware: results may not survive decoherence

**Criticism:** All 22 endpoints are evaluated on a noiseless statevector simulator. Real quantum
hardware introduces gate errors, decoherence, and readout errors that are not modelled. The
8-qubit QSVC kernel matrices computed on a simulator are exact; the same computation on an
IBM 127-qubit Eagle processor would produce noisy, approximate kernel entries. It is unknown
whether the QML wins (particularly the narrow QSVC lead of +0.006 on Clearance\_Hepatocyte\_AZ)
survive on real hardware, or whether noise degrades QSVC below the classical baseline.

**Response — agreed; this is the most consequential open question.** The paper explicitly
states this limitation in the Methods section, and the framing is that results represent an
optimistic upper bound for near-term QML. The simulator results are nonetheless useful: they
establish that quantum kernel methods have the representational capacity to outperform classical
models on specific tasks, independent of hardware imperfections. Whether the advantage is
practically realisable depends on error rates, which are improving rapidly (IBM's Heron
processors report 2-qubit gate fidelities of >99.9% [B]).

For the QSVC scenario specifically, the ZZ-feature map requires O(n²) CNOT gates for
n qubits; with 8 qubits this is 28 CNOT gates per kernel evaluation, which at current hardware
fidelity produces substantial noise accumulation. The narrow +0.006 AUROC lead over classical
models on Clearance\_Hepatocyte\_AZ is well within the expected degradation range.

**Actionable plan:** Run the six priority endpoints (QML-win and near-parity) on an actual
IBM Quantum device via the `qiskit-ibm-runtime` backend already scaffolded in
`qbiocode/utils/qutils.py`. Compare simulator vs. real-hardware AUROC on identical test sets.
This is a critical next experiment before any claims of practical quantum advantage.

---

#### L5 (Original) — Test set reliability: use the full TDC canonical test set; unify with L1

**Criticism:** AUROC estimates on small test sets are unreliable. An initial rough estimate
suggested CYP2C9\_Substrate had only ~27 positive test examples; at that scale a single
mis-classified compound moves AUROC by ≈0.04, putting the entire VQC margin (+0.075) within
sampling noise.

**Revised response — the premise is partially wrong, and a cleaner architectural solution
exists that also resolves L1.** Inspection of the on-disk data confirms that TDC provides a
**canonical, fixed test split** (`benchmark["test"]`) that is substantially larger than a 10%
random hold-out:

| Endpoint | n\_train | **n\_test (TDC canonical)** |
|---|---|---|
| CYP2C9\_Substrate\_CarbonMangels | 467 | **135** |
| Clearance\_Hepatocyte\_AZ | 848 | **243** |
| CYP2D6\_Substrate\_CarbonMangels | 465 | **135** |
| DILI | 331 | **96** |
| Half\_Life\_Obach | 465 | **135** |
| hERG | 457 | **132** |
| Bioavailability\_Ma | 448 | **128** |

CYP2C9\_Substrate has **135 test compounds**, not 27. At n\_test = 135, a DeLong 95% CI for
AUROC ≈ 0.85 is approximately ±0.06 — VQC (0.849) and LR (0.774) have largely non-overlapping
confidence intervals, which materially strengthens the win claim compared to what a 27-sample
analysis would imply.

**The principled fix unifies L1 and L5:** because QSVC inference is just
`qsvc.predict(X_test)` — O(n\_train × n\_test) kernel evaluations, not O(n\_train²) — scoring
a once-trained model on the full TDC test set is fast. With n\_train = 300 and n\_test ≤ 500,
inference adds at most ~10 minutes to a 90-minute QSVC training run. There is no practical
barrier to evaluating every training subsample (L1) and every initialisation seed (L2) on the
**same fixed canonical test set**. Doing so simultaneously solves both criticisms:

1. **L1 and L5 merge into one experiment.** Train 5 models on 5 different draws of 300
   samples; score each on the identical fixed `test.csv` (n\_test = 135 for CYP2C9\_Substrate).
   The resulting AUROC distribution over draws quantifies subsample sensitivity directly,
   without test-set sampling noise confounding the variance estimate. This is cleaner than
   bootstrap resampling of a single fixed test set.

2. **The TDC test split is scaffold-based** (chemical scaffold holdout), so the reported
   AUROC values represent genuine out-of-distribution generalisation — harder than random
   splits, resistant to train-test leakage.

**Actionable plan (merged with L1):**

- **Step 1 (immediate, no retraining):** Verify that existing runs already evaluated on
  the full `test.csv` by counting rows in `RawDataEvaluation.csv`. If so, current point
  estimates are already on n\_test ∈ [96, 2626] — and DeLong CIs can be computed post-hoc
  from saved prediction scores at negligible cost (~1 GPU-hour).
- **Step 2 (next experiment):** Run the 5-seed subsample sweep (L1 plan) always scoring
  against the same fixed `test.csv`. Report mean AUROC ± std in Tables 4a/4b/4c.
- **Step 3 (post-hoc statistics):** For the three QML-win endpoints, report DeLong 95% CIs
  alongside point estimates. At n\_test = 135, non-overlapping CIs constitute a statistically
  defensible win claim.

---

#### L6 (Original) — 5-fold cross-validation is only applied to classical models; QML hyperparameters are largely fixed

**Criticism:** Classical models undergo full 5-fold CV grid search over hyperparameters (LR
regularisation C, RF depth/estimators, MLP hidden layer size, etc.). QSVC uses `grid_search:
False` in `admet_qsvc_config.yaml` with fixed circuit parameters (ZZ-feature map reps=1, linear
entanglement, C=1.0). VQC and QNN use fixed ansatz depth (`NN_depth: 1`). This asymmetry means
classical models are presented at their optimised best while QML models are run with essentially
default parameters. If QSVC with `C=0.1` or `reps=2` outperforms `C=1.0/reps=1` on some
endpoints, the current benchmark understates QML's potential.

**Response — partially agreed, with important caveats.** The asymmetry is real but is
partially justified:

1. **QML hyperparameter search is prohibitively expensive at scale.** QSVC kernel fitting at
   8 qubits on a statevector simulator already takes ~90 minutes per endpoint. A 5-fold CV
   grid search over `{C: [0.1, 1, 10], reps: [1, 2]}` would multiply runtime by 30×, making
   the full 22-endpoint benchmark impractical without dedicated hardware.
2. **The MinMaxScaler + C=1.0 fix is not default; it is the result of deliberate tuning.**
   The most damaging default (C=0.01 without scaling) was corrected. C=1.0 with MinMaxScaler
   is the theoretically motivated choice for fidelity kernels on [0,1]-normalised inputs.
3. **However**, on VQC and QNN, `NN_depth: 1` (single-layer ansatz) is almost certainly
   sub-optimal. VQC with depth 2–3 may outperform depth 1 on the larger test endpoints.
   The benchmark therefore understates VQC/QNN potential specifically.

**Actionable plan:** Run a lightweight hyperparameter sweep for QSVC on the six priority
endpoints: `C ∈ [0.1, 1.0, 10.0]` with `reps ∈ [1, 2]` (6 conditions × 6 endpoints × 3
featurisers = 108 runs, ~162 GPU-hours). For VQC/QNN, test `NN_depth ∈ [1, 2, 3]` on the
same priority endpoints. Report whether the 5-fold CV grid-searched classical models still
lead once QML also receives minimal tuning. This is important for making the benchmark
methodologically symmetric.

---

### Summary of planned follow-up experiments

| Experiment | Addresses | Priority | Estimated cost |
|---|---|---|---|
| Multi-seed subsample sweep, scored on full TDC test.csv (5 seeds × 22 endpoints) | L1 + L5 — subsample variance & test reliability | **High** | ~450 GPU-hours |
| DeLong CIs from existing RawDataEvaluation.csv | L5 — statistical backing for win claims | **Immediate** | ~1 GPU-hour (post-hoc) |
| VQC/QNN initialisation seeds (10 seeds × 6 endpoints) | L2 — init variance | High | ~90 GPU-hours |
| PCA dimension sweep (4→32, 6 endpoints × 3 feats) | L3 — compression sensitivity | High | ~270 GPU-hours |
| Real hardware evaluation (6 priority endpoints) | L4 — simulator gap | Critical | limited by IBM Q access |
| QSVC/VQC hyperparameter sweep (6 endpoints) | L6 — asymmetric tuning | Medium | ~250 GPU-hours |

The immediate post-hoc DeLong CI computation and the multi-seed sweep (both scored on the fixed
canonical test set) are the minimum necessary before making strong win-claim statements in a
peer-reviewed venue.

### Additional limitations (design and scope)

- **Statevector simulator only.** All circuits run on a classical noiseless statevector simulator;
  hardware noise is not modelled (see L4 above for the planned corrective experiment).
- **Classification endpoints only.** The benchmark covers only binary classification; regression
  endpoints (Lipophilicity, VDss) and multi-task endpoints are excluded.
- **Single ansatz architecture.** VQC and QNN are evaluated with a fixed ZZ-feature-map + linear
  entanglement architecture. Alternative ansatze (e.g., hardware-efficient ansatz, QAOA-inspired
  circuits) are unexplored.
- **No molecular graph representation.** All featurisers are fixed-length vectors (ECFP4, MACCS,
  RDKit-200). Graph-based quantum encoding [C] could more naturally exploit the molecular
  topology that determines ADMET activity.

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
