# QML-ADMET Benchmark: Gist

## Motivation

Whether quantum kernel methods offer practical advantage over classical ML on small, real-world molecular datasets is an open and contested question. To the best of knowledge, no prior QML comparisons on molecular property prediction tasks (e.g., ADMET) test the effectiveness of quantum machine learning methods versus classical machine learning methods over a comprehensive ADMET benchmark experiment. This experiment was designed to answer the question.

---

## Experiments — Baseline and Ablation

Two experiments were run in the baseline phase: a full model comparison and a confounder ablation. Together they comprised approximately 650 model training and evaluation runs.

**Full model comparison.** Five QML classifiers (QSVC, VQC, QNN, PQK, QEnsemble) and five classical classifiers (LR, RF, MLP, XGBoost, SVC) were evaluated across all 22 ADMET classification endpoints from TDC, using three molecular featurizers: ECFP4 (2048 bits), MACCS keys (167 bits), and RDKit-200 (200 descriptors). TDC provides pre-defined scaffold-based train/validation/test splits for each endpoint; the test set is TDC's fixed held-out set and is never modified. The training portion is further divided: QML models train on a stratified subsample of up to 300 points drawn from the TDC training split (seed 42); classical models train on the full TDC training split. All five QML models run on an 8-qubit statevector simulator; QSVC additionally requires `C=1.0` and MinMaxScaler input normalization to prevent exponential kernel concentration. This yields 22 endpoints × 3 featurizers = 66 combinations. However, one combination (CYP3A4\_Substrate/maccs) failed to produce output due to a pipeline error and is excluded, resulting in 65 valid endpoint–featurizer combinations per model.

**Confounder ablation.** To measure the independent contribution of the size of training data, dimensionality reduction, and the quantum kernel to the classical–quantum performance gap, classical models were also trained under two matched conditions: capped at 300 training samples with raw features (cl@300/none), and capped at 300 samples with 8-component PCA (cl@300/pca) or UMAP (cl@300/umap). Combined with the full-data classical run (cl@full/none) and QSVC at 300 samples with PCA or UMAP (QSVC@300/pca, QSVC@300/umap), sequential differences isolate each penalty: the size of training set = cl@full minus cl@300/none; feature compression = cl@300/none minus cl@300/pca; direct gap between classical and quantum = cl@300/pca minus QSVC@300/pca.

---

## Experiments — Robustness Ablations

Then, three robustness experiments followed.

**Phase 1 (subsample variance):** QSVC with 8-component PCA was re-run across 12 subsample seeds per endpoint–featurizer combination, scoring on the fixed TDC test set, to quantify how much of the gap between classical ML and QML is driven by the single seed-42 subsample draw. 

**Phase 2 (VQC initialization variance):** VQC was re-run with 3–47 distinct circuit initialization seeds (`q_seed`) on 3 priority endpoints (CYP2C9\_Substrate, CYP2D6\_Substrate, Clearance\_Hepatocyte\_AZ) × 3 featurizers × all available embeddings; these three endpoints were selected because they showed the strongest QML signals in the baseline. The goal was to determine whether the result, VQC AUROC of 0.849 on CYP2C9\_Substrate, is reproducible or an initialization artifact. 

**Phase 3 (qubit-count sweep):** QSVC was re-run with PCA by varying degrees of K (4,8,12,16,32), where K equals the number of qubits. We first ran K=4 on all 22 endpoints, and then ran K=12/16/32 on the same 6 priority endpoints (945 total runs).

---

## Results — Baseline and Statistical Significance

Classical models outperform QML in the full-data comparison (Table 1). Under hardware-equivalent matched conditions (Table 2), the gap between classical ML and QML is −0.044 AUROC for both PCA and UMAP embeddings, preceded by a data-size-matched result of −0.040. This means data starvation and the quantum kernel each contribute roughly equally to the total classical advantage. QSVC exceeds the classical baseline on 3 of 65 endpoint–featurizer combinations (Clearance/maccs +0.041, Pgp/ecfp4 +0.020, Solubility/rdkit200 +0.004). However, the DeLong 95% confidence intervals overlap for all three, so there is no statistical significance (Table 3).

**Table 1 — Mean test AUROC, 65 endpoint–featurizer combinations**

| Model | Type | Mean AUROC | Std |
|---|---|---|---|
| MLP | Classical | 0.797 | 0.095 |
| RF | Classical | 0.787 | 0.093 |
| LR | Classical | 0.786 | 0.093 |
| XGBoost | Classical | 0.763 | 0.093 |
| SVC | Classical | 0.705 | 0.107 |
| **QSVC** | **QML** | **0.653** | **0.103** |
| PQK | QML | 0.615 | 0.144 |
| VQC | QML | 0.611 | 0.111 |
| QNN | QML | 0.603 | 0.091 |
| QEnsemble | QML | 0.541 | 0.070 |

**Table 2 — Four-condition ablation decomposing the C↔Q gap**

| Condition | Mean AUROC | Step Δ |
|---|---|---|
| cl@full / none (classical, raw, full data) | 0.804 | — |
| cl@300 / none (classical, raw, ≤300 samples) | 0.764 | −0.040 (data starvation) |
| cl@300 / pca (classical, 8-PCA, ≤300) | 0.803 | +0.039 (PCA regularises classical) |
| cl@300 / umap (classical, 8-UMAP, ≤300) | 0.771 | — |
| QSVC@300 / pca | 0.759 | **−0.044 vs cl@300/pca** |
| QSVC@300 / umap | 0.727 | **−0.044 vs cl@300/umap** |

**Table 3 — DeLong 95% CI on the three nominal QSVC wins**

| Endpoint | Featuriser | QSVC AUROC [95% CI] | Best Classical [95% CI] | Δ | Definitive? |
|---|---|---|---|---|---|
| Clearance\_Hepatocyte\_AZ | maccs | 0.756 [0.557–0.955] | RF 0.715 [0.525–0.905] | +0.041 | No |
| Pgp\_Broccatelli | ecfp4 | 0.940 [0.704–1.000] | MLP 0.920 [0.688–1.000] | +0.020 | No |
| Solubility\_AqSolDB | rdkit200 | 0.790 [0.718–0.863] | LR 0.786 [0.714–0.858] | +0.004 | No |

---

## Results — Robustness Experiments

All three robustness experiments fail to show the QML advantages. Phase 1 (seed stability): mean cross-seed AUROC standard deviation for QSVC is **0.084** (6σ above the 0.05 stability threshold); only 4 of 66 endpoint–featurizer combinations are stable. The seed-averaged delta vs. best classical is **−0.180**, compared to −0.044 at seed 42 alone, which means the seed-average gap is even larger than the initial single-seed result. Phase 2 (VQC init): VQC mean AUROC across the q_seeds is **0.518 ± 0.046**. None of 9 endpoint–featurizer combinations show a higher score for QMLs than classical ones. Phase 3 (K sweep): K=8 is empirically optimal (mean AUROC 0.642, optimal in 61% of combinations). Performance declines at K=12 and K=16. At K=32, QSVC AUROC = 0.500 ± 0.000 on all 18 tested combinations, which means it failed to converge (Table 4). This is a direct empirical demonstration of exponential kernel concentration: with ~240 training points and a 2³²-dimensional Hilbert space, all pairwise kernel values collapse to a constant and QSVC learns no boundary. The empirical results approximately follow the rule **2^K ≈ n_train** (K=8 for n≈300).

**Table 4 — QSVC mean AUROC by K (PCA components = qubits), averaged over all tested endpoints**

| K | Hilbert dim. | Mean AUROC | Std | Optimal in N combos |
|---|---|---|---|---|
| 4 | 16 | 0.618 | 0.068 | 20 / 66 |
| **8** | **256** | **0.642** | **0.106** | **40 / 66** |
| 12 | 4,096 | 0.592 | 0.068 | 2 / 66 |
| 16 | 65,536 | 0.592 | 0.069 | 4 / 66 |
| 32 | ~4.3 billion | **0.500** | **0.000** | 0 / 66 |
