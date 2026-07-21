# QBioCode-ADMET: A Rigorous QML Benchmark Suite for ADMET Property Prediction

**Branch:** `admet`  
**Target Venues:** *Nature Machine Intelligence* / *Journal of Chemical Information and Modeling*  
**Priority:** ★★★★★

---

## 1. Problem Statement

Every QML drug-discovery paper uses different dataset splits, different classical baselines, and
different evaluation metrics. The field lacks a definitive answer to "Is QML better than classical
ML for ADMET?" because no fair head-to-head study exists. QBioCode already implements QProfiler
(automated benchmarking across 15+ data complexity metrics) and QSage (meta-learning for model
selection) — but has never been applied to standardized ADMET data.

---

## 2. Core Idea

Extend QBioCode's QProfiler and QSage to the full TDC ADMET Benchmark Group
(https://tdcommons.ai/benchmark/admet_group/overview/) using:

1. **QBioCode quantum models** (QSVC, PQK, VQC, QNN, QEnsemble) as the QML panel.
   - `qensemble` is registered in `compute_ml_dict` via a wrapper matching the standard
     `compute_*(X_train, X_test, y_train, y_test, args, model, data_key, ...)` signature.
2. **SOTA classical baseline**: MMELON (`ibm-research/biomed.sm.mv-te-84m`, HuggingFace) —
   frozen embeddings + sklearn classification head.
3. **Standard classical baselines**: RF, SVC, MLP, XGBoost, LR (already in QBioCode).
4. **Data complexity profiling** via QProfiler's 23 metrics to characterize *when* QML outperforms
   classical models.
5. **QSage meta-model**: surrogate trained on (dataset complexity features → best model) to
   produce a practical decision rule for practitioners.

---

## 3. Repository Structure (admet branch additions)

```
QBioCode/
├── paper_plan.md                              ← this file
├── pyproject.toml                             ← admet optional-deps group added
├── qbiocode/
│   ├── apps/qprofiler/configs/
│   │   └── admet_config.yaml                 ← QProfiler config for ADMET sweep
│   ├── data_adapters/                         ← NEW subpackage
│   │   ├── __init__.py
│   │   ├── tdc_admet_loader.py               ← TDC → QBioCode CSV pipeline
│   │   └── molecular_featurizers.py          ← ECFP4, MACCS, RDKit200 → numpy
│   ├── learning/
│   │   └── compute_mmelon.py                 ← NEW: MMELON baseline
│   ├── evaluation/
│   │   └── model_evaluation.py               ← MODIFIED: +AUPRC, +MCC, +checkpoint hook
│   └── utils/
│       └── model_checkpoint.py               ← NEW: save/load/infer checkpoint API
├── experiments/admet_benchmark/              ← NEW: paper experiment scripts
│   ├── 01_prepare_admet_datasets.py
│   ├── 02_run_qprofiler_admet.sh
│   ├── 03_train_qsage_admet.py
│   ├── 04_test_inference.py
│   ├── 05_analysis.ipynb
│   └── configs/
│       └── admet_batch_config.yaml
└── tutorial/ADMET_Benchmark/                 ← NEW: community tutorial
    └── admet_benchmark_tutorial.ipynb
```

---

## 4. Key Design Decisions

### 4.1 Task Type: Binary Classification Only
QBioCode's QML models (QSVC, QSVC, VQC, QNN, PQK) are classifiers. All TDC regression
endpoints (Caco-2, Lipophilicity, Solubility, LD50, VDss, PPBR, Clearance, Half-Life) are
**binarized at TDC-recommended thresholds** at data-preparation time. This:
- Keeps QML models valid
- Aligns with TDC leaderboard evaluation (AUROC)
- Preserves all 23 complexity metrics in `evaluate()`

### 4.2 Featurization: Three Molecular Fingerprint Schemes
For each endpoint, three independent feature sets are generated:
- **ECFP4**: 2048-bit Morgan fingerprints (radius=2), bit-vector
- **MACCS**: 167-bit MACCS key fingerprints
- **RDKit200**: 200 RDKit 2D physicochemical descriptors, z-score standardized

Each featurizer produces `train.csv`, `valid.csv`, `test.csv` in
`data/admet/{endpoint}/{featurizer}/` — last column is always the binary label (QBioCode convention).

### 4.3 QML Sample Cap
Quantum kernel methods (QSVC, PQK) scale O(n²). Training sets are capped at **300 stratified
samples** for QML models. Classical models and MMELON use the full training set. The cap is
applied inside the data loader and flagged in `metadata.json` per endpoint.

### 4.4 Model Checkpointing (New Feature)
Current QBioCode discards fitted model objects after prediction. For ADMET:
- `utils/model_checkpoint.py` adds `save_checkpoint()` / `load_best_checkpoint()` / `infer_from_best()`
- `modeleval()` gains an optional `checkpoint_dir` parameter (backward-compatible)
- Best-by-validation-F1 selection: `best_models.json` indexes the optimal checkpoint per
  `(endpoint, model)` pair
- `dill` serialization (already a dep) handles Qiskit ML objects

### 4.5 Extended Metrics
`modeleval()` gains two new optional metrics for ADMET:
- **AUPRC** (`average_precision_score`) — for imbalanced endpoints (DILI, hERG, CYP*)
- **MCC** (`matthews_corrcoef`) — standard for imbalanced binary tasks

Both are backward-compatible (returned only when `task='admet'` or explicitly requested).

### 4.6 MMELON Baseline
`compute_mmelon.py` follows the exact same signature as all `compute_*.py` functions:
```python
compute_mmelon(X_train, X_test, y_train, y_test, args, model, data_key, ...)
```
SMILES strings are passed through the HuggingFace tokenizer; CLS pooled embeddings are extracted
from `ibm-research/biomed.sm.mv-te-84m` (frozen). A sklearn RF classification head is fitted on
the embeddings. Embeddings are cached to `data/mmelon_cache/` to avoid recomputation.

---

## 5. TDC ADMET Endpoints (22 total)

| # | Endpoint | Category | Original Task | Binarize? | Threshold |
|---|----------|----------|---------------|-----------|-----------|
| 1 | Caco2_Wang | Absorption | Regression | ✓ | −5.15 log cm/s |
| 2 | HIA_Hou | Absorption | Binary | ✗ | — |
| 3 | Pgp_Broccatelli | Absorption | Binary | ✗ | — |
| 4 | Bioavailability_Ma | Absorption | Binary | ✗ | — |
| 5 | Lipophilicity_AstraZeneca | Distribution | Regression | ✓ | 2.0 logD |
| 6 | Solubility_AqSolDB | Distribution | Regression | ✓ | −3.0 log mol/L |
| 7 | BBB_Martins | Distribution | Binary | ✗ | — |
| 8 | PPBR_AstraZeneca | Distribution | Regression | ✓ | 90% bound |
| 9 | VDss_Lombardo | Distribution | Regression | ✓ | 0.71 L/kg |
| 10 | CYP2C19_Veith | Metabolism | Binary | ✗ | — |
| 11 | CYP2D6_Veith | Metabolism | Binary | ✗ | — |
| 12 | CYP3A4_Veith | Metabolism | Binary | ✗ | — |
| 13 | CYP1A2_Veith | Metabolism | Binary | ✗ | — |
| 14 | CYP2C9_Veith | Metabolism | Binary | ✗ | — |
| 15 | CYP2C9_Substrate_CarbonMangels | Metabolism | Binary | ✗ | — |
| 16 | CYP2D6_Substrate_CarbonMangels | Metabolism | Binary | ✗ | — |
| 17 | CYP3A4_Substrate_CarbonMangels | Metabolism | Binary | ✗ | — |
| 18 | Half_Life_Obach | Excretion | Regression | ✓ | median split |
| 19 | Clearance_Hepatocyte_AZ | Excretion | Regression | ✓ | median split |
| 20 | hERG | Toxicity | Binary | ✗ | — |
| 21 | AMES | Toxicity | Binary | ✗ | — |
| 22 | DILI | Toxicity | Binary | ✗ | — |

---

## 6. Execution Phases

### Phase 1 — Data Infrastructure (Week 1–2)
**Files:** `qbiocode/data_adapters/tdc_admet_loader.py`, `molecular_featurizers.py`

- `tdc_admet_loader.py`: wraps `tdc.benchmark_group.admet_group.get()` → serializes canonical
  train/valid/test splits; applies binarization; caps QML subset; writes `metadata.json`
- `molecular_featurizers.py`: ECFP4 / MACCS / RDKit200 pipelines; returns numpy arrays and writes
  `{split}_{featurizer}.csv` files with label in last column
- Validation: run `01_prepare_admet_datasets.py` → 22 endpoints × 3 featurizers × 3 splits = 198 CSVs

**Output:**
```
data/admet/
├── metadata.json
├── Caco2_Wang/
│   ├── ecfp4/train.csv, valid.csv, test.csv
│   ├── maccs/train.csv, valid.csv, test.csv
│   └── rdkit200/train.csv, valid.csv, test.csv
└── ...
```

### Phase 2 — MMELON Baseline (Week 2–3)
**Files:** `qbiocode/learning/compute_mmelon.py`, update `evaluation/model_run.py`

- Implements `compute_mmelon()` matching existing `compute_*.py` signature
- Loads `ibm-research/biomed.sm.mv-te-84m` via HuggingFace `transformers`
- Caches embeddings per dataset key to `data/mmelon_cache/`
- RF head for classification (consistent with MMELON intended use)
- Registered as `"mmelon"` in `compute_ml_dict` in `model_run.py`

### Phase 3 — Checkpoint Layer (Week 3)
**Files:** `qbiocode/utils/model_checkpoint.py`, modify `evaluation/model_evaluation.py`

- `save_checkpoint(model_obj, name, dataset, split_id, val_f1, out_dir)` — dill serialize
- `load_best_checkpoint(name, dataset, checkpoint_dir)` — reads `best_models.json`
- `infer_from_best(name, dataset, X_test, checkpoint_dir)` — load + predict
- `modeleval()` gains `checkpoint_dir=None` and `fitted_model=None` optional params

### Phase 4 — QProfiler Sweep (Week 3–6, compute-heavy)
**Files:** `qbiocode/apps/qprofiler/configs/admet_config.yaml`, `experiments/admet_benchmark/02_run_qprofiler_admet.sh`

Config highlights:
- `model: ['svc','rf','mlp','xgb','lr','qsvc','vqc','qnn','pqk','mmelon']`
- `embeddings: ['none','pca']`
- `n_components: 10` (10-qubit budget)
- `iter: 5` (5 train/valid splits)
- `stratify: ['y']` (imbalanced endpoints)
- `backend: simulator` for full sweep; `ibm_least` for top-3 highlighted endpoints
- `checkpoint_dir: results/admet_benchmark/checkpoints`

**Compute estimate:** 22 × 3 × 10 × 5 × 2 = 6,600 model fits. With QML 300-sample cap and
40-core HPC: ~72h wall time. Uses `qprofiler-batch` with `--n-jobs 40` and `--checkpoint` restart.

### Phase 5 — QSage + Analysis (Week 6–8)
**Files:** `experiments/admet_benchmark/03_train_qsage_admet.py`, `04_test_inference.py`, `05_analysis.ipynb`

1. Compile results: `combine_results()` → `compiled_ModelResults.csv`
2. QSage: `xgboost_optuna` mode, 200 trials, 10-fold CV; 4 held-out endpoints as QSage test set
3. SHAP on trained XGBoost QSage → top complexity features predicting QML superiority
4. Decision rule: threshold classifier on top-2 features → human-readable rule
5. Test inference: `infer_from_best()` on TDC canonical test splits → final paper metrics
6. TDC leaderboard comparison via official TDC evaluation functions

---

## 7. New Dependencies

| Package | Purpose | Added to pyproject.toml |
|---------|---------|------------------------|
| `rdkit>=2023.3` | ECFP4, MACCS, 2D descriptors | `[admet]` group |
| `PyTDC>=0.4.1` | TDC dataset loader + canonical splits | `[admet]` group |
| `transformers>=4.40` | MMELON (HuggingFace model) | `[admet]` group |
| `shap>=0.45` | Decision rule extraction | `[admet]` group |
| `torch` | MMELON inference | Already in requirements-base.txt |
| `dill` | Checkpoint serialization | Already in requirements-base.txt |

Install: `pip install 'qbiocode[admet]'`

---

## 8. Extended Metrics (additions to modeleval)

| Metric | Function | Use case |
|--------|----------|----------|
| AUROC | `roc_auc_score` | Already present — primary TDC metric |
| AUPRC | `average_precision_score` | Imbalanced endpoints (DILI, hERG, CYP*) |
| MCC | `matthews_corrcoef` | Imbalanced binary tasks, standard in drug discovery |

---

## 9. Expected Paper Contributions

1. **First standardized QML-vs-classical ADMET benchmark** with SOTA foundation model baseline
   (MMELON) and canonical TDC splits — directly comparable to TDC leaderboard entries.
2. **Empirical decision rule**: "Use QML when dataset size < N AND intrinsic dimensionality > D"
   — derived from QSage SHAP analysis across all 22 endpoints.
3. **Open benchmark artifact integrated into QBioCode** — community resource; `pip install
   'qbiocode[admet]'` reproduces the full benchmark.
4. **Complexity-stratified analysis**: 3–5 ADMET endpoint types where QML is robustly superior,
   characterized by QProfiler complexity metrics.

---

## 10. Timeline

| Week | Deliverable |
|------|-------------|
| 1–2 | Phase 1: 198 CSVs generated and validated |
| 2–3 | Phase 2: MMELON module passing QProfiler integration test |
| 3 | Phase 3: Checkpoint layer live; pilot on 2 endpoints |
| 3–6 | Phase 4: Full QProfiler sweep on HPC |
| 6–7 | Phase 5a: QSage trained; SHAP analysis; decision rule derived |
| 7–8 | Phase 5b: TDC leaderboard comparison; all figures |
| 8–10 | Writing: manuscript draft, supplement, artifact packaging |

---

## 11. Citation Target

> Smaldone et al., *Chem. Rev.* 2025 — most-cited QML drug-discovery review explicitly calls for
> a standardized QML-vs-classical ADMET benchmark with SOTA baselines.

```bibtex
@article{qbiocode_admet_2025,
  title   = {QBioCode-ADMET: A Rigorous Quantum Machine Learning Benchmark Suite
             for ADMET Property Prediction},
  journal = {Nature Machine Intelligence},
  year    = {2025}
}
```

---

## 12. Future Work

Items deferred from the current implementation for follow-up:

1. **Endpoint-level QSage held-out validation** *(Gap 2, deferred)*
   Currently, `03_train_qsage_admet.py` performs a row-level 80/20 train/test split
   of the compiled `ModelResults.csv`. A more rigorous evaluation would hold out
   **4 complete ADMET endpoints** (e.g., one per ADMET category) from QSage training
   entirely, then validate the meta-model's predictions on those unseen endpoints.
   This tests whether QSage generalises across endpoint types, not just across
   individual model-fit rows. Implementation: stratify the split by endpoint name
   before calling `train_sub_sages()`.

2. **Native regression support in QBioCode**
   All regression ADMET endpoints are currently binarized at intake. Adding
   `*Regressor` variants of all `compute_*.py` modules and extending `modeleval()`
   with a `task='regression'` branch (RMSE, MAE, R²) would allow QProfiler to handle
   regression endpoints natively without binarization, enabling a richer benchmark.

3. **Real-hardware QML runs on top-3 endpoints**
   The full sweep uses `backend: simulator`. For the paper's highlight results,
   re-run the 3 endpoints where QML shows the largest advantage on `ibm_least` real
   hardware to confirm that noise does not eliminate the QML benefit.

4. **QSage cross-benchmark generalisation**
   Train QSage on ADMET results and test prediction quality on a held-out
   non-ADMET benchmark (e.g., TDC ADME group or MoleculeNet) to demonstrate
   that the decision rule generalises beyond the training distribution.

5. **Featurizer-stratified QSage**
   Current QSage treats all three featurizers as independent data points.
   A featurizer-aware QSage (separate sub-sage per featurizer, or featurizer
   as an additional meta-feature) could produce sharper decision rules of the
   form: *"Use ECFP4 + QML when intrinsic dimensionality > D"*.
