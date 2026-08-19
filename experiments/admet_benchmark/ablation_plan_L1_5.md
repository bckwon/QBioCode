# Ablation Plan: L1–L5 Robustness Experiments
## QML-ADMET Benchmark — Follow-up Experiments

**Purpose:** Provide a step-by-step, optimised execution plan to address Limitations L1–L5
from `paper/summary.md`. This document is self-contained: a coding agent (Bob) reading only
this file should be able to implement and run every step from scratch.

**Repo root:** `/proj/bmfm/users/bckwon/projects/QBioCode`
**Python env:** `.venv/bin/python3` (all scripts use this interpreter)
**Cluster:** IBM LSF, A100 GPU nodes (`zu-a100-{b05,c06,c08}-*`)

---

## Background: What Each Limitation Requires

| ID | Issue | What we need to run |
|---|---|---|
| **L1** | Single fixed draw of 300 training samples — unknown variance | 4 additional seeds × 22 endpoints × 3 feats → QSVC + classical@300 |
| **L2** | No CI over VQC/QNN initialisation seeds | 9 additional `q_seed` values × 3 priority endpoints × 3 feats → VQC only |
| **L3** | No ablation over PCA dimensionality | 4 additional `n_components` values × 6 endpoints × 3 feats → QSVC + classical@300 |
| **L4** | Results only on statevector simulator | Real IBM Quantum hardware run for 6 endpoints (external dependency — not automated here) |
| **L5** | Test-set size concern (resolved in design: TDC test is fixed, large) | Post-hoc CI computation from existing `RawDataEvaluation.csv` — **no new training** |

**Key insight (L1 + L5 unified):** Every new training run must evaluate against the
**same fixed TDC `test.csv`** for each endpoint/featuriser. This is already how the pipeline
works — `test.csv` is never modified. Running 5 training seeds gives 5 AUROC values on the
same test set, directly quantifying subsample variance.

---

## Critical: What NOT to Re-run

The following artifacts already exist on disk and must **never** be regenerated:

| Artifact | Location | What it is |
|---|---|---|
| `data/admet/{endpoint}/{feat}/train.csv` | `data/admet/` | Full training split — **frozen** |
| `data/admet/{endpoint}/{feat}/test.csv` | `data/admet/` | TDC canonical test — **frozen** |
| `data/admet/{endpoint}/{feat}/valid.csv` | `data/admet/` | Validation split — **frozen** |
| `data/admet/{endpoint}/{feat}/train_qml.csv` | `data/admet/` | seed=42 subsample — **this is the seed-42 draw; keep it** |
| All results in `results/admet_config/` | `results/` | Full-data classical + all QML baseline — **do not re-run** |
| Compiled tables in `results/admet_benchmark/tables/` | `results/admet_benchmark/` | Final paper tables — **do not overwrite** |

New experiment outputs must always go to **new directories** with unique names. Never pass
`--input-dir` pointing at the original `data/admet/` path when writing new `train_qml.csv`
files — instead generate new per-seed data files in a dedicated sub-directory (see below).

---

## Phase 0 — Post-hoc DeLong CIs (L5, no new training, ~1 hour)

**Goal:** Compute 95% DeLong confidence intervals on AUROC for all 22 endpoints using
existing prediction scores. Produces the CI columns needed in Tables 4a/4b/4c.

### 0.1 — Understand the existing RawDataEvaluation.csv schema

Each `results/admet_config/dataset=test.csv/{timestamp}/RawDataEvaluation.csv` contains one
row per compound per model per featuriser. Columns include at minimum:
`Dataset, # Features, # Samples, ...` (dataset-level statistics, NOT per-compound predictions).

**Check first:** The QSVC predictions are stored as `y_predicted` (hard labels from
`qsvc.predict(X_test)`) in `ModelResults.csv`, but AUROC requires continuous scores.

Run this check before any other step:

```bash
# From repo root
.venv/bin/python3 - <<'EOF'
import glob, pandas as pd

# Look at one ModelResults.csv to see if probability scores are stored
f = next(iter(glob.glob(
    "results/admet_qsvc_config/dataset=train_qml.csv/**/ModelResults.csv",
    recursive=True
)))
df = pd.read_csv(f)
print(df.columns.tolist())
print(df.head(2).to_string())
EOF
```

**Two possible outcomes:**

- **A) Probability scores are stored** (column like `y_prob`, `y_score`, `prob_pos`):
  DeLong CI can be computed directly. Proceed to Step 0.2.

- **B) Only hard labels are stored** (AUROC was computed internally using
  `predict_proba` but scores weren't saved): Bootstrap CI using per-endpoint
  `y_test` + `y_predicted` columns if available, or rerun inference only
  (not training) using saved checkpoints from `results/admet_benchmark/checkpoints/`.

### 0.2 — Write script: `experiments/admet_benchmark/07_delong_ci.py`

Create this script. It should:

1. Load all `ModelResults.csv` files from `results/admet_config/dataset=test.csv/` and
   `results/admet_qsvc_config/dataset=train_qml.csv/` (the QSVC results).
2. For each `(endpoint, featuriser, model)` triple, compute the DeLong 95% CI on AUROC
   using the `scipy` implementation or a hand-rolled DeLong estimator:

```python
# DeLong AUROC CI — place inside the script
import numpy as np
from scipy import stats

def delong_ci(y_true, y_score, alpha=0.05):
    """Compute DeLong 95% CI for AUROC.
    Reference: DeLong et al. (1988) Biometrics.
    """
    n1 = int(y_true.sum())        # positives
    n0 = len(y_true) - n1         # negatives
    if n1 == 0 or n0 == 0:
        return float('nan'), float('nan')
    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]
    # Placement values
    V10 = np.array([np.mean(ps > neg_scores) + 0.5 * np.mean(ps == neg_scores)
                    for ps in pos_scores])
    V01 = np.array([np.mean(ns < pos_scores) + 0.5 * np.mean(ns == pos_scores)
                    for ns in neg_scores])
    auc = V10.mean()
    s10 = np.var(V10, ddof=1) / n1
    s01 = np.var(V01, ddof=1) / n0
    se = np.sqrt(s10 + s01)
    z = stats.norm.ppf(1 - alpha / 2)
    return max(0, auc - z * se), min(1, auc + z * se)
```

3. **If only hard labels exist** (outcome B from 0.1): use bootstrap CI instead:

```python
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score

def bootstrap_auroc_ci(y_true, y_pred_hard, n_boot=1000, alpha=0.05, seed=42):
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n_boot):
        idx = resample(range(len(y_true)), random_state=rng.randint(0, 99999))
        yt, yp = np.array(y_true)[idx], np.array(y_pred_hard)[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))
    lo = np.percentile(aucs, 100 * alpha / 2)
    hi = np.percentile(aucs, 100 * (1 - alpha / 2))
    return lo, hi
```

4. Output: `results/admet_benchmark/tables/auroc_ci_delong.csv` with columns:
   `endpoint, featuriser, model, auroc, ci_lo, ci_hi, n_test, method`

5. Also output a focused table for QML-win endpoints:
   `results/admet_benchmark/tables/auroc_ci_qml_wins.csv`
   (only rows where `endpoint ∈ {CYP2C9_Substrate_CarbonMangels, Clearance_Hepatocyte_AZ, CYP2D6_Substrate_CarbonMangels}`)

**Run:**
```bash
cd /proj/bmfm/users/bckwon/projects/QBioCode
.venv/bin/python3 experiments/admet_benchmark/07_delong_ci.py
```

**Expected runtime:** < 5 minutes. No GPU required.

---

## Phase 1 — Multi-seed Subsample Sweep (L1 + L5, ~450 GPU-hours)

**Goal:** Train all models (QSVC + classical@300) on 4 additional subsample seeds
(`seed ∈ {0, 21, 84, 100}`; seed=42 already exists), always evaluated on the fixed TDC
`test.csv`. Produces mean ± std AUROC per endpoint across 5 draws.

### Why 4 seeds and not more?

Each QSVC run costs ~90 min/endpoint × 22 endpoints × 3 feats = ~99 GPU-hours per seed.
4 additional seeds = ~396 GPU-hours. Classical@300 is fast (~5 min/endpoint), adding ~14
GPU-hours. Total: ~410 GPU-hours, feasible across a week of cluster time.

### 1.1 — Generate new `train_qml_{seed}.csv` files for each endpoint/featuriser

**Do NOT overwrite `data/admet/{endpoint}/{feat}/train_qml.csv` (the seed=42 draw).**

Instead, create a dedicated directory per seed:
`data/admet_seeds/seed_{S}/{endpoint}/{feat}/` containing:
- `train_qml.csv` — new stratified draw of 300 samples with seed S
- `test.csv` — **symlink** to `../../data/admet/{endpoint}/{feat}/test.csv` (same fixed test)
- `valid.csv` — symlink to original valid.csv

Write script `experiments/admet_benchmark/08_generate_seed_splits.py`:

```python
#!/usr/bin/env python3
"""
Generate per-seed training subsamples for L1 multi-seed ablation.

For each seed in SEEDS (excluding 42 which already exists as train_qml.csv),
creates data/admet_seeds/seed_{S}/{endpoint}/{feat}/train_qml.csv
by calling _cap_qml_split with seed=S.

Test and valid splits are SYMLINKED from the original data/admet/ directory
(never copied — both point to the same TDC canonical files).

Usage:
    .venv/bin/python3 experiments/admet_benchmark/08_generate_seed_splits.py
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

SEEDS = [0, 21, 84, 100]   # seed=42 already in data/admet/{ep}/{feat}/train_qml.csv
QML_CAP = 300
ENDPOINTS = [
    "AMES", "BBB_Martins", "Bioavailability_Ma", "CYP1A2_Veith",
    "CYP2C19_Veith", "CYP2C9_Substrate_CarbonMangels", "CYP2C9_Veith",
    "CYP2D6_Substrate_CarbonMangels", "CYP2D6_Veith",
    "CYP3A4_Substrate_CarbonMangels", "CYP3A4_Veith", "Caco2_Wang",
    "Clearance_Hepatocyte_AZ", "DILI", "HIA_Hou", "Half_Life_Obach",
    "Lipophilicity_AstraZeneca", "PPBR_AstraZeneca", "Pgp_Broccatelli",
    "Solubility_AqSolDB", "VDss_Lombardo", "hERG",
]
FEATS = ["ecfp4", "maccs", "rdkit200"]


def cap_qml_split(train_df: pd.DataFrame, seed: int, cap: int = QML_CAP) -> pd.DataFrame:
    """Stratified subsample — mirrors tdc_admet_loader._cap_qml_split."""
    if len(train_df) <= cap:
        return train_df.copy()
    classes = train_df["Y"].unique()
    n_per_class = cap // len(classes)
    rng = np.random.RandomState(seed)
    parts = []
    for cls in sorted(classes):
        cls_df = train_df[train_df["Y"] == cls]
        n = min(n_per_class, len(cls_df))
        parts.append(cls_df.sample(n=n, random_state=rng.randint(0, 10000)))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def main():
    src_base = REPO / "data" / "admet"
    dst_base = REPO / "data" / "admet_seeds"

    total = 0
    for seed in SEEDS:
        print(f"\n=== seed={seed} ===")
        for ep in ENDPOINTS:
            for feat in FEATS:
                src_dir = src_base / ep / feat
                train_path = src_dir / "train.csv"
                if not train_path.exists():
                    print(f"  SKIP {ep}/{feat}: no train.csv")
                    continue

                dst_dir = dst_base / f"seed_{seed}" / ep / feat
                dst_dir.mkdir(parents=True, exist_ok=True)

                # 1. Generate new train_qml.csv
                train_df = pd.read_csv(train_path)
                qml_df = cap_qml_split(train_df, seed)
                qml_df.to_csv(dst_dir / "train_qml.csv", index=False)

                # 2. Symlink test.csv and valid.csv from original (never copy)
                for fname in ["test.csv", "valid.csv"]:
                    link = dst_dir / fname
                    target = os.path.relpath(src_dir / fname, dst_dir)
                    if link.exists() or link.is_symlink():
                        link.unlink()
                    link.symlink_to(target)

                total += 1
                print(f"  OK  {ep}/{feat}  n_qml={len(qml_df)}")

    print(f"\nGenerated {total} train_qml.csv files across {len(SEEDS)} seeds.")


if __name__ == "__main__":
    main()
```

**Run:**
```bash
cd /proj/bmfm/users/bckwon/projects/QBioCode
.venv/bin/python3 experiments/admet_benchmark/08_generate_seed_splits.py
```

**Expected runtime:** < 2 minutes. No GPU needed.

**Verify:**
```bash
wc -l data/admet_seeds/seed_0/CYP2C9_Substrate_CarbonMangels/ecfp4/train_qml.csv
# Should be 242 (241 rows + header), same as seed=42
ls -la data/admet_seeds/seed_0/CYP2C9_Substrate_CarbonMangels/ecfp4/test.csv
# Should show: test.csv -> ../../../admet/CYP2C9_Substrate_CarbonMangels/ecfp4/test.csv
```

### 1.2 — Create QSVC config for seed ablation

Create `qbiocode/apps/qprofiler/configs/admet_qsvc_seed_ablation.yaml`.
**Diff from `admet_qsvc_config.yaml`:** only two lines change —
`seed: {S}` and `q_seed: {S}`. Everything else is identical.

Write script `experiments/admet_benchmark/09_generate_seed_configs.py`:

```python
#!/usr/bin/env python3
"""
Generate per-seed YAML configs for the L1 seed ablation.
Creates configs/admet_qsvc_seed_{S}.yaml for each seed S.
"""
import os, yaml
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SEEDS = [0, 21, 84, 100]
BASE_CONFIG = REPO / "qbiocode/apps/qprofiler/configs/admet_qsvc_config.yaml"
OUT_DIR = REPO / "qbiocode/apps/qprofiler/configs"

with open(BASE_CONFIG) as f:
    base = yaml.safe_load(f)

for seed in SEEDS:
    cfg = dict(base)
    cfg["seed"] = seed
    cfg["q_seed"] = seed
    cfg["config_file_name"] = f"admet_qsvc_seed_{seed}"
    out = OUT_DIR / f"admet_qsvc_seed_{seed}.yaml"
    with open(out, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"Wrote {out}")

# Also generate classical@300 seed configs (for cl@300/pca ablation)
BASE_CL = REPO / "qbiocode/apps/qprofiler/configs/admet_classical300_config.yaml"
with open(BASE_CL) as f:
    base_cl = yaml.safe_load(f)

for seed in SEEDS:
    cfg = dict(base_cl)
    cfg["seed"] = seed
    cfg["q_seed"] = seed
    cfg["config_file_name"] = f"admet_classical300_seed_{seed}"
    out = OUT_DIR / f"admet_classical300_seed_{seed}.yaml"
    with open(out, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"Wrote {out}")
```

**Run:**
```bash
.venv/bin/python3 experiments/admet_benchmark/09_generate_seed_configs.py
```

### 1.3 — Create LSF job array for QSVC seed ablation

Create `experiments/admet_benchmark/lsf/job_L1_qsvc_seeds.sh`:

```bash
#!/usr/bin/env bash
#==============================================================================
# LSF Job Array: L1 Multi-seed QSVC ablation
# 4 seeds × 22 endpoints × 3 featurisers = 264 tasks
# Each task: QSVC on train_qml.csv from data/admet_seeds/seed_{S}/{ep}/{feat}/
# Always evaluates on fixed test.csv (symlinked to original TDC test)
#==============================================================================
#BSUB -J admet_qsvc_seeds[1-264]
#BSUB -q normal
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4000]"
#BSUB -m "zu-a100-b05-02 zu-a100-c06-01 zu-a100-c08-02 zu-a100-c08-03"
#BSUB -gpu "num=1:mode=shared:j_exclusive=no"
#BSUB -o logs/admet/l1_qsvc_seeds_%I_%J.out
#BSUB -e logs/admet/l1_qsvc_seeds_%I_%J.err
#BSUB -cwd /proj/bmfm/users/bckwon/projects/QBioCode
#BSUB -r
#BSUB -nr 2

set -euo pipefail
REPO_ROOT="/proj/bmfm/users/bckwon/projects/QBioCode"
cd "${REPO_ROOT}"
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}"

# Task list: 4 seeds × 22 endpoints × 3 feats = 264 entries (index 1-based)
# Generated by: python3 experiments/admet_benchmark/10_generate_task_list.py
source experiments/admet_benchmark/lsf/l1_task_list.sh   # defines TASKS array
# (see Step 1.4 for how l1_task_list.sh is generated)

TASK="${TASKS[${LSB_JOBINDEX}]}"
SEED="${TASK%%:::*}"
REST="${TASK#*:::}"
ENDPOINT="${REST%%:::*}"
FEAT="${REST##*:::}"

INPUT_DIR="${REPO_ROOT}/data/admet_seeds/seed_${SEED}/${ENDPOINT}/${FEAT}"
CONFIG="${REPO_ROOT}/qbiocode/apps/qprofiler/configs/admet_qsvc_seed_${SEED}.yaml"
DATA_TYPE="${ENDPOINT}_${FEAT}_seed${SEED}"

echo "Job ${LSB_JOBID:-local}[${LSB_JOBINDEX}] | $(hostname) | $(date)"
echo "Seed=${SEED}  Endpoint=${ENDPOINT}  Feat=${FEAT}"
echo "Input: ${INPUT_DIR}"

PYTHONPATH="${REPO_ROOT}" qprofiler-batch \
  --input-dir "${INPUT_DIR}" \
  --config "${CONFIG}" \
  --data-type "${DATA_TYPE}" \
  --n-jobs 1
```

### 1.4 — Generate the task list file

Create `experiments/admet_benchmark/10_generate_task_list.py`:

```python
#!/usr/bin/env python3
"""
Generates experiments/admet_benchmark/lsf/l1_task_list.sh
A bash array of 264 tasks for the L1 seed ablation.
Format per entry: "SEED:::ENDPOINT:::FEAT"
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SEEDS = [0, 21, 84, 100]
ENDPOINTS = [
    "AMES", "BBB_Martins", "Bioavailability_Ma", "CYP1A2_Veith",
    "CYP2C19_Veith", "CYP2C9_Substrate_CarbonMangels", "CYP2C9_Veith",
    "CYP2D6_Substrate_CarbonMangels", "CYP2D6_Veith",
    "CYP3A4_Substrate_CarbonMangels", "CYP3A4_Veith", "Caco2_Wang",
    "Clearance_Hepatocyte_AZ", "DILI", "HIA_Hou", "Half_Life_Obach",
    "Lipophilicity_AstraZeneca", "PPBR_AstraZeneca", "Pgp_Broccatelli",
    "Solubility_AqSolDB", "VDss_Lombardo", "hERG",
]
FEATS = ["ecfp4", "maccs", "rdkit200"]

lines = ["# Auto-generated by 10_generate_task_list.py", "TASKS=(", '    ""  # 0 unused (LSF is 1-based)']
idx = 1
for seed in SEEDS:
    for ep in ENDPOINTS:
        for feat in FEATS:
            src = REPO / "data" / "admet_seeds" / f"seed_{seed}" / ep / feat / "train_qml.csv"
            if src.exists():
                lines.append(f'    "{seed}:::{ep}:::{feat}"  # {idx}')
                idx += 1
lines.append(")")
out = REPO / "experiments/admet_benchmark/lsf/l1_task_list.sh"
out.write_text("\n".join(lines) + "\n")
print(f"Wrote {out}  ({idx-1} tasks)")
```

**Run after Step 1.1:**
```bash
.venv/bin/python3 experiments/admet_benchmark/10_generate_task_list.py
# Verify task count:
grep -c ':::' experiments/admet_benchmark/lsf/l1_task_list.sh
```

### 1.5 — Also run classical@300 for the new seeds

Classical training is fast (~5 min/endpoint) so run all 264 classical tasks in one array too.

Create `experiments/admet_benchmark/lsf/job_L1_classical300_seeds.sh` — identical structure
to `job_L1_qsvc_seeds.sh` but:
- Use `admet_classical300_seed_{S}.yaml` instead of `admet_qsvc_seed_{S}.yaml`
- `n 6` (more parallelism) instead of `n 1`
- `rusage[mem=8000]`
- No GPU required: replace `#BSUB -gpu ...` with a comment

### 1.6 — Submit jobs

```bash
# From repo root, after Steps 1.1–1.5:
bsub < experiments/admet_benchmark/lsf/job_L1_qsvc_seeds.sh
bsub < experiments/admet_benchmark/lsf/job_L1_classical300_seeds.sh
```

Monitor:
```bash
bjobs -noheader | grep "admet_qsvc_seeds\|admet_cl300_seeds" | awk '{print $3}' | sort | uniq -c
```

### 1.7 — Aggregate results across seeds

Write `experiments/admet_benchmark/11_aggregate_seed_results.py`:

```python
#!/usr/bin/env python3
"""
Aggregates ModelResults.csv files from all seed runs and the baseline seed=42.
Produces:
  results/admet_benchmark/tables/seed_ablation_raw.csv   — all rows, all seeds
  results/admet_benchmark/tables/seed_ablation_summary.csv — mean/std/min/max per
      (endpoint, featuriser, model, embedding) across 5 seeds
"""
import glob, re
import pandas as pd
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
rows = []

# Seed=42 baseline: results/admet_qsvc_config/dataset=train_qml.csv/**/ModelResults.csv
for f in glob.glob(str(REPO / "results/admet_qsvc_config/dataset=train_qml.csv/**/ModelResults.csv"),
                   recursive=True):
    df = pd.read_csv(f)
    df["seed"] = 42
    rows.append(df)

# New seeds: results/{ENDPOINT}_{FEAT}_seed{S}_batch_*/ModelResults.csv
for f in glob.glob(str(REPO / "results/*/*seed**/ModelResults.csv"), recursive=True):
    m = re.search(r"seed(\d+)", f)
    if not m:
        continue
    df = pd.read_csv(f)
    df["seed"] = int(m.group(1))
    rows.append(df)

all_df = pd.concat(rows, ignore_index=True)
all_df.to_csv(REPO / "results/admet_benchmark/tables/seed_ablation_raw.csv", index=False)

# Summary: mean ± std across seeds
key_cols = ["endpoint", "featurizer", "model", "embeddings"]
existing = [c for c in key_cols if c in all_df.columns]
summary = (
    all_df.groupby(existing)["auroc"]
    .agg(["mean", "std", "min", "max", "count"])
    .reset_index()
)
summary.columns = existing + ["auroc_mean", "auroc_std", "auroc_min", "auroc_max", "n_seeds"]
summary.to_csv(REPO / "results/admet_benchmark/tables/seed_ablation_summary.csv", index=False)
print(f"Done. {len(all_df)} rows total, {summary['n_seeds'].value_counts().to_dict()}")
```

---

## Phase 2 — VQC/QNN Initialisation Seed Sweep (L2, ~90 GPU-hours)

**Goal:** For the 3 QML-win endpoints (CYP2C9\_Substrate, Clearance\_Hepatocyte\_AZ,
CYP2D6\_Substrate) plus all 3 featurisers, run VQC with 9 additional `q_seed` values.
The training DATA stays fixed at the existing `data/admet/{ep}/{feat}/train_qml.csv` (seed=42).
Only the quantum circuit initialisation seed changes.

### 2.1 — Why only 3 endpoints?

VQC on a statevector simulator takes ~30–60 min/endpoint. 9 seeds × 3 endpoints × 3 feats
= 81 runs = ~50–80 GPU-hours. This is tractable. The question is specifically whether the
VQC wins on CYP2C9\_Substrate are seed-robust, so scoping to win/near-win endpoints is correct.

### 2.2 — Generate VQC q_seed configs

Extend `09_generate_seed_configs.py` or write `09b_generate_qseed_configs.py`:

```python
#!/usr/bin/env python3
"""
Generates admet_vqc_qseed_{S}.yaml for q_seed sweep (L2).
Data seed (train subsample) stays at 42 — only q_seed changes.
"""
import yaml
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
Q_SEEDS = [0, 7, 21, 73, 84, 100, 123, 200, 314]   # 9 additional; q_seed=42 already run
BASE = REPO / "qbiocode/apps/qprofiler/configs/admet_config.yaml"
OUT = REPO / "qbiocode/apps/qprofiler/configs"

with open(BASE) as f:
    base = yaml.safe_load(f)

for qs in Q_SEEDS:
    cfg = dict(base)
    cfg["seed"] = 42            # data seed stays fixed
    cfg["q_seed"] = qs
    cfg["model"] = ["vqc"]      # VQC only — QNN optional, add if budget allows
    cfg["config_file_name"] = f"admet_vqc_qseed_{qs}"
    out = OUT / f"admet_vqc_qseed_{qs}.yaml"
    with open(out, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"Wrote {out}")
```

### 2.3 — Create LSF job array for VQC q_seed sweep

Priority endpoints: `CYP2C9_Substrate_CarbonMangels`, `Clearance_Hepatocyte_AZ`,
`CYP2D6_Substrate_CarbonMangels` × 3 feats × 9 q_seeds = 81 tasks.

Create `experiments/admet_benchmark/lsf/job_L2_vqc_qseeds.sh`:

```bash
#!/usr/bin/env bash
#BSUB -J admet_vqc_qseeds[1-81]
#BSUB -q normal
#BSUB -n 6
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16000]"
#BSUB -m "zu-a100-b05-02 zu-a100-c06-01 zu-a100-c08-02"
#BSUB -gpu "num=1:mode=shared:j_exclusive=no"
#BSUB -o logs/admet/l2_vqc_qseeds_%I_%J.out
#BSUB -e logs/admet/l2_vqc_qseeds_%I_%J.err
#BSUB -cwd /proj/bmfm/users/bckwon/projects/QBioCode
#BSUB -r
#BSUB -nr 2

# Task list generated by 10_generate_task_list.py (L2 variant)
# Format: "QSEED:::ENDPOINT:::FEAT"
source experiments/admet_benchmark/lsf/l2_task_list.sh

TASK="${TASKS[${LSB_JOBINDEX}]}"
QSEED="${TASK%%:::*}"; REST="${TASK#*:::}"; ENDPOINT="${REST%%:::*}"; FEAT="${REST##*:::}"
INPUT_DIR="${REPO_ROOT}/data/admet/${ENDPOINT}/${FEAT}"
CONFIG="${REPO_ROOT}/qbiocode/apps/qprofiler/configs/admet_vqc_qseed_${QSEED}.yaml"

qprofiler-batch --input-dir "${INPUT_DIR}" --config "${CONFIG}" \
  --data-type "${ENDPOINT}_${FEAT}_vqc_qseed${QSEED}" --n-jobs 1
```

Generate task list (add to `10_generate_task_list.py` or write `10b`):

```python
PRIORITY_EPS = [
    "CYP2C9_Substrate_CarbonMangels",
    "Clearance_Hepatocyte_AZ",
    "CYP2D6_Substrate_CarbonMangels",
]
Q_SEEDS = [0, 7, 21, 73, 84, 100, 123, 200, 314]
lines = ["TASKS=(", '    ""']
idx = 1
for qs in Q_SEEDS:
    for ep in PRIORITY_EPS:
        for feat in ["ecfp4", "maccs", "rdkit200"]:
            lines.append(f'    "{qs}:::{ep}:::{feat}"  # {idx}')
            idx += 1
lines.append(")")
Path("experiments/admet_benchmark/lsf/l2_task_list.sh").write_text("\n".join(lines))
```

### 2.4 — Aggregate L2 results

Write `experiments/admet_benchmark/12_aggregate_qseed_results.py` — same structure as
`11_aggregate_seed_results.py` but group by `q_seed` and focus on VQC rows. Output:
`results/admet_benchmark/tables/vqc_qseed_summary.csv` with columns:
`endpoint, featuriser, q_seed, auroc, auprc, mcc, f1`

---

## Phase 3 — PCA Dimension Sweep (L3, ~270 GPU-hours)

**Goal:** Sweep `n_components ∈ [4, 8, 12, 16, 32]` for QSVC on 6 priority endpoints
× 3 featurisers. The dimension=8 result already exists in
`results/admet_qsvc_config/`. Only dims 4, 12, 16, 32 need to be run.
Classical@300 counterpart (`cl@N/pca`) also needed for each dim to compute the delta.

**Critical optimisation:** The PCA model is fit inside `get_embeddings()` **at runtime**
using `X_train_emb = PCA(n_components=K).fit_transform(X_train)`. There is no cached PCA
artifact on disk. Therefore each dim requires a fresh training run — but the full training
data (`train.csv`) is already on disk. **Do not regenerate any CSVs.**

### 3.1 — Generate PCA-dim configs

Create `experiments/admet_benchmark/13_generate_pca_configs.py`:

```python
#!/usr/bin/env python3
"""
Generates admet_qsvc_pca_{K}.yaml and admet_cl300_pca_{K}.yaml
for n_components in [4, 12, 16, 32]. K=8 already exists.
"""
import yaml
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DIMS = [4, 12, 16, 32]   # K=8 already run
CFG_DIR = REPO / "qbiocode/apps/qprofiler/configs"

# QSVC configs
with open(CFG_DIR / "admet_qsvc_config.yaml") as f:
    base_qsvc = yaml.safe_load(f)

for K in DIMS:
    cfg = dict(base_qsvc)
    cfg["n_components"] = K
    cfg["config_file_name"] = f"admet_qsvc_pca_{K}"
    with open(CFG_DIR / f"admet_qsvc_pca_{K}.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

# Classical@300 configs (for matched cl@300/pca-K comparison)
with open(CFG_DIR / "admet_classical300_config.yaml") as f:
    base_cl = yaml.safe_load(f)

for K in DIMS:
    cfg = dict(base_cl)
    cfg["n_components"] = K
    cfg["config_file_name"] = f"admet_cl300_pca_{K}"
    with open(CFG_DIR / f"admet_cl300_pca_{K}.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

print("Done.")
```

### 3.2 — Priority endpoints for PCA sweep

Run the full 22-endpoint suite only for K=4 (cheapest, ~99 GPU-hours for QSVC).
For K=12, 16, 32, restrict to 6 priority endpoints to limit cost:

**Priority endpoints:**
- CYP2C9\_Substrate\_CarbonMangels (QML win)
- Clearance\_Hepatocyte\_AZ (QML win)
- CYP2D6\_Substrate\_CarbonMangels (parity)
- Bioavailability\_Ma (near-parity)
- BBB\_Martins (near-parity)
- PPBR\_AstraZeneca (near-parity AUPRC)

### 3.3 — Create LSF job arrays for PCA sweep

Create `experiments/admet_benchmark/lsf/job_L3_pca_sweep.sh`:

```bash
#!/usr/bin/env bash
# K=4: 22 endpoints × 3 feats = 66 tasks (QSVC) + 66 (classical@300) = 132
# K=12,16,32: 6 priority endpoints × 3 feats × 3 dims = 54 (QSVC) + 54 (cl@300) = 108
# Total: 240 tasks across two arrays
```

Task list generation: extend `10_generate_task_list.py` with L3 variants:
- `l3a_task_list.sh`: K=4, all 22 endpoints (QSVC)
- `l3b_task_list.sh`: K=12,16,32, 6 priority endpoints (QSVC)
- Separate classical@300 arrays for each K

Format: `"K:::ENDPOINT:::FEAT"` — job reads the correct `admet_qsvc_pca_{K}.yaml`.

### 3.4 — Aggregate PCA sweep results

Write `experiments/admet_benchmark/14_aggregate_pca_results.py`:

Collect all `ModelResults.csv` from K=4,8,12,16,32 runs. Add a `n_components` column.
Output: `results/admet_benchmark/tables/pca_sweep_raw.csv` and
`results/admet_benchmark/tables/pca_sweep_summary.csv` with columns:
`endpoint, featuriser, model, n_components, auroc_mean, auroc_std`.

The K=8 rows come from the **already-existing** `results/admet_qsvc_config/` — include them
by filtering for `embeddings == 'pca'` and `n_components == 8`.

---

## Phase 4 — Final Analysis and Table Generation (all limitations)

Write `experiments/admet_benchmark/15_final_robustness_analysis.py`:

This script aggregates all Phase 0–3 outputs into the paper-ready tables.

### 4.1 — Table R1: Seed stability (L1)

From `seed_ablation_summary.csv`, for each endpoint/model/featuriser, report:
`auroc_mean ± auroc_std` across 5 seeds. Flag endpoints where `std > 0.05` as "unstable".

### 4.2 — Table R2: VQC init stability (L2)

From `vqc_qseed_summary.csv`, compute mean ± std AUROC across 10 q_seeds for VQC on the
3 QML-win endpoints. If `mean - std > best_classical_auroc`, the win is init-robust.

### 4.3 — Table R3: PCA dimension curve (L3)

From `pca_sweep_raw.csv`, produce AUROC vs. n\_components table for QSVC and
best-classical-at-K for each priority endpoint. Identify the crossover dimension.

### 4.4 — Table R4: DeLong CIs (L5)

From `auroc_ci_delong.csv`, add CI columns to Tables 4a/4b/4c in `paper/summary.md`.

### 4.5 — Update paper/summary.md

After all analysis scripts run, update the Limitations section with actual numbers:
- Replace "directional signal" hedges with confirmed/refuted win claims based on L1/L2 CI results
- Add Table R3 inline in L3 response
- Add CI columns to Tables 4a/4b/4c for L5 response

---

## Full Execution Order (Optimal Sequence)

```
┌─────────────────────────────────────────────────────────────────┐
│ IMMEDIATE (no GPU, < 30 min total)                              │
│                                                                 │
│  Phase 0:   python3 07_delong_ci.py                            │
│  Phase 1.1: python3 08_generate_seed_splits.py                 │
│  Phase 1.2: python3 09_generate_seed_configs.py                │
│  Phase 1.4: python3 10_generate_task_list.py                   │
│  Phase 2.2: python3 09b_generate_qseed_configs.py              │
│  Phase 3.1: python3 13_generate_pca_configs.py                 │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│ SUBMIT (can all start in parallel — independent)                │
│                                                                 │
│  bsub < lsf/job_L1_qsvc_seeds.sh         (~396 GPU-hrs)        │
│  bsub < lsf/job_L1_classical300_seeds.sh (~14 GPU-hrs)         │
│  bsub < lsf/job_L2_vqc_qseeds.sh         (~80 GPU-hrs)         │
│  bsub < lsf/job_L3_pca_sweep_k4.sh       (~99 GPU-hrs QSVC)    │
│  bsub < lsf/job_L3_pca_sweep_kN.sh       (~54 GPU-hrs QSVC)    │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼  (after all jobs finish)
┌─────────────────────────────────────────────────────────────────┐
│ AGGREGATE                                                       │
│                                                                 │
│  python3 11_aggregate_seed_results.py       (L1 → Table R1)    │
│  python3 12_aggregate_qseed_results.py      (L2 → Table R2)    │
│  python3 14_aggregate_pca_results.py        (L3 → Table R3)    │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│ FINAL ANALYSIS                                                  │
│                                                                 │
│  python3 15_final_robustness_analysis.py    (all → paper tables)│
│  → Update paper/summary.md Limitations section                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Compute Budget Summary

| Phase | Description | New GPU-hours | Wall-clock (est.) |
|---|---|---|---|
| 0 | DeLong CIs (post-hoc) | ~0 | 30 min |
| 1 | L1: 4 seeds × 22ep × 3feat (QSVC + cl@300) | ~410 | 5–7 days |
| 2 | L2: 9 q_seeds × 3ep × 3feat (VQC only) | ~80 | 2–3 days |
| 3 | L3: 4 PCA dims × 6–22ep × 3feat (QSVC + cl@300) | ~270 | 4–5 days |
| **Total** | | **~760 GPU-hours** | **~7 days parallel** |

Phases 1, 2, 3 can all run **concurrently** on the cluster. With 6–8 nodes available in
parallel, wall-clock time is approximately 7 days. Phase 0 can be done today.

---

## File Manifest (new files to create)

| Path | Purpose |
|---|---|
| `experiments/admet_benchmark/07_delong_ci.py` | Phase 0: Post-hoc CIs |
| `experiments/admet_benchmark/08_generate_seed_splits.py` | Phase 1: Generate per-seed train_qml.csv |
| `experiments/admet_benchmark/09_generate_seed_configs.py` | Phase 1: Generate per-seed YAML configs |
| `experiments/admet_benchmark/09b_generate_qseed_configs.py` | Phase 2: Generate per-q_seed YAML configs |
| `experiments/admet_benchmark/10_generate_task_list.py` | Phase 1+2: Generate LSF task arrays |
| `experiments/admet_benchmark/11_aggregate_seed_results.py` | Phase 1: Aggregate seed ablation |
| `experiments/admet_benchmark/12_aggregate_qseed_results.py` | Phase 2: Aggregate VQC q_seed ablation |
| `experiments/admet_benchmark/13_generate_pca_configs.py` | Phase 3: Generate per-dim YAML configs |
| `experiments/admet_benchmark/14_aggregate_pca_results.py` | Phase 3: Aggregate PCA dim sweep |
| `experiments/admet_benchmark/15_final_robustness_analysis.py` | Final: Merge all, update paper tables |
| `experiments/admet_benchmark/lsf/job_L1_qsvc_seeds.sh` | LSF: L1 QSVC seed array |
| `experiments/admet_benchmark/lsf/job_L1_classical300_seeds.sh` | LSF: L1 classical@300 seed array |
| `experiments/admet_benchmark/lsf/job_L2_vqc_qseeds.sh` | LSF: L2 VQC q_seed array |
| `experiments/admet_benchmark/lsf/job_L3_pca_sweep_k4.sh` | LSF: L3 PCA sweep K=4 (all endpoints) |
| `experiments/admet_benchmark/lsf/job_L3_pca_sweep_kN.sh` | LSF: L3 PCA sweep K=12,16,32 (priority only) |
| `experiments/admet_benchmark/lsf/l1_task_list.sh` | Auto-generated LSF task array (L1) |
| `experiments/admet_benchmark/lsf/l2_task_list.sh` | Auto-generated LSF task array (L2) |
| `data/admet_seeds/seed_{0,21,84,100}/{ep}/{feat}/train_qml.csv` | New per-seed training subsamples |
| `data/admet_seeds/seed_{0,21,84,100}/{ep}/{feat}/test.csv` | Symlinks → original TDC test.csv |
| `qbiocode/apps/qprofiler/configs/admet_qsvc_seed_{0,21,84,100}.yaml` | Per-seed QSVC configs |
| `qbiocode/apps/qprofiler/configs/admet_classical300_seed_{0,21,84,100}.yaml` | Per-seed cl@300 configs |
| `qbiocode/apps/qprofiler/configs/admet_vqc_qseed_{0,7,21,73,84,100,123,200,314}.yaml` | Per-q_seed VQC configs |
| `qbiocode/apps/qprofiler/configs/admet_qsvc_pca_{4,12,16,32}.yaml` | Per-dim QSVC PCA configs |
| `qbiocode/apps/qprofiler/configs/admet_cl300_pca_{4,12,16,32}.yaml` | Per-dim cl@300 PCA configs |
| `results/admet_benchmark/tables/auroc_ci_delong.csv` | Phase 0 output |
| `results/admet_benchmark/tables/seed_ablation_raw.csv` | Phase 1 output |
| `results/admet_benchmark/tables/seed_ablation_summary.csv` | Phase 1 output |
| `results/admet_benchmark/tables/vqc_qseed_summary.csv` | Phase 2 output |
| `results/admet_benchmark/tables/pca_sweep_raw.csv` | Phase 3 output |
| `results/admet_benchmark/tables/pca_sweep_summary.csv` | Phase 3 output |

---

## Key Invariants the Coding Agent Must Preserve

1. **Never overwrite** `data/admet/{ep}/{feat}/train_qml.csv` — this is the seed=42 draw.
2. **Never overwrite** `data/admet/{ep}/{feat}/test.csv` — this is the canonical TDC test.
3. **New training data goes in `data/admet_seeds/`**, not `data/admet/`.
4. **New results go in new directories** under `results/` with descriptive names; never
   write into existing `results/admet_config/`, `results/admet_qsvc_config/` directories.
5. **PCA is not cached** — it is refit inside `get_embeddings()` each run from
   `X_train`. No pre-caching step is needed or possible with the current pipeline.
6. **`qprofiler-batch` reads the file it's pointed at** — the `--input-dir` argument
   tells it where to find `train_qml.csv`. For new seeds, point it at
   `data/admet_seeds/seed_{S}/{ep}/{feat}/`. For PCA dim sweep, point it at the
   original `data/admet/{ep}/{feat}/` (same data, different `n_components` in config).
7. **All `test.csv` evaluations are on the same fixed TDC data** — confirmed because
   `test.csv` in `data/admet_seeds/` are symlinks to the originals.
