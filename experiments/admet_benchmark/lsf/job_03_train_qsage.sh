#!/usr/bin/env bash
#BSUB -J admet_qsage
#BSUB -q normal
#BSUB -m "zu-a100-c08-03"
#BSUB -n 4
#BSUB -R "span[hosts=1] rusage[mem=32000]"
#BSUB -M 32000
#BSUB -o logs/admet/qsage_%J.out
#BSUB -e logs/admet/qsage_%J.err
#BSUB -cwd /proj/bmfm/users/bckwon/projects/QBioCode

set -euo pipefail

REPO_ROOT="/proj/bmfm/users/bckwon/projects/QBioCode"
cd "${REPO_ROOT}"
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}"

echo "Job ${LSB_JOBID:-local} | Host: $(hostname) | $(date)"

# QuantumSage.__init__ requires columns: datatype, model_embed_datatype,
# BestParams_GridSearch — which don't exist in the current ModelResults schema.
# We inject them via a pre-processing wrapper before calling script 03.
python3 - "${REPO_ROOT}" <<'PYEOF'
import sys, os, dill, logging, argparse
sys.path.insert(0, sys.argv[1])
import pandas as pd

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

repo = sys.argv[1]
results_dir = os.path.join(repo, "results/admet_config")
output_dir  = os.path.join(repo, "results/admet_benchmark/sage")
os.makedirs(output_dir, exist_ok=True)

# ── 1. Collect all ModelResults.csv ──────────────────────────────────────
import glob
dfs = []
for f in glob.glob(f"{results_dir}/**/ModelResults.csv", recursive=True):
    try:
        dfs.append(pd.read_csv(f))
    except Exception as e:
        log.warning(f"Skip {f}: {e}")

if not dfs:
    log.error("No ModelResults.csv found — aborting.")
    sys.exit(1)

model_df = pd.concat(dfs, ignore_index=True)
log.info(f"Collected {len(dfs)} files → {len(model_df)} rows")

# ── 2. Add synthetic columns that QuantumSage requires ──────────────────
# datatype: the split file name (Dataset column = "test.csv", "valid.csv", etc.)
model_df["datatype"] = model_df.get("Dataset", pd.Series(["unknown"]*len(model_df)))

# model_embed_datatype: compound key combining model + embedding + dataset
model_df["model_embed_datatype"] = (
    model_df.get("model", "").astype(str) + "_" +
    model_df.get("embeddings", "").fillna("none").astype(str) + "_" +
    model_df["datatype"].astype(str)
)

# BestParams_GridSearch: alias of Model_Parameters
model_df["BestParams_GridSearch"] = model_df.get(
    "Model_Parameters", pd.Series(["{}"]*len(model_df)))

# ── 3. Persist compiled files ─────────────────────────────────────────────
compiled_csv = os.path.join(output_dir, "compiled_ModelResults.csv")
model_df.to_csv(compiled_csv, index=False)
log.info(f"Saved compiled ModelResults → {compiled_csv}")

# ── 4. Train QSage ────────────────────────────────────────────────────────
from qbiocode.apps.sage.sage import QuantumSage
import dill

log.info("Initialising QuantumSage…")
sage = QuantumSage(data_input=model_df)
sage.set_seed(42)

log.info(f"Available models  : {sage._available_models}")
log.info(f"Available metrics : {sage._available_metrics}")
log.info("Training QSage (xgboost_optuna, n_iter=200, cv=10)…")
sage.train_sub_sages(test_size=0.2, sage_type="xgboost_optuna", n_iter=200, cv=10)

sage_pkl = os.path.join(output_dir, "trained_sage.pkl")
with open(sage_pkl, "wb") as f:
    dill.dump(sage, f)
log.info(f"Trained QSage saved → {sage_pkl}")

# ── 5. SHAP + summary ─────────────────────────────────────────────────────
# SHAP analysis (best effort — skip if matplotlib/shap unavailable on node)
try:
    import shap, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for metric in sage._available_metrics:
        for model_name in sage._available_models:
            result = sage._results_subsages.get(metric, {}).get(model_name)
            if result is None: continue
            explainer = shap.Explainer(result["model"])
            model_indices = sage._input_data_metadata[
                sage._input_data_metadata["model"] == model_name].index
            X = sage._input_data_features_only.loc[model_indices]
            shap_values = explainer(X)
            shap.summary_plot(shap_values, X, show=False)
            fname = os.path.join(output_dir, f"shap_{model_name}_{metric}.pdf")
            plt.savefig(fname, bbox_inches="tight")
            plt.close()
            log.info(f"  SHAP plot saved: {fname}")
except Exception as e:
    log.warning(f"SHAP analysis skipped: {e}")

log.info(f"All outputs in: {output_dir}")
PYEOF

echo "Done: $(date) | Exit: $?"
