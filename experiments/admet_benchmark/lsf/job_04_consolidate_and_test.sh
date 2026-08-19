#!/usr/bin/env bash
#BSUB -J admet_test_inf
#BSUB -q normal
#BSUB -m "zu-a100-c08-03"
#BSUB -n 8
#BSUB -R "span[hosts=1] rusage[mem=64000]"
#BSUB -M 64000
#BSUB -o logs/admet/test_inference_%J.out
#BSUB -e logs/admet/test_inference_%J.err
#BSUB -cwd /proj/bmfm/users/bckwon/projects/QBioCode

set -euo pipefail

REPO_ROOT="/proj/bmfm/users/bckwon/projects/QBioCode"
CKPT_DIR="${REPO_ROOT}/results/admet_benchmark/checkpoints"
cd "${REPO_ROOT}"
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}"

echo "Job ${LSB_JOBID:-local} | Host: $(hostname) | $(date)"

# ── Step 1: consolidate all best_models.json + .pkl files ─────────────────
# Each qprofiler run writes checkpoints under its own Hydra output dir.
# We collect all model .pkl files and merge all best_models.json indices
# into a single flat directory that script 04 expects.
echo "Consolidating checkpoints..."
mkdir -p "${CKPT_DIR}"

python3 - "${REPO_ROOT}" <<'PYEOF'
import json, os, shutil, glob, sys

repo  = sys.argv[1]
ckpt  = os.path.join(repo, "results/admet_benchmark/checkpoints")
merged_index = {}

for jf in glob.glob(
        f"{repo}/results/admet_config/**/best_models.json", recursive=True):
    try:
        idx = json.load(open(jf))
    except Exception as e:
        print(f"  WARN skip {jf}: {e}", file=sys.stderr)
        continue

    # Each key is a dataset_name like "valid_none_8_1"
    # Model paths inside are relative to the repo root
    src_dir = os.path.dirname(jf)  # …/checkpoints/

    for dataset_key, models in idx.items():
        if dataset_key not in merged_index:
            merged_index[dataset_key] = {}
        for model_name, info in models.items():
            # Only keep the checkpoint with the highest val_f1
            existing = merged_index[dataset_key].get(model_name)
            if existing and existing["val_f1"] >= info["val_f1"]:
                continue
            # Copy the .pkl file to the flat consolidated dir
            src_pkl = os.path.join(repo, info["path"])
            if not os.path.exists(src_pkl):
                # Try relative to the best_models.json location
                src_pkl = os.path.join(src_dir, os.path.basename(info["path"]))
            if not os.path.exists(src_pkl):
                continue
            dst_subdir = os.path.join(ckpt, dataset_key)
            os.makedirs(dst_subdir, exist_ok=True)
            dst_pkl = os.path.join(dst_subdir, os.path.basename(src_pkl))
            shutil.copy2(src_pkl, dst_pkl)
            merged_index[dataset_key][model_name] = {
                "path": os.path.join(
                    "results/admet_benchmark/checkpoints",
                    dataset_key, os.path.basename(src_pkl)),
                "val_f1":  info["val_f1"],
                "split_id": info["split_id"],
            }

out_json = os.path.join(ckpt, "best_models.json")
with open(out_json, "w") as f:
    json.dump(merged_index, f, indent=2)
print(f"Consolidated {len(merged_index)} dataset keys -> {out_json}")
PYEOF

echo "Consolidation done: $(date)"
echo "Keys in consolidated index: $(python3 -c "import json; print(len(json.load(open('${CKPT_DIR}/best_models.json'))))")"

# ── Step 2: run test inference ────────────────────────────────────────────
echo "Running test inference..."
python experiments/admet_benchmark/04_test_inference.py \
    --data-dir       data/admet \
    --checkpoint-dir "${CKPT_DIR}" \
    --output-dir     results/admet_benchmark/test_results \
    --featurizers    ecfp4 maccs rdkit200

echo "Done: $(date) | Exit: $?"
