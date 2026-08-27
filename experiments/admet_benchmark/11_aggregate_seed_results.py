#!/usr/bin/env python3
"""
Aggregate L1 seed ablation results.

Collects ModelResults.csv from:
  - Seed=42 baseline: results/admet_qsvc_config/dataset=train_qml.csv/**/ModelResults.csv
  - New seeds (L1):   results/admet_qsvc_seed_*/dataset=*/  and
                      results/admet_classical300_seed_*/dataset=*/

Produces:
  results/admet_benchmark/tables/seed_ablation_raw.csv
    — all rows, all seeds, all models
  results/admet_benchmark/tables/seed_ablation_summary.csv
    — mean ± std AUROC per (endpoint, featuriser, model, embeddings) across seeds

Run AFTER all L1 jobs complete:
    .venv/bin/python3 experiments/admet_benchmark/11_aggregate_seed_results.py
"""
import glob
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

TABLES_DIR = REPO / "results/admet_benchmark/tables"
FEATS = ["ecfp4", "maccs", "rdkit200"]


def split_data_type(dt: str):
    """Split data_type string into (endpoint, featurizer, seed_tag)."""
    for feat in FEATS:
        # Try patterns like {ep}_{feat}_seed{S} or {ep}_{feat}_cl300_seed{S}
        for pattern in [f"_{feat}_cl300_seed", f"_{feat}_seed", f"_{feat}"]:
            if f"_{feat}" in dt:
                base = dt[:dt.index(f"_{feat}")]
                tail = dt[dt.index(f"_{feat}") + len(f"_{feat}"):]
                return base, feat, tail
    return dt, "unknown", ""


def extract_seed_from_path(fpath: str) -> int:
    """Extract seed number from result path or data_type."""
    # Config name pattern: admet_qsvc_seed_42, admet_classical300_seed_0 ...
    m = re.search(r"seed[_]?(\d+)", fpath)
    if m:
        return int(m.group(1))
    return -1


def load_model_results(pattern: str, default_seed: int = -1) -> list[dict]:
    rows = []
    for fpath in glob.glob(pattern, recursive=True):
        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            print(f"  WARN: could not read {fpath}: {e}")
            continue
        if df.empty:
            continue

        run_dir = Path(fpath).parent
        hydra_cfg = run_dir / ".hydra" / "config.yaml"

        data_type = None
        seed = default_seed
        if hydra_cfg.exists():
            with open(hydra_cfg) as hf:
                for line in hf:
                    if line.startswith("data_type:"):
                        data_type = line.split(":", 1)[1].strip()
                    if line.startswith("seed:"):
                        try:
                            seed = int(line.split(":", 1)[1].strip())
                        except ValueError:
                            pass
                    if data_type and seed != default_seed:
                        break

        if data_type is None:
            # Fallback: extract from path
            data_type = Path(fpath).parent.name

        ep, feat, _ = split_data_type(data_type)
        if seed == -1:
            seed = extract_seed_from_path(str(fpath))

        for _, row in df.iterrows():
            rows.append({
                "endpoint": ep,
                "featurizer": feat,
                "embeddings": row.get("embeddings", "unknown"),
                "model": row.get("model", "unknown"),
                "auroc": row.get("auc", float("nan")),
                "auprc": row.get("auprc", float("nan")),
                "mcc": row.get("mcc", float("nan")),
                "f1": row.get("f1_score", float("nan")),
                "accuracy": row.get("accuracy", float("nan")),
                "seed": seed,
                "source_file": str(fpath),
            })
    return rows


def main():
    rows = []

    # 1. Seed=42 QSVC baseline (already merged into admet_qsvc_config)
    print("Loading seed=42 QSVC baseline …")
    pattern42 = str(REPO / "results/admet_qsvc_config/dataset=train_qml.csv/**/ModelResults.csv")
    r = load_model_results(pattern42, default_seed=42)
    print(f"  {len(r)} rows")
    rows.extend(r)

    # 2. New seed runs — QSVC (config_file_name contains admet_qsvc_seed_{S})
    print("Loading L1 QSVC seed runs …")
    for seed in [0, 21, 84, 100]:
        pattern = str(REPO / f"results/admet_qsvc_seed_{seed}*/dataset=*/simulator_**/ModelResults.csv")
        r = load_model_results(pattern, default_seed=seed)
        print(f"  seed={seed}: {len(r)} rows")
        rows.extend(r)

    # 3. New seed runs — classical@300
    print("Loading L1 classical@300 seed runs …")
    for seed in [0, 21, 84, 100]:
        pattern = str(REPO / f"results/admet_classical300_seed_{seed}*/dataset=*/simulator_**/ModelResults.csv")
        r = load_model_results(pattern, default_seed=seed)
        print(f"  seed={seed}: {len(r)} rows")
        rows.extend(r)

    if not rows:
        print("No results found. Run L1 jobs first.")
        return

    all_df = pd.DataFrame(rows)
    out_raw = TABLES_DIR / "seed_ablation_raw.csv"
    all_df.to_csv(out_raw, index=False)
    print(f"\nWrote {len(all_df)} rows → {out_raw}")

    # Summary: mean ± std across seeds per (endpoint, featurizer, model, embeddings)
    key_cols = ["endpoint", "featurizer", "model", "embeddings"]
    summary = (
        all_df.groupby(key_cols)["auroc"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    summary.columns = key_cols + ["auroc_mean", "auroc_std", "auroc_min", "auroc_max", "n_seeds"]
    out_summary = TABLES_DIR / "seed_ablation_summary.csv"
    summary.to_csv(out_summary, index=False)
    print(f"Wrote {len(summary)} summary rows → {out_summary}")
    print(f"\nSeed counts: {summary['n_seeds'].value_counts().to_dict()}")
    print(f"Mean AUROC std across models: {summary['auroc_std'].mean():.4f}")


if __name__ == "__main__":
    main()
