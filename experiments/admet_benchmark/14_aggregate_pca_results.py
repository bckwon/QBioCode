#!/usr/bin/env python3
"""
Aggregate L3 PCA dimension sweep results.

Collects ModelResults.csv from:
  - K=8 baseline (already run): results/admet_qsvc_config/dataset=test.csv/**/ModelResults.csv
    (filter embeddings == 'pca' → these are all K=8 runs)
  - K=4 (all endpoints): results/admet_qsvc_pca_4/dataset=*/...
  - K=12,16,32 (priority endpoints): results/admet_qsvc_pca_{K}/dataset=*/...

Produces:
  results/admet_benchmark/tables/pca_sweep_raw.csv
    — all (endpoint, feat, model, K, embeddings) AUROC rows
  results/admet_benchmark/tables/pca_sweep_summary.csv
    — mean AUROC per (endpoint, featuriser, n_components) across featurisers

Run AFTER all L3 jobs complete:
    .venv/bin/python3 experiments/admet_benchmark/14_aggregate_pca_results.py
"""
import glob
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

TABLES_DIR = REPO / "results/admet_benchmark/tables"
FEATS = ["ecfp4", "maccs", "rdkit200"]
ALL_DIMS = [4, 8, 12, 16, 32]


def split_data_type(dt: str):
    for feat in FEATS:
        if f"_{feat}" in dt:
            base = dt[:dt.index(f"_{feat}")]
            tail = dt[dt.index(f"_{feat}") + len(f"_{feat}"):]
            return base, feat, tail
    return dt, "unknown", ""


def load_results(pattern: str, n_components: int) -> list[dict]:
    rows = []
    for fpath in glob.glob(pattern, recursive=True):
        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            print(f"  WARN: {fpath}: {e}")
            continue
        if df.empty:
            continue

        # For K=8 baseline: filter to pca embedding rows only
        if n_components == 8 and "embeddings" in df.columns:
            df = df[df["embeddings"] == "pca"]

        run_dir = Path(fpath).parent
        hydra_cfg = run_dir / ".hydra" / "config.yaml"
        data_type = None
        if hydra_cfg.exists():
            with open(hydra_cfg) as hf:
                for line in hf:
                    if line.startswith("data_type:"):
                        data_type = line.split(":", 1)[1].strip()
                        break
        if data_type is None:
            data_type = Path(fpath).parent.name

        ep, feat, _ = split_data_type(data_type)
        for _, row in df.iterrows():
            rows.append({
                "endpoint": ep,
                "featurizer": feat,
                "embeddings": row.get("embeddings", "pca"),
                "model": row.get("model", "unknown"),
                "n_components": n_components,
                "auroc": row.get("auc", float("nan")),
                "auprc": row.get("auprc", float("nan")),
                "mcc": row.get("mcc", float("nan")),
                "f1": row.get("f1_score", float("nan")),
                "source_file": str(fpath),
            })
    return rows


def main():
    rows = []

    # K=8 baseline from existing QSVC results
    print("Loading K=8 baseline (pca rows from admet_qsvc_config) …")
    pattern8 = str(REPO / "results/admet_qsvc_config/dataset=test.csv/**/ModelResults.csv")
    r = load_results(pattern8, n_components=8)
    print(f"  {len(r)} rows")
    rows.extend(r)

    # New PCA dims from L3 jobs — Hydra dirs and flat batch dirs
    for K in [4, 12, 16, 32]:
        print(f"Loading K={K} …")
        # Hydra output dirs (simulator_* subdirs)
        pattern_hydra = str(REPO / f"results/admet_qsvc_pca_{K}*/dataset=*/simulator_**/ModelResults.csv")
        r = load_results(pattern_hydra, n_components=K)
        # Flat batch dirs: results/{ep}_{feat}_qsvc_pca{K}_batch_*/ModelResults.csv
        pattern_flat = str(REPO / f"results/*_qsvc_pca{K}_batch_*/ModelResults.csv")
        r += load_results(pattern_flat, n_components=K)
        print(f"  {len(r)} rows")
        rows.extend(r)

    if not rows:
        print("No PCA sweep results found. Run L3 jobs first.")
        return

    all_df = pd.DataFrame(rows)
    out_raw = TABLES_DIR / "pca_sweep_raw.csv"
    all_df.to_csv(out_raw, index=False)
    print(f"\nWrote {len(all_df)} rows → {out_raw}")

    # Summary: mean AUROC per (endpoint, featuriser, model, n_components)
    key_cols = ["endpoint", "featurizer", "model", "n_components"]
    summary = (
        all_df.groupby(key_cols)["auroc"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = key_cols + ["auroc_mean", "auroc_std", "n_runs"]
    out_summary = TABLES_DIR / "pca_sweep_summary.csv"
    summary.to_csv(out_summary, index=False)
    print(f"Wrote {len(summary)} summary rows → {out_summary}")

    # Print QSVC pivot: n_components vs endpoint (mean AUROC)
    qsvc_df = summary[summary["model"] == "qsvc"].copy()
    if not qsvc_df.empty:
        pivot = qsvc_df.pivot_table(
            index="endpoint", columns="n_components", values="auroc_mean", aggfunc="mean"
        )
        print("\nQSVC AUROC by endpoint × n_components (mean over featurisers):")
        print(pivot.round(3).to_string())


if __name__ == "__main__":
    main()
