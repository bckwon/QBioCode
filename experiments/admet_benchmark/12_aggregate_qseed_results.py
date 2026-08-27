#!/usr/bin/env python3
"""
Aggregate L2 VQC q_seed sweep results.

Collects ModelResults.csv from:
  - q_seed=42 baseline: results/admet_config/dataset=test.csv/**/ModelResults.csv
    (filtered to VQC rows and priority endpoints)
  - New q_seed runs (L2): results/admet_vqc_qseed_*/dataset=*/

Produces:
  results/admet_benchmark/tables/vqc_qseed_raw.csv
    — all VQC rows for priority endpoints, all q_seeds
  results/admet_benchmark/tables/vqc_qseed_summary.csv
    — mean ± std AUROC per (endpoint, featuriser, q_seed) across featurisers

Run AFTER all L2 jobs complete:
    .venv/bin/python3 experiments/admet_benchmark/12_aggregate_qseed_results.py
"""
import glob
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

TABLES_DIR = REPO / "results/admet_benchmark/tables"
FEATS = ["ecfp4", "maccs", "rdkit200"]
PRIORITY_EPS = {
    "CYP2C9_Substrate_CarbonMangels",
    "Clearance_Hepatocyte_AZ",
    "CYP2D6_Substrate_CarbonMangels",
}


def split_data_type(dt: str):
    for feat in FEATS:
        if f"_{feat}" in dt:
            base = dt[:dt.index(f"_{feat}")]
            tail = dt[dt.index(f"_{feat}") + len(f"_{feat}"):]
            return base, feat, tail
    return dt, "unknown", ""


def load_vqc_results(pattern: str, q_seed: int) -> list[dict]:
    rows = []
    for fpath in glob.glob(pattern, recursive=True):
        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            print(f"  WARN: {fpath}: {e}")
            continue
        if df.empty:
            continue

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

        # Filter to VQC model and priority endpoints
        vqc_rows = df[df["model"] == "vqc"] if "model" in df.columns else df
        for _, row in vqc_rows.iterrows():
            rows.append({
                "endpoint": ep,
                "featurizer": feat,
                "embeddings": row.get("embeddings", "unknown"),
                "model": "vqc",
                "q_seed": q_seed,
                "auroc": row.get("auc", float("nan")),
                "auprc": row.get("auprc", float("nan")),
                "mcc": row.get("mcc", float("nan")),
                "f1": row.get("f1_score", float("nan")),
                "source_file": str(fpath),
            })
    return rows


def main():
    rows = []

    # 1. q_seed=42 baseline from main admet_config results
    print("Loading q_seed=42 VQC baseline …")
    pattern42 = str(REPO / "results/admet_config/dataset=test.csv/**/ModelResults.csv")
    r = load_vqc_results(pattern42, q_seed=42)
    r = [x for x in r if x["endpoint"] in PRIORITY_EPS]
    print(f"  {len(r)} VQC rows for priority endpoints")
    rows.extend(r)

    # 2. New q_seed runs (L2)
    Q_SEEDS = [0, 7, 21, 73, 84, 100, 123, 200, 314]
    for qs in Q_SEEDS:
        print(f"Loading q_seed={qs} …")
        pattern = str(REPO / f"results/admet_vqc_qseed_{qs}*/dataset=*/simulator_**/ModelResults.csv")
        r = load_vqc_results(pattern, q_seed=qs)
        r = [x for x in r if x["endpoint"] in PRIORITY_EPS]
        print(f"  {len(r)} rows")
        rows.extend(r)

    if not rows:
        print("No VQC q_seed results found. Run L2 jobs first.")
        return

    all_df = pd.DataFrame(rows)
    out_raw = TABLES_DIR / "vqc_qseed_raw.csv"
    all_df.to_csv(out_raw, index=False)
    print(f"\nWrote {len(all_df)} rows → {out_raw}")

    # Summary: mean ± std per (endpoint, featuriser, embeddings)
    key_cols = ["endpoint", "featurizer", "embeddings"]
    summary = (
        all_df.groupby(key_cols)["auroc"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    summary.columns = key_cols + ["auroc_mean", "auroc_std", "auroc_min", "auroc_max", "n_qseeds"]
    out_summary = TABLES_DIR / "vqc_qseed_summary.csv"
    summary.to_csv(out_summary, index=False)
    print(f"Wrote {len(summary)} summary rows → {out_summary}")
    print(summary[["endpoint", "featurizer", "embeddings",
                   "auroc_mean", "auroc_std", "n_qseeds"]].to_string(index=False))


if __name__ == "__main__":
    main()
