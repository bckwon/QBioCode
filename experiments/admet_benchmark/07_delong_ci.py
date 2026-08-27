#!/usr/bin/env python3
"""
Phase 0: Compute 95% DeLong-style confidence intervals for AUROC.

Since ModelResults.csv stores only aggregate AUROC (no per-compound probability
scores), we use the DeLong variance formula directly from the AUROC point
estimate and the positive/negative class counts (Hanley & McNeil 1982, eq. 5).

  V(AUC) = [AUC*(1-AUC) + (n_pos-1)*Q1 + (n_neg-1)*Q2] / (n_pos * n_neg)
  where Q1 = AUC / (2-AUC),  Q2 = 2*AUC^2 / (1+AUC)

This is the closed-form DeLong SE under the assumption of a trapezoidal AUC
estimator, which is accurate for large test sets (n_test >= 100).

Inputs:
  results/admet_benchmark/tables/performance_table.csv
    — endpoint, featurizer, model, auroc, auprc, mcc ...
  results/admet_config/dataset=test.csv/**/ModelResults.csv
    — Dataset col = test.csv, # Samples = n_test (total), plus embeddings
  We infer n_pos from the class prevalence stored in the 22-endpoint list
  or directly from the ModelResults # Samples and the performance_table.

Outputs:
  results/admet_benchmark/tables/auroc_ci_delong.csv
    — endpoint, featurizer, model, auroc, ci_lo, ci_hi, n_test, n_pos, n_neg, se
  results/admet_benchmark/tables/auroc_ci_qml_wins.csv
    — filtered to the 3 QML-win endpoints

Usage:
    cd /proj/bmfm/users/bckwon/projects/QBioCode
    .venv/bin/python3 experiments/admet_benchmark/07_delong_ci.py
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

TABLES_DIR = REPO / "results/admet_benchmark/tables"
ALPHA = 0.05
Z = stats.norm.ppf(1 - ALPHA / 2)   # 1.96

QML_WIN_ENDPOINTS = {
    "CYP2C9_Substrate_CarbonMangels",
    "Clearance_Hepatocyte_AZ",
    "CYP2D6_Substrate_CarbonMangels",
}


# ---------------------------------------------------------------------------
# DeLong variance (Hanley & McNeil 1982, eq. 5)
# ---------------------------------------------------------------------------
def delong_ci(auroc: float, n_pos: int, n_neg: int, alpha: float = ALPHA):
    """Closed-form DeLong 95% CI from AUROC + class counts."""
    if n_pos <= 0 or n_neg <= 0 or np.isnan(auroc):
        return float("nan"), float("nan"), float("nan")
    A = float(auroc)
    Q1 = A / (2.0 - A)
    Q2 = 2.0 * A * A / (1.0 + A)
    var = (A * (1 - A) + (n_pos - 1) * Q1 + (n_neg - 1) * Q2) / (n_pos * n_neg)
    se = np.sqrt(max(var, 0.0))
    z = stats.norm.ppf(1 - alpha / 2)
    return max(0.0, A - z * se), min(1.0, A + z * se), se


# ---------------------------------------------------------------------------
# Load test-set sizes from ModelResults.csv files
# Map: (data_type, embeddings) → (n_test, n_features)
# We'll join by (endpoint, featurizer, embeddings) using data_type = ep_feat
# ---------------------------------------------------------------------------
def load_test_sizes() -> pd.DataFrame:
    """Return DataFrame with columns: endpoint, featurizer, embeddings, n_test."""
    pattern = str(REPO / "results/admet_config/dataset=test.csv/**/ModelResults.csv")
    rows = []
    for fpath in glob.glob(pattern, recursive=True):
        # Extract data_type from .hydra/config.yaml in the same run dir
        run_dir = Path(fpath).parent
        hydra_cfg = run_dir / ".hydra" / "config.yaml"
        if not hydra_cfg.exists():
            continue
        # Parse data_type quickly
        data_type = None
        with open(hydra_cfg) as hf:
            for line in hf:
                if line.startswith("data_type:"):
                    data_type = line.split(":", 1)[1].strip()
                    break
        if not data_type:
            continue

        df = pd.read_csv(fpath)
        if "# Samples" not in df.columns or "embeddings" not in df.columns:
            continue
        # Each row is one (embeddings, model) combination; n_test is the same
        # across all rows in this file (same test set).
        sub = df[["embeddings", "# Samples"]].drop_duplicates()
        sub = sub.rename(columns={"# Samples": "n_test"})
        sub["data_type"] = data_type
        rows.append(sub)

    if not rows:
        return pd.DataFrame()

    all_df = pd.concat(rows, ignore_index=True)

    # Split data_type into endpoint + featurizer
    # data_type format: "{endpoint}_{feat}"  where feat ∈ {ecfp4, maccs, rdkit200}
    FEATS = ["ecfp4", "maccs", "rdkit200"]

    def split_data_type(dt):
        for feat in FEATS:
            suffix = f"_{feat}"
            if dt.endswith(suffix):
                return dt[: -len(suffix)], feat
        # Fallback — shouldn't happen with canonical endpoints
        parts = dt.rsplit("_", 1)
        return parts[0], parts[1] if len(parts) == 2 else ("", dt)

    all_df[["endpoint", "featurizer"]] = pd.DataFrame(
        all_df["data_type"].apply(split_data_type).tolist(),
        index=all_df.index,
    )
    # Keep most recent n_test per (endpoint, featurizer, embeddings)
    all_df = (
        all_df.sort_values("n_test", ascending=False)
        .groupby(["endpoint", "featurizer", "embeddings"], as_index=False)
        .first()[["endpoint", "featurizer", "embeddings", "n_test"]]
    )
    return all_df


# ---------------------------------------------------------------------------
# Estimate n_pos from existing results/admet_config ModelResults.csv
# Use: n_test from the file + class balance from the Dataset column
# Simpler: read test.csv directly from data/admet/
# ---------------------------------------------------------------------------
def load_class_counts() -> pd.DataFrame:
    """Return DataFrame: endpoint, featurizer, n_test, n_pos, n_neg."""
    rows = []
    data_root = REPO / "data" / "admet"
    FEATS = ["ecfp4", "maccs", "rdkit200"]
    for ep_dir in sorted(data_root.iterdir()):
        if not ep_dir.is_dir():
            continue
        ep = ep_dir.name
        for feat in FEATS:
            test_csv = ep_dir / feat / "test.csv"
            if not test_csv.exists():
                continue
            df = pd.read_csv(test_csv)
            if "Y" not in df.columns:
                continue
            n_test = len(df)
            n_pos = int((df["Y"] >= 0.5).sum()) if df["Y"].dtype != object else int((df["Y"] == 1).sum())
            n_neg = n_test - n_pos
            rows.append({"endpoint": ep, "featurizer": feat,
                          "n_test": n_test, "n_pos": n_pos, "n_neg": n_neg})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading performance table …")
    perf = pd.read_csv(TABLES_DIR / "performance_table.csv")

    print("Loading class counts from test.csv files …")
    counts = load_class_counts()
    print(f"  Found class counts for {len(counts)} (endpoint, featurizer) pairs")

    # Merge performance table with class counts
    merged = perf.merge(counts, on=["endpoint", "featurizer"], how="left")

    if merged["n_pos"].isna().any():
        missing = merged[merged["n_pos"].isna()][["endpoint", "featurizer"]].drop_duplicates()
        print(f"WARNING: Missing class counts for {len(missing)} pairs:\n{missing.to_string()}")

    # Compute CIs
    ci_rows = []
    for _, row in merged.iterrows():
        auroc = row.get("auroc", float("nan"))
        n_pos = row.get("n_pos", float("nan"))
        n_neg = row.get("n_neg", float("nan"))
        if pd.isna(n_pos) or pd.isna(n_neg):
            lo, hi, se = float("nan"), float("nan"), float("nan")
        else:
            lo, hi, se = delong_ci(auroc, int(n_pos), int(n_neg))
        ci_rows.append({
            "endpoint": row["endpoint"],
            "featurizer": row["featurizer"],
            "model": row["model"],
            "auroc": auroc,
            "ci_lo": round(lo, 4) if not np.isnan(lo) else lo,
            "ci_hi": round(hi, 4) if not np.isnan(hi) else hi,
            "se": round(se, 4) if not np.isnan(se) else se,
            "n_test": int(row["n_test"]) if not pd.isna(row.get("n_test", float("nan"))) else float("nan"),
            "n_pos": int(n_pos) if not pd.isna(n_pos) else float("nan"),
            "n_neg": int(n_neg) if not pd.isna(n_neg) else float("nan"),
            "method": "delong_hanley_mcneil",
        })

    ci_df = pd.DataFrame(ci_rows)
    out_all = TABLES_DIR / "auroc_ci_delong.csv"
    ci_df.to_csv(out_all, index=False)
    print(f"\nWrote {len(ci_df)} rows → {out_all}")

    # QML-win subset
    wins_df = ci_df[ci_df["endpoint"].isin(QML_WIN_ENDPOINTS)].copy()
    out_wins = TABLES_DIR / "auroc_ci_qml_wins.csv"
    wins_df.to_csv(out_wins, index=False)
    print(f"Wrote {len(wins_df)} rows → {out_wins}")

    # Summary: print CI width statistics
    ci_df["ci_width"] = ci_df["ci_hi"] - ci_df["ci_lo"]
    print("\nCI width statistics across all (endpoint, feat, model):")
    print(ci_df["ci_width"].describe().round(4).to_string())
    print("\nQML-win endpoints CI summary:")
    print(
        wins_df[["endpoint", "model", "auroc", "ci_lo", "ci_hi", "n_test"]]
        .sort_values(["endpoint", "model"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
