#!/usr/bin/env python3
"""
05_export_performance_table.py
==============================
Produces a clean, publication-ready performance table from test-set inference
results, combining metrics across all endpoints, featurizers, and models.

Outputs (all written to --output-dir):
  performance_table.csv          — full table: endpoint × featurizer × model × metrics
  performance_summary_model.csv  — mean ± std across endpoints, grouped by model
  performance_summary_endpoint.csv — best model per endpoint with all metrics
  performance_by_category.csv    — mean AUROC per ADMET category × model
  qml_vs_classical.csv           — pairwise QML advantage (ΔF1, ΔAUROC) per endpoint
  performance_table_wide.csv     — wide pivot: rows=endpoint, cols=model (AUROC)
  performance_table.txt          — human-readable ASCII table (printed + saved)

Usage::

    .venv/bin/python experiments/admet_benchmark/05_export_performance_table.py

    .venv/bin/python experiments/admet_benchmark/05_export_performance_table.py \\
        --test-results  results/admet_benchmark/test_results/test_results.csv \\
        --metadata      data/admet/metadata.json \\
        --output-dir    results/admet_benchmark/tables
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model groupings for QML-vs-classical analysis
# ---------------------------------------------------------------------------
QML_MODELS: List[str] = ["qsvc", "vqc", "qnn", "pqk", "qensemble"]
CLASSICAL_MODELS: List[str] = ["svc", "rf", "mlp", "xgb", "lr"]
FOUNDATION_MODELS: List[str] = ["mmelon"]
ALL_MODELS: List[str] = QML_MODELS + CLASSICAL_MODELS + FOUNDATION_MODELS

# Display-friendly model names
MODEL_DISPLAY: Dict[str, str] = {
    "qsvc": "QSVC", "vqc": "VQC", "qnn": "QNN", "pqk": "PQK",
    "qensemble": "QEnsemble",
    "svc": "SVC", "rf": "RF", "mlp": "MLP", "xgb": "XGBoost", "lr": "LR",
    "mmelon": "MMELON",
}

# Metric columns and their display names
METRICS: Dict[str, str] = {
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "mcc":   "MCC",
    "f1":    "F1",
    "accuracy": "Accuracy",
}

# ADMET category lookup (duplicated from loader so this script is self-contained)
ENDPOINT_CATEGORY: Dict[str, str] = {
    "Caco2_Wang": "Absorption", "HIA_Hou": "Absorption",
    "Pgp_Broccatelli": "Absorption", "Bioavailability_Ma": "Absorption",
    "Lipophilicity_AstraZeneca": "Distribution", "Solubility_AqSolDB": "Distribution",
    "BBB_Martins": "Distribution", "PPBR_AstraZeneca": "Distribution",
    "VDss_Lombardo": "Distribution",
    "CYP2C19_Veith": "Metabolism", "CYP2D6_Veith": "Metabolism",
    "CYP3A4_Veith": "Metabolism", "CYP1A2_Veith": "Metabolism",
    "CYP2C9_Veith": "Metabolism",
    "CYP2C9_Substrate_CarbonMangels": "Metabolism",
    "CYP2D6_Substrate_CarbonMangels": "Metabolism",
    "CYP3A4_Substrate_CarbonMangels": "Metabolism",
    "Half_Life_Obach": "Excretion", "Clearance_Hepatocyte_AZ": "Excretion",
    "hERG": "Toxicity", "AMES": "Toxicity", "DILI": "Toxicity",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export clean performance tables from ADMET test-set results."
    )
    parser.add_argument(
        "--test-results",
        default="results/admet_benchmark/test_results/test_results.csv",
        help="Path to test_results.csv from 04_test_inference.py",
    )
    parser.add_argument(
        "--metadata",
        default="data/admet/metadata.json",
        help="Path to metadata.json from 01_prepare_admet_datasets.py",
    )
    parser.add_argument(
        "--output-dir",
        default="results/admet_benchmark/tables",
    )
    parser.add_argument(
        "--primary-metric",
        default="auroc",
        choices=list(METRICS.keys()),
        help="Primary metric used for ranking (default: auroc)",
    )
    parser.add_argument(
        "--primary-featurizer",
        default=None,
        help=(
            "If set, tables are computed from this featurizer only "
            "(default: mean over all three featurizers)"
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mean_std(series: pd.Series) -> str:
    """Format as 'mean ± std' for display."""
    return f"{series.mean():.4f} ± {series.std():.4f}"


def fmt(val) -> str:
    """Format a single float for display."""
    try:
        return f"{float(val):.4f}"
    except (TypeError, ValueError):
        return str(val)


def load_metadata(path: str) -> Dict:
    if not os.path.exists(path):
        log.warning(f"metadata.json not found at {path} — category info will be unavailable")
        return {}
    with open(path) as f:
        return json.load(f)


def enrich(df: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
    """Add category, original_task, n_train, class_balance columns from metadata."""
    df = df.copy()
    df["category"] = df["endpoint"].map(ENDPOINT_CATEGORY).fillna("Unknown")
    df["model_type"] = df["model"].apply(
        lambda m: "QML" if m in QML_MODELS else ("Foundation" if m in FOUNDATION_MODELS else "Classical")
    )
    df["model_display"] = df["model"].map(MODEL_DISPLAY).fillna(df["model"])

    if metadata:
        df["n_train"] = df["endpoint"].map(
            lambda e: metadata.get(e, {}).get("n_train", np.nan)
        )
        df["class_balance"] = df["endpoint"].map(
            lambda e: metadata.get(e, {}).get("class_balance_train", np.nan)
        )
        df["binarized"] = df["endpoint"].map(
            lambda e: metadata.get(e, {}).get("binarize", False)
        )
    return df


# ---------------------------------------------------------------------------
# Table generators
# ---------------------------------------------------------------------------

def make_full_table(df: pd.DataFrame) -> pd.DataFrame:
    """Full table: one row per (endpoint, featurizer, model)."""
    cols = ["endpoint", "category", "featurizer", "model", "model_display", "model_type",
            "val_f1_best_checkpoint"] + list(METRICS.keys())
    available = [c for c in cols if c in df.columns]
    full = df[available].copy()
    for m in METRICS:
        if m in full.columns:
            full[m] = full[m].round(4)
    return full.sort_values(["endpoint", "featurizer", "model"])


def make_summary_by_model(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean ± std of primary metric across endpoints, grouped by model."""
    rows = []
    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model]
        for m_col, m_label in METRICS.items():
            if m_col not in sub.columns:
                continue
            vals = sub[m_col].dropna()
            rows.append({
                "model": model,
                "model_display": MODEL_DISPLAY.get(model, model),
                "model_type": sub["model_type"].iloc[0] if len(sub) > 0 else "?",
                "metric": m_label,
                "mean": round(vals.mean(), 4),
                "std": round(vals.std(), 4),
                "median": round(vals.median(), 4),
                "min": round(vals.min(), 4),
                "max": round(vals.max(), 4),
                "n_endpoints": len(vals),
            })
    return pd.DataFrame(rows).sort_values(["metric", "mean"], ascending=[True, False])


def make_summary_by_endpoint(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Best model per endpoint across all featurizers."""
    rows = []
    for ep in sorted(df["endpoint"].unique()):
        sub = df[df["endpoint"] == ep]
        if metric not in sub.columns:
            continue
        best_idx = sub[metric].idxmax()
        best = sub.loc[best_idx]
        row = {
            "endpoint": ep,
            "category": best.get("category", "?"),
            "n_train": best.get("n_train", np.nan),
            "class_balance": best.get("class_balance", np.nan),
            "binarized": best.get("binarized", "?"),
            "best_model": best["model"],
            "best_model_type": best["model_type"],
            "best_featurizer": best.get("featurizer", "?"),
        }
        for m_col, m_label in METRICS.items():
            if m_col in best.index:
                row[m_label] = round(float(best[m_col]), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def make_wide_pivot(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Wide table: rows=endpoint, columns=model, values=mean metric over featurizers."""
    pivot = (
        df.groupby(["endpoint", "model"])[metric]
        .mean()
        .unstack("model")
        .round(4)
    )
    # Add category column and sort
    pivot.insert(0, "category", pivot.index.map(ENDPOINT_CATEGORY))
    pivot = pivot.sort_values(["category", "endpoint"])
    # Rename columns to display names
    pivot.columns = [MODEL_DISPLAY.get(c, c) if c != "category" else c for c in pivot.columns]
    return pivot


def make_category_summary(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean metric per ADMET category × model."""
    return (
        df.groupby(["category", "model"])[metric]
        .mean()
        .unstack("model")
        .round(4)
        .rename(columns=MODEL_DISPLAY)
    )


def make_qml_vs_classical(df: pd.DataFrame) -> pd.DataFrame:
    """Per-endpoint: best QML metric vs best classical metric → delta."""
    rows = []
    for ep in sorted(df["endpoint"].unique()):
        sub = df[df["endpoint"] == ep]
        for m_col, m_label in METRICS.items():
            if m_col not in sub.columns:
                continue
            qml_sub = sub[sub["model"].isin(QML_MODELS)][m_col].dropna()
            cls_sub = sub[sub["model"].isin(CLASSICAL_MODELS)][m_col].dropna()
            fnd_sub = sub[sub["model"].isin(FOUNDATION_MODELS)][m_col].dropna()
            if qml_sub.empty or cls_sub.empty:
                continue
            best_qml = qml_sub.max()
            best_cls = cls_sub.max()
            best_fnd = fnd_sub.max() if not fnd_sub.empty else np.nan
            best_qml_model = sub[sub["model"].isin(QML_MODELS)].loc[
                sub[sub["model"].isin(QML_MODELS)][m_col].idxmax(), "model"
            ]
            best_cls_model = sub[sub["model"].isin(CLASSICAL_MODELS)].loc[
                sub[sub["model"].isin(CLASSICAL_MODELS)][m_col].idxmax(), "model"
            ]
            rows.append({
                "endpoint": ep,
                "category": ENDPOINT_CATEGORY.get(ep, "Unknown"),
                "metric": m_label,
                "best_qml_score": round(best_qml, 4),
                "best_qml_model": best_qml_model,
                "best_classical_score": round(best_cls, 4),
                "best_classical_model": best_cls_model,
                "best_mmelon_score": round(best_fnd, 4) if not np.isnan(best_fnd) else np.nan,
                "delta_qml_minus_classical": round(best_qml - best_cls, 4),
                "qml_wins": best_qml > best_cls,
            })
    return pd.DataFrame(rows)


def make_ascii_table(summary_model: pd.DataFrame, metric: str) -> str:
    """Render a clean ASCII summary table for terminal / .txt output."""
    auroc_rows = summary_model[summary_model["metric"] == METRICS.get(metric, metric.upper())]
    if auroc_rows.empty:
        return "(no data)"

    lines = []
    sep = "+" + "-" * 14 + "+" + "-" * 12 + "+" + "-" * 8 + "+" + "-" * 8 + "+" + "-" * 8 + "+" + "-" * 8 + "+" + "-" * 6 + "+"
    header = f"| {'Model':<12} | {'Type':<10} | {'Mean':>6} | {'Std':>6} | {'Median':>6} | {'Max':>6} | {'N':>4} |"
    lines.append(sep)
    lines.append(f"| {METRICS.get(metric, metric.upper())} — Model Summary (mean over all endpoints × featurizers)".ljust(len(sep) - 2) + " |")
    lines.append(sep)
    lines.append(header)
    lines.append(sep)
    for _, row in auroc_rows.sort_values("mean", ascending=False).iterrows():
        model_d = MODEL_DISPLAY.get(row["model"], row["model"])
        lines.append(
            f"| {model_d:<12} | {row['model_type']:<10} | {row['mean']:>6.4f} | "
            f"{row['std']:>6.4f} | {row['median']:>6.4f} | {row['max']:>6.4f} | "
            f"{int(row['n_endpoints']):>4} |"
        )
    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    log.info("=" * 60)
    log.info("QBioCode-ADMET: Performance Table Export")
    log.info("=" * 60)

    # ── Load results ──────────────────────────────────────────────────────────
    if not os.path.exists(args.test_results):
        log.error(
            f"Test results not found: {args.test_results}\n"
            "Run 04_test_inference.py first."
        )
        sys.exit(1)

    df = pd.read_csv(args.test_results)
    metadata = load_metadata(args.metadata)
    log.info(f"Loaded {len(df)} rows from {args.test_results}")

    # Optionally filter to one featurizer
    if args.primary_featurizer:
        df = df[df["featurizer"] == args.primary_featurizer]
        log.info(f"Filtered to featurizer='{args.primary_featurizer}' → {len(df)} rows")

    # Enrich with category / model-type info
    df = enrich(df, metadata)

    metric = args.primary_metric

    # ── Build and save tables ──────────────────────────────────────────────────
    tables = {}

    tables["performance_table"] = make_full_table(df)
    tables["performance_summary_model"] = make_summary_by_model(df, metric)
    tables["performance_summary_endpoint"] = make_summary_by_endpoint(df, metric)
    tables["performance_table_wide"] = make_wide_pivot(df, metric)
    tables["performance_by_category"] = make_category_summary(df, metric)
    tables["qml_vs_classical"] = make_qml_vs_classical(df)

    for name, tbl in tables.items():
        path = os.path.join(args.output_dir, f"{name}.csv")
        tbl.to_csv(path, index=(name in ("performance_table_wide", "performance_by_category")))
        log.info(f"  Saved: {path}  ({tbl.shape[0]} rows × {tbl.shape[1]} cols)")

    # ── ASCII table ───────────────────────────────────────────────────────────
    ascii_str = make_ascii_table(tables["performance_summary_model"], metric)
    print("\n" + ascii_str + "\n")

    txt_path = os.path.join(args.output_dir, "performance_table.txt")
    with open(txt_path, "w") as f:
        f.write(ascii_str + "\n\n")

        # Append QML-wins summary
        qml_df = tables["qml_vs_classical"]
        auroc_qml = qml_df[qml_df["metric"] == "AUROC"]
        if not auroc_qml.empty:
            n_wins = auroc_qml["qml_wins"].sum()
            n_total = len(auroc_qml)
            f.write(f"\nQML wins (AUROC): {n_wins} / {n_total} endpoints\n\n")
            f.write("Endpoint breakdown (AUROC):\n")
            f.write(
                auroc_qml[
                    ["endpoint", "category", "best_qml_model", "best_qml_score",
                     "best_classical_model", "best_classical_score",
                     "best_mmelon_score", "delta_qml_minus_classical", "qml_wins"]
                ].to_string(index=False)
            )
            f.write("\n")

    log.info(f"  Saved: {txt_path}")

    # ── Terminal summary ──────────────────────────────────────────────────────
    qml_df = tables["qml_vs_classical"]
    auroc_rows = qml_df[qml_df["metric"] == "AUROC"]
    if not auroc_rows.empty:
        n_wins = int(auroc_rows["qml_wins"].sum())
        n_total = len(auroc_rows)
        log.info(f"\nQML wins (AUROC): {n_wins}/{n_total} endpoints")
        log.info(
            f"Mean ΔAUROC (QML − Classical): "
            f"{auroc_rows['delta_qml_minus_classical'].mean():+.4f}"
        )
        winners = auroc_rows[auroc_rows["qml_wins"]]["endpoint"].tolist()
        log.info(f"Endpoints where QML wins: {winners}")

    log.info("=" * 60)
    log.info(f"All tables written to {args.output_dir}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
