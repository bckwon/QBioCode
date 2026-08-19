#!/usr/bin/env python3
"""
04b_build_test_results.py
==========================
Builds test_results.csv directly from existing dataset=test.csv ModelResults.csv
files in results/admet_config/, without needing fitted model checkpoints.

This replaces the 04_test_inference.py pipeline for cases where model .pkl
checkpoints were not saved during the qprofiler sweep.

Output columns match what 05_export_performance_table.py expects:
  endpoint, featurizer, model, val_f1_best_checkpoint,
  auroc, auprc, mcc, f1, accuracy
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Build test_results.csv from existing test-split ModelResults."
    )
    p.add_argument("--results-dir",  default="results/admet_config",
                   help="Root of the Hydra results tree.")
    p.add_argument("--valid-dir",    default="results/admet_config",
                   help="Same root, used to look up best val_f1 per (ep, feat, model).")
    p.add_argument("--output-dir",   default="results/admet_benchmark/test_results")
    return p.parse_args()


def read_config_fp(hydra_dir: str) -> str | None:
    """Return folder_path from .hydra/config.yaml, tolerating corrupt YAMLs."""
    cfg_path = os.path.join(hydra_dir, ".hydra", "config.yaml")
    if not os.path.exists(cfg_path):
        return None
    try:
        with open(cfg_path) as f:
            c = yaml.safe_load(f)
        return c.get("folder_path", "") if isinstance(c, dict) else None
    except Exception:
        # Fallback: raw text grep
        try:
            raw = open(cfg_path).read()
            m = re.search(r"folder_path:\s*(.+)", raw)
            return m.group(1).strip() if m else None
        except Exception:
            return None


def collect_split(results_dir: str, split: str) -> pd.DataFrame:
    """Walk dataset={split} dirs and collect all ModelResults rows with ep/feat."""
    split_dir = os.path.join(results_dir, f"dataset={split}")
    if not os.path.isdir(split_dir):
        log.warning(f"Split dir not found: {split_dir}")
        return pd.DataFrame()

    rows = []
    ep_re = re.compile(r"data/admet/([^/]+)/([^/]+?)/?$")

    for sim_name in os.listdir(split_dir):
        sim_dir = os.path.join(split_dir, sim_name)
        csv_path = os.path.join(sim_dir, "ModelResults.csv")
        if not os.path.isfile(csv_path):
            continue
        fp = read_config_fp(sim_dir)
        if not fp:
            continue
        m = ep_re.search(fp)
        if not m:
            continue
        endpoint, featurizer = m.group(1), m.group(2)
        if featurizer not in ("ecfp4", "maccs", "rdkit200"):
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            log.warning(f"  Could not read {csv_path}: {e}")
            continue
        if df.empty:
            continue
        df["endpoint"]   = endpoint
        df["featurizer"] = featurizer
        rows.append(df)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_val_f1_index(valid_df: pd.DataFrame) -> dict:
    """Best val_f1 per (endpoint, featurizer, model) across all valid.csv rows."""
    if valid_df.empty or "f1_score" not in valid_df.columns:
        return {}
    idx = {}
    for (ep, feat, model), grp in valid_df.groupby(
            ["endpoint", "featurizer", "model"], observed=True):
        idx[(ep, feat, model)] = grp["f1_score"].max()
    return idx


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    log.info("=" * 60)
    log.info("Build test_results.csv from ModelResults")
    log.info(f"  results-dir : {args.results_dir}")
    log.info(f"  output-dir  : {args.output_dir}")
    log.info("=" * 60)

    # ── Collect valid.csv rows for best val_f1 lookup ────────────────────
    log.info("Collecting valid.csv rows for val_f1 index…")
    valid_df = collect_split(args.valid_dir, "valid.csv")
    val_f1_idx = build_val_f1_index(valid_df)
    log.info(f"  {len(val_f1_idx)} (endpoint, featurizer, model) val_f1 entries")

    # ── Collect test.csv rows ────────────────────────────────────────────
    log.info("Collecting test.csv rows…")
    test_df = collect_split(args.results_dir, "test.csv")
    if test_df.empty:
        log.error("No test.csv ModelResults found — aborting.")
        sys.exit(1)
    log.info(f"  Raw rows: {len(test_df)}")

    # ── Rename columns to match 05_export_performance_table expectation ──
    test_df = test_df.rename(columns={
        "auc":      "auroc",
        "f1_score": "f1",
    })

    # ── Deduplicate: keep best auroc per (endpoint, featurizer, model) ───
    # Multiple runs may exist; keep the one with highest test auroc.
    test_df = (
        test_df
        .sort_values("auroc", ascending=False)
        .drop_duplicates(subset=["endpoint", "featurizer", "model"], keep="first")
        .reset_index(drop=True)
    )
    log.info(f"  After dedup: {len(test_df)} rows "
             f"({test_df['endpoint'].nunique()} endpoints × "
             f"{test_df['featurizer'].nunique()} featurizers × "
             f"{test_df['model'].nunique()} models)")

    # ── Add val_f1_best_checkpoint ────────────────────────────────────────
    test_df["val_f1_best_checkpoint"] = test_df.apply(
        lambda r: val_f1_idx.get((r["endpoint"], r["featurizer"], r["model"]),
                                 float("nan")),
        axis=1,
    )

    # ── Select and order output columns ──────────────────────────────────
    keep = ["endpoint", "featurizer", "model", "val_f1_best_checkpoint",
            "auroc", "auprc", "mcc", "f1", "accuracy"]
    available = [c for c in keep if c in test_df.columns]
    out = test_df[available].sort_values(["endpoint", "featurizer", "model"])

    # ── Write outputs ─────────────────────────────────────────────────────
    out_csv = os.path.join(args.output_dir, "test_results.csv")
    out.to_csv(out_csv, index=False)
    log.info(f"Wrote {len(out)} rows → {out_csv}")

    # Summary by model
    summary = (
        out.groupby("model")["auroc"]
        .agg(mean_auroc="mean", std_auroc="std", n="count")
        .reset_index()
        .sort_values("mean_auroc", ascending=False)
    )
    summary_csv = os.path.join(args.output_dir, "test_summary_by_model.csv")
    summary.to_csv(summary_csv, index=False)
    log.info(f"Wrote model summary → {summary_csv}")
    log.info("\nModel AUROC summary:")
    log.info(summary.to_string(index=False))


if __name__ == "__main__":
    main()
