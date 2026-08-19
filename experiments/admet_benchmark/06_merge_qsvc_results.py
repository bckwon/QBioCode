#!/usr/bin/env python3
"""
06_merge_qsvc_results.py
=========================
Merges QSVC re-run results (from results/admet_qsvc_config/) into the master
results (results/admet_config/), replacing only the QSVC rows.

Background
----------
The original ADMET sweep (job_02) produced degenerate QSVC results (AUC=0.5)
due to two bugs:
  1. ZZFeatureMap kernel collapse: PCA/UMAP output is zero-centred but
     ZZFeatureMap needs inputs in [0,1]. Fix: MinMaxScaler applied inside
     compute_qsvc.py before kernel computation.
  2. C=0.01 too small: no separating boundary can form. Fix: C changed to 1.0.

The QSVC re-run (job_02g) runs ONLY qsvc with the fixed config and writes
results to results/admet_qsvc_config/ — a completely separate directory tree.

This script:
  1. Discovers all result dirs in results/admet_qsvc_config/ via .hydra/config.yaml
     (which contains folder_path = the exact input data directory, uniquely
     identifying endpoint × featurizer).
  2. For each QSVC result dir, finds the matching master dir(s) in
     results/admet_config/ using the same folder_path key.
  3. In each matched master ModelResults.csv:
     - Drops all rows where model == 'qsvc'
     - Appends the new QSVC rows from the re-run
     - Writes back in-place (original backed up as ModelResults.csv.bak)

Usage
-----
    # Dry-run (see what would change, no writes):
    .venv/bin/python experiments/admet_benchmark/06_merge_qsvc_results.py --dry-run

    # Live merge:
    .venv/bin/python experiments/admet_benchmark/06_merge_qsvc_results.py

    # Custom paths:
    .venv/bin/python experiments/admet_benchmark/06_merge_qsvc_results.py \\
        --qsvc-dir  results/admet_qsvc_config \\
        --master-dir results/admet_config \\
        --dry-run

Prerequisites
-------------
- job_02g_qsvc_rerun.sh must be complete (results/admet_qsvc_config/ populated).
- Run from repo root.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_hydra_config(sim_dir: Path) -> dict | None:
    """Read .hydra/config.yaml from a simulator result directory."""
    cfg_path = sim_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        return None
    try:
        with open(cfg_path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        log.warning(f"Could not read {cfg_path}: {e}")
        return None


def find_model_results(root: Path) -> list[Path]:
    """Walk root and return all ModelResults.csv paths."""
    return sorted(root.rglob("ModelResults.csv"))


def get_folder_path(sim_dir: Path) -> str | None:
    """Return the folder_path stored in the Hydra config for this run."""
    cfg = load_hydra_config(sim_dir)
    if cfg is None:
        return None
    return cfg.get("folder_path")


# ── Core logic ───────────────────────────────────────────────────────────────

def build_master_index(master_dir: Path) -> dict[tuple, list[Path]]:
    """
    Index master ModelResults.csv files by (folder_path, dataset_split).

    Returns
    -------
    dict mapping (folder_path, dataset_split) → list of ModelResults.csv paths
    """
    index: dict[tuple, list[Path]] = {}
    for csv_path in find_model_results(master_dir):
        sim_dir = csv_path.parent
        fp = get_folder_path(sim_dir)
        if fp is None:
            continue
        # dataset split is encoded in the parent-parent folder name: dataset=test.csv
        split_dir = sim_dir.parent
        split = split_dir.name.replace("dataset=", "")
        key = (fp, split)
        index.setdefault(key, []).append(csv_path)
    log.info(f"Master index: {len(index)} unique (folder_path, split) keys "
             f"across {sum(len(v) for v in index.values())} CSV files")
    return index


def find_qsvc_results(qsvc_dir: Path) -> list[tuple[str, str, Path]]:
    """
    Find all QSVC result CSVs and return (folder_path, split, csv_path) tuples.
    """
    results = []
    for csv_path in find_model_results(qsvc_dir):
        sim_dir = csv_path.parent
        fp = get_folder_path(sim_dir)
        if fp is None:
            log.warning(f"No .hydra/config.yaml in {sim_dir}, skipping")
            continue
        split_dir = sim_dir.parent
        split = split_dir.name.replace("dataset=", "")
        results.append((fp, split, csv_path))
    log.info(f"Found {len(results)} QSVC result CSVs in {qsvc_dir}")
    return results


def merge_qsvc_into_master(
    qsvc_csv: Path,
    master_csvs: list[Path],
    dry_run: bool,
) -> dict:
    """
    Replace QSVC rows in each master CSV with rows from qsvc_csv.

    Returns a summary dict with counts.
    """
    # Load new QSVC rows
    try:
        qsvc_df = pd.read_csv(qsvc_csv)
    except Exception as e:
        log.error(f"Cannot read QSVC CSV {qsvc_csv}: {e}")
        return {"error": str(e)}

    # Filter to only qsvc rows (safety — config should only produce qsvc)
    qsvc_df = qsvc_df[qsvc_df["model"] == "qsvc"].copy()
    if len(qsvc_df) == 0:
        log.warning(f"No qsvc rows in {qsvc_csv}, skipping")
        return {"skipped": True}

    n_new = len(qsvc_df)
    summary = {"qsvc_csv": str(qsvc_csv), "new_rows": n_new, "masters_updated": 0,
               "old_rows_removed": 0, "dry_run": dry_run}

    for master_csv in master_csvs:
        try:
            master_df = pd.read_csv(master_csv)
        except Exception as e:
            log.error(f"Cannot read master CSV {master_csv}: {e}")
            continue

        # Count old QSVC rows to be replaced
        old_qsvc = (master_df["model"] == "qsvc").sum()
        # Drop old QSVC rows
        master_df_clean = master_df[master_df["model"] != "qsvc"].copy()
        # Append new QSVC rows
        merged = pd.concat([master_df_clean, qsvc_df], ignore_index=True)

        summary["old_rows_removed"] += old_qsvc
        summary["masters_updated"] += 1

        if dry_run:
            log.info(
                f"  [DRY-RUN] {master_csv}: "
                f"would remove {old_qsvc} old QSVC rows, add {n_new} new rows"
            )
        else:
            # Back up original
            bak = master_csv.with_suffix(".csv.bak")
            if not bak.exists():
                shutil.copy2(master_csv, bak)
            # Write merged
            merged.to_csv(master_csv, index=False)
            log.info(
                f"  ✅ {master_csv}: "
                f"removed {old_qsvc} old rows, added {n_new} new rows"
            )

    return summary


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge QSVC re-run results into master ADMET results."
    )
    p.add_argument(
        "--qsvc-dir",
        default="results/admet_qsvc_config",
        help="Root of QSVC re-run results (default: results/admet_qsvc_config)",
    )
    p.add_argument(
        "--master-dir",
        default="results/admet_config",
        help="Root of master results to update (default: results/admet_config)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without writing anything",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    qsvc_dir  = Path(args.qsvc_dir)
    master_dir = Path(args.master_dir)

    if not qsvc_dir.exists():
        log.error(f"QSVC results dir not found: {qsvc_dir}")
        log.error("Run job_02g_qsvc_rerun.sh first, then re-run this script.")
        sys.exit(1)
    if not master_dir.exists():
        log.error(f"Master results dir not found: {master_dir}")
        sys.exit(1)

    mode = "DRY-RUN" if args.dry_run else "LIVE"
    log.info("=" * 60)
    log.info(f"QSVC Results Merge  [{mode}]")
    log.info(f"  QSVC source : {qsvc_dir}")
    log.info(f"  Master dest : {master_dir}")
    log.info("=" * 60)

    # Build master index: (folder_path, split) → [csv_paths]
    master_index = build_master_index(master_dir)

    # Find all QSVC result CSVs
    qsvc_results = find_qsvc_results(qsvc_dir)
    if not qsvc_results:
        log.error("No QSVC result CSVs found — is the re-run complete?")
        sys.exit(1)

    # Deduplicate: if multiple runs produced results for the same
    # (folder_path, split) key (e.g. from job 75734 + job 76376), keep only
    # the most recently modified CSV.  All runs use the same fixed config so
    # any result is equally valid, but applying two to the same master file
    # would cause a redundant second rewrite.
    dedup: dict[tuple, Path] = {}
    for folder_path, split, qsvc_csv in qsvc_results:
        key = (folder_path, split)
        if key not in dedup or qsvc_csv.stat().st_mtime > dedup[key].stat().st_mtime:
            dedup[key] = qsvc_csv
    n_before = len(qsvc_results)
    qsvc_results = [(fp, sp, csv) for (fp, sp), csv in dedup.items()]
    if len(qsvc_results) < n_before:
        log.info(
            f"Deduplicated QSVC results: {n_before} → {len(qsvc_results)} "
            f"(kept most recent per ep/feat/split)"
        )

    # Merge
    total_new = total_removed = total_masters = 0
    unmatched = []

    for folder_path, split, qsvc_csv in qsvc_results:
        key = (folder_path, split)
        master_csvs = master_index.get(key, [])

        if not master_csvs:
            log.warning(f"No master match for key ({Path(folder_path).name}, {split})")
            unmatched.append(key)
            continue

        log.info(f"\n→ {Path(folder_path).name} / {split} "
                 f"({len(master_csvs)} master file(s))")
        summary = merge_qsvc_into_master(qsvc_csv, master_csvs, args.dry_run)

        total_new      += summary.get("new_rows", 0)
        total_removed  += summary.get("old_rows_removed", 0)
        total_masters  += summary.get("masters_updated", 0)

    log.info("\n" + "=" * 60)
    log.info(f"Merge complete [{mode}]")
    log.info(f"  Master CSVs updated : {total_masters}")
    log.info(f"  Old QSVC rows removed: {total_removed}")
    log.info(f"  New QSVC rows added  : {total_new}")
    if unmatched:
        log.warning(f"  Unmatched QSVC keys  : {len(unmatched)}")
        for k in unmatched:
            log.warning(f"    {k}")
    if not args.dry_run and total_masters > 0:
        log.info("  Backups saved as ModelResults.csv.bak (first merge only)")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
