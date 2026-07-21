#!/usr/bin/env python3
"""
01_prepare_admet_datasets.py
============================
Phase 1 experiment script: Download all 22 TDC ADMET endpoints, apply
binarization, create QML-capped training splits, and featurize with
ECFP4, MACCS, and RDKit200.

Output layout::

    data/admet/
    ├── metadata.json
    ├── {endpoint}/
    │   ├── train_smiles.csv, valid_smiles.csv, test_smiles.csv
    │   ├── train_qml_smiles.csv
    │   ├── ecfp4/  train.csv  valid.csv  test.csv  train_qml.csv
    │   ├── maccs/  train.csv  valid.csv  test.csv  train_qml.csv
    │   └── rdkit200/ train.csv  valid.csv  test.csv  train_qml.csv
    └── ...

Usage::

    # From repo root
    .venv/bin/python experiments/admet_benchmark/01_prepare_admet_datasets.py

    # Subset of endpoints
    .venv/bin/python experiments/admet_benchmark/01_prepare_admet_datasets.py \\
        --endpoints hERG AMES DILI

    # Custom output directory
    .venv/bin/python experiments/admet_benchmark/01_prepare_admet_datasets.py \\
        --data-dir /scratch/admet_data
"""

import argparse
import logging
import os
import sys

# Ensure repo root is on path when running from experiments/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from qbiocode.data_adapters import ADMET_ENDPOINTS, MolecularFeaturizer, TDCAdmetLoader
from qbiocode.data_adapters.molecular_featurizers import FEATURIZERS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare TDC ADMET datasets for QBioCode QProfiler sweep."
    )
    parser.add_argument(
        "--endpoints",
        nargs="+",
        default=None,
        metavar="ENDPOINT",
        help=(
            "Subset of endpoint names to prepare (default: all 22). "
            f"Available: {list(ADMET_ENDPOINTS.keys())}"
        ),
    )
    parser.add_argument(
        "--data-dir",
        default="data/admet",
        help="Root directory for output data (default: data/admet)",
    )
    parser.add_argument(
        "--featurizers",
        nargs="+",
        default=FEATURIZERS,
        choices=FEATURIZERS,
        help=f"Featurizers to run (default: all — {FEATURIZERS})",
    )
    parser.add_argument(
        "--qml-cap",
        type=int,
        default=300,
        help="Max training samples for QML-capped split (default: 300)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    log.info("=" * 60)
    log.info("QBioCode-ADMET: Dataset Preparation")
    log.info("=" * 60)
    log.info(f"Output directory : {args.data_dir}")
    log.info(f"Endpoints        : {args.endpoints or 'ALL (' + str(len(ADMET_ENDPOINTS)) + ')'}")
    log.info(f"Featurizers      : {args.featurizers}")
    log.info(f"QML train cap    : {args.qml_cap}")
    log.info(f"Seed             : {args.seed}")
    log.info("=" * 60)

    # ── Step 1: Download + binarize via TDC ──────────────────────────────────
    log.info("\n[Step 1] Downloading and binarizing TDC ADMET endpoints...")
    loader = TDCAdmetLoader(
        data_dir=args.data_dir,
        qml_train_cap=args.qml_cap,
        seed=args.seed,
    )
    metadata = loader.prepare_all(endpoints=args.endpoints)
    log.info(f"  ✓ {len(metadata)} endpoints prepared.")

    # ── Step 2: Molecular featurization ──────────────────────────────────────
    log.info("\n[Step 2] Featurizing SMILES with molecular fingerprints...")
    endpoints_to_featurize = args.endpoints or list(metadata.keys())

    for feat_name in args.featurizers:
        log.info(f"\n  Featurizer: {feat_name.upper()}")
        feat = MolecularFeaturizer(featurizer=feat_name)
        for endpoint in endpoints_to_featurize:
            ep_dir = os.path.join(args.data_dir, endpoint)
            if not os.path.isdir(ep_dir):
                log.warning(f"    Skipping {endpoint} — directory not found")
                continue
            try:
                feat.featurize_endpoint(ep_dir)
            except Exception as exc:
                log.error(f"    FAILED {endpoint}/{feat_name}: {exc}")

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("Preparation complete. Summary:")
    log.info(f"  Endpoints processed : {len(metadata)}")
    log.info(f"  Featurizers applied : {args.featurizers}")
    log.info(f"  Output root         : {os.path.abspath(args.data_dir)}")
    log.info(f"  Metadata            : {os.path.join(args.data_dir, 'metadata.json')}")
    log.info("=" * 60)

    # ── Verify a few output files ─────────────────────────────────────────────
    log.info("\nSpot-checking output files:")
    import pandas as pd
    n_ok, n_fail = 0, 0
    for ep in list(endpoints_to_featurize)[:3]:
        for feat_name in args.featurizers:
            path = os.path.join(args.data_dir, ep, feat_name, "train.csv")
            if os.path.exists(path):
                df = pd.read_csv(path)
                log.info(f"  ✓ {ep}/{feat_name}/train.csv — {df.shape[0]} rows × {df.shape[1]} cols")
                n_ok += 1
            else:
                log.warning(f"  ✗ {ep}/{feat_name}/train.csv — MISSING")
                n_fail += 1
    log.info(f"\nSpot-check: {n_ok} OK, {n_fail} missing")


if __name__ == "__main__":
    main()
