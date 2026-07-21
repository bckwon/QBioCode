#!/usr/bin/env python3
"""
04_test_inference.py
====================
Phase 5b experiment script: load the best model checkpoint for each
(endpoint × featurizer × model) combination and evaluate on the TDC
canonical test split.

This is the script that generates the **final paper metrics** — validation
F1 selects the best checkpoint; test-set metrics are reported using BOTH:
  1. sklearn metrics (AUROC, AUPRC, MCC, F1, Accuracy) — standard library
  2. TDC official evaluator (``admet_group.evaluate()``) — leaderboard-comparable

Outputs (all in --output-dir):
  test_results.csv           — full results, sklearn metrics
  test_results_tdc.csv       — TDC official metric per endpoint per model
  test_summary_by_model.csv  — mean sklearn metrics grouped by model

Usage::

    .venv/bin/python experiments/admet_benchmark/04_test_inference.py

    .venv/bin/python experiments/admet_benchmark/04_test_inference.py \\
        --data-dir data/admet \\
        --checkpoint-dir results/admet_benchmark/checkpoints \\
        --output-dir results/admet_benchmark/test_results \\
        --featurizers ecfp4 maccs rdkit200
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from qbiocode.utils.model_checkpoint import get_best_index, infer_from_best

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# TDC endpoint names that the official evaluator accepts
# (same as ADMET_ENDPOINTS keys in tdc_admet_loader.py)
_TDC_EVAL_NAMES = {
    "Caco2_Wang", "HIA_Hou", "Pgp_Broccatelli", "Bioavailability_Ma",
    "Lipophilicity_AstraZeneca", "Solubility_AqSolDB", "BBB_Martins",
    "PPBR_AstraZeneca", "VDss_Lombardo",
    "CYP2C19_Veith", "CYP2D6_Veith", "CYP3A4_Veith", "CYP1A2_Veith",
    "CYP2C9_Veith", "CYP2C9_Substrate_CarbonMangels",
    "CYP2D6_Substrate_CarbonMangels", "CYP3A4_Substrate_CarbonMangels",
    "Half_Life_Obach", "Clearance_Hepatocyte_AZ",
    "hERG", "AMES", "DILI",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test-set inference using best checkpoints from QProfiler ADMET sweep."
    )
    parser.add_argument("--data-dir", default="data/admet")
    parser.add_argument("--checkpoint-dir", default="results/admet_benchmark/checkpoints")
    parser.add_argument("--output-dir", default="results/admet_benchmark/test_results")
    parser.add_argument(
        "--featurizers",
        nargs="+",
        default=["ecfp4", "maccs", "rdkit200"],
        choices=["ecfp4", "maccs", "rdkit200"],
    )
    parser.add_argument(
        "--tdc-data-path",
        default=None,
        help="TDC raw data cache directory (default: data/admet/_tdc_raw)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# sklearn metrics
# ---------------------------------------------------------------------------

def evaluate_sklearn(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute full sklearn metric suite from hard-label predictions."""
    metrics: dict = {}
    try:
        metrics["auroc"] = roc_auc_score(y_true, y_pred)
    except Exception:
        metrics["auroc"] = float("nan")
    try:
        metrics["auprc"] = average_precision_score(y_true, y_pred)
    except Exception:
        metrics["auprc"] = float("nan")
    try:
        metrics["mcc"] = matthews_corrcoef(y_true, y_pred)
    except Exception:
        metrics["mcc"] = float("nan")
    try:
        metrics["f1"] = f1_score(y_true, y_pred, average="weighted")
    except Exception:
        metrics["f1"] = float("nan")
    try:
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
    except Exception:
        metrics["accuracy"] = float("nan")
    return metrics


# ---------------------------------------------------------------------------
# TDC official evaluator
# ---------------------------------------------------------------------------

def _load_tdc_group(tdc_data_path: str | None):
    """Lazily import and instantiate TDC admet_group."""
    try:
        from tdc.benchmark_group import admet_group
        path = tdc_data_path or "data/admet/_tdc_raw"
        return admet_group(path=path)
    except Exception as exc:
        log.warning(f"TDC evaluator not available: {exc}")
        return None


def evaluate_tdc(
    group,
    endpoint: str,
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> dict:
    """
    Run TDC's official ``group.evaluate()`` for one endpoint.

    TDC expects predictions as a dict ``{drug_id: prediction}``; the test
    split DataFrame from TDC has an 'Drug_ID' column we match against the
    order of the test CSV.  Since we re-read the CSV we rely on row-order
    alignment (TDC preserves original row order in test splits).

    Returns a dict like ``{'AUROC': 0.8721}`` or ``{'MAE': 0.12}`` depending
    on the endpoint's original task type.  For regression endpoints that were
    binarized, TDC still evaluates on the binary predictions here.
    """
    if group is None or endpoint not in _TDC_EVAL_NAMES:
        return {}
    try:
        benchmark = group.get(endpoint)
        test_df = benchmark["test"].copy()

        # TDC evaluate() needs a dict {Drug_ID: pred_value}
        id_col = "Drug_ID" if "Drug_ID" in test_df.columns else test_df.columns[0]
        drug_ids = test_df[id_col].tolist()

        # Align length: y_pred may be shorter if some SMILES failed featurization
        n = min(len(drug_ids), len(y_pred))
        pred_dict = {drug_ids[i]: float(y_pred[i]) for i in range(n)}

        result = group.evaluate(pred_dict, benchmark=endpoint)
        # Flatten nested dicts: TDC returns {endpoint: {metric: value}}
        if endpoint in result:
            return {f"tdc_{k}": v for k, v in result[endpoint].items()}
        return {f"tdc_{k}": v for k, v in result.items()}
    except Exception as exc:
        log.warning(f"  TDC evaluate failed for {endpoint}: {exc}")
        return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    log.info("=" * 60)
    log.info("QBioCode-ADMET: Test-Set Inference")
    log.info("=" * 60)
    log.info(f"Checkpoint dir : {args.checkpoint_dir}")
    log.info(f"Data dir       : {args.data_dir}")
    log.info(f"Featurizers    : {args.featurizers}")

    # Load the best-model index
    best_index = get_best_index(args.checkpoint_dir)
    if not best_index:
        log.error(
            f"No best_models.json found in {args.checkpoint_dir}. "
            "Run the QProfiler sweep first."
        )
        sys.exit(1)

    log.info(
        f"Found checkpoints for {len(best_index)} dataset keys "
        f"across {sum(len(v) for v in best_index.values())} (dataset, model) pairs"
    )

    # Load TDC group once (used for official evaluation)
    tdc_group = _load_tdc_group(args.tdc_data_path)
    if tdc_group is not None:
        log.info("TDC official evaluator loaded successfully.")
    else:
        log.warning("TDC evaluator unavailable — only sklearn metrics will be computed.")

    sklearn_rows = []
    tdc_rows = []

    for dataset_key, model_dict in sorted(best_index.items()):
        # dataset_key format: "{endpoint}_{featurizer}"
        # Parse by trying known featurizer suffixes first (most robust)
        feat = None
        endpoint = dataset_key
        for f in ("ecfp4", "maccs", "rdkit200"):
            if dataset_key.endswith(f"_{f}"):
                feat = f
                endpoint = dataset_key[: -(len(f) + 1)]
                break
        if feat is None:
            # Fallback: rsplit on last underscore
            parts = dataset_key.rsplit("_", 1)
            endpoint = parts[0] if len(parts) == 2 else dataset_key
            feat = parts[1] if len(parts) == 2 else "unknown"

        if feat not in args.featurizers:
            continue

        # Load test CSV (features + label)
        test_csv = os.path.join(args.data_dir, endpoint, feat, "test.csv")
        if not os.path.exists(test_csv):
            log.warning(f"  Test CSV not found: {test_csv} — skipping")
            continue

        test_df = pd.read_csv(test_csv)
        X_test = test_df.iloc[:, :-1].to_numpy()
        y_test = test_df.iloc[:, -1].to_numpy().astype(int)

        for model_name, ckpt_info in model_dict.items():
            val_f1 = ckpt_info.get("val_f1", float("nan"))

            try:
                y_pred = infer_from_best(
                    model_name, dataset_key, X_test, args.checkpoint_dir
                )

                # ── sklearn metrics ──────────────────────────────────────────
                sk_metrics = evaluate_sklearn(y_test, y_pred)
                sklearn_rows.append({
                    "endpoint": endpoint,
                    "featurizer": feat,
                    "model": model_name,
                    "val_f1_best_checkpoint": val_f1,
                    **sk_metrics,
                })

                # ── TDC official metrics ─────────────────────────────────────
                tdc_metrics = evaluate_tdc(tdc_group, endpoint, y_pred, y_test)
                if tdc_metrics:
                    tdc_rows.append({
                        "endpoint": endpoint,
                        "featurizer": feat,
                        "model": model_name,
                        "val_f1_best_checkpoint": val_f1,
                        **tdc_metrics,
                    })

                log.info(
                    f"  {endpoint:40s} / {feat:8s} / {model_name:10s} "
                    f"AUROC={sk_metrics.get('auroc', float('nan')):.4f}  "
                    f"AUPRC={sk_metrics.get('auprc', float('nan')):.4f}  "
                    f"MCC={sk_metrics.get('mcc', float('nan')):.4f}"
                    + (f"  TDC={list(tdc_metrics.values())[0]:.4f}" if tdc_metrics else "")
                )

            except Exception as exc:
                log.warning(f"  FAILED {dataset_key}/{model_name}: {exc}")
                sklearn_rows.append({
                    "endpoint": endpoint, "featurizer": feat, "model": model_name,
                    "val_f1_best_checkpoint": val_f1,
                    "auroc": float("nan"), "auprc": float("nan"),
                    "mcc": float("nan"), "f1": float("nan"), "accuracy": float("nan"),
                })

    # ── Save sklearn results ──────────────────────────────────────────────────
    results_df = pd.DataFrame(sklearn_rows)
    out_csv = os.path.join(args.output_dir, "test_results.csv")
    results_df.to_csv(out_csv, index=False)
    log.info(f"\nSklearn test results saved to {out_csv}")

    # ── Save TDC official results ─────────────────────────────────────────────
    if tdc_rows:
        tdc_df = pd.DataFrame(tdc_rows)
        tdc_csv = os.path.join(args.output_dir, "test_results_tdc.csv")
        tdc_df.to_csv(tdc_csv, index=False)
        log.info(f"TDC official results saved to {tdc_csv}")
    else:
        log.warning("No TDC evaluation results produced.")

    # ── Quick summary ─────────────────────────────────────────────────────────
    if not results_df.empty:
        summary = (
            results_df.groupby("model")[["auroc", "auprc", "mcc", "f1"]]
            .mean()
            .round(4)
            .sort_values("auroc", ascending=False)
        )
        log.info(f"\nOverall model summary (mean sklearn, all endpoints × featurizers):\n{summary}")
        summary.to_csv(os.path.join(args.output_dir, "test_summary_by_model.csv"))

    log.info("=" * 60)
    log.info("Test inference complete.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
