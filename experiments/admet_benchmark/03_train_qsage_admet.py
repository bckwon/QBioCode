#!/usr/bin/env python3
"""
03_train_qsage_admet.py
========================
Phase 5a experiment script: compile QProfiler results from all ADMET runs
and train a QSage meta-model.

Steps
-----
1. Walk ``results/admet_benchmark/`` and concatenate all
   ``ModelResults.csv`` and ``RawDataEvaluation.csv`` files.
2. Train a QSage (XGBoost-Optuna) on the compiled results.
3. Save the trained sage to ``results/admet_benchmark/sage/trained_sage.pkl``.
4. Generate SHAP feature importance plots.

Usage::

    .venv/bin/python experiments/admet_benchmark/03_train_qsage_admet.py

    # Custom paths
    .venv/bin/python experiments/admet_benchmark/03_train_qsage_admet.py \\
        --results-dir results/admet_benchmark \\
        --output-dir results/admet_benchmark/sage \\
        --sage-type xgboost_optuna --n-iter 200 --cv 10
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train QSage meta-model on ADMET QProfiler results."
    )
    parser.add_argument("--results-dir", default="results/admet_benchmark")
    parser.add_argument("--output-dir", default="results/admet_benchmark/sage")
    parser.add_argument(
        "--sage-type",
        default="xgboost_optuna",
        choices=["random_forest", "mlp", "xgboost_optuna"],
    )
    parser.add_argument("--n-iter", type=int, default=200)
    parser.add_argument("--cv", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def collect_results(results_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk results_dir and concatenate all ModelResults and RawDataEvaluation CSVs."""
    model_dfs, eval_dfs = [], []

    for root, dirs, files in os.walk(results_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            if fname == "ModelResults.csv":
                try:
                    model_dfs.append(pd.read_csv(fpath))
                except Exception as e:
                    log.warning(f"Could not read {fpath}: {e}")
            elif fname == "RawDataEvaluation.csv":
                try:
                    eval_dfs.append(pd.read_csv(fpath))
                except Exception as e:
                    log.warning(f"Could not read {fpath}: {e}")

    if not model_dfs:
        raise FileNotFoundError(
            f"No ModelResults.csv files found under {results_dir}. "
            "Run 02_run_qprofiler_admet.sh first."
        )

    model_df = pd.concat(model_dfs, ignore_index=True)
    eval_df = pd.concat(eval_dfs, ignore_index=True) if eval_dfs else pd.DataFrame()
    log.info(f"Collected {len(model_dfs)} ModelResults files → {len(model_df)} rows")
    log.info(f"Collected {len(eval_dfs)} RawDataEvaluation files → {len(eval_df)} rows")
    return model_df, eval_df


def run_shap_analysis(sage, output_dir: str) -> None:
    """Generate SHAP feature importance plots for the trained QSage."""
    try:
        import shap
        import matplotlib.pyplot as plt

        for metric in sage._available_metrics:
            for model_name in sage._available_models:
                result = sage._results_subsages.get(metric, {}).get(model_name)
                if result is None:
                    continue
                fitted = result["fit_model"]
                # XGBoost-Optuna returns raw model, not SearchCV wrapper
                explainer = shap.TreeExplainer(fitted)
                model_indices = sage._input_data_metadata[
                    sage._input_data_metadata["model"] == model_name
                ].index
                X = sage._input_data_features_only.loc[model_indices]
                X = X.replace([float("inf"), float("-inf")], float("nan")).fillna(0)

                shap_values = explainer.shap_values(X)
                plt.figure(figsize=(8, 5))
                shap.summary_plot(
                    shap_values, X,
                    plot_type="bar",
                    show=False,
                    max_display=15,
                )
                plt.title(f"SHAP — {model_name} / {metric}")
                plt.tight_layout()
                fname = os.path.join(output_dir, f"shap_{model_name}_{metric}.pdf")
                plt.savefig(fname, bbox_inches="tight")
                plt.close()
                log.info(f"  SHAP plot saved: {fname}")
    except Exception as exc:
        log.warning(f"SHAP analysis failed (non-fatal): {exc}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    log.info("=" * 60)
    log.info("QBioCode-ADMET: QSage Training")
    log.info("=" * 60)

    # ── Step 1: Collect results ───────────────────────────────────────────────
    model_df, eval_df = collect_results(args.results_dir)

    # Save compiled CSVs
    compiled_model = os.path.join(args.output_dir, "compiled_ModelResults.csv")
    compiled_eval = os.path.join(args.output_dir, "compiled_RawDataEvaluation.csv")
    model_df.to_csv(compiled_model, index=False)
    if not eval_df.empty:
        eval_df.to_csv(compiled_eval, index=False)
    log.info(f"Compiled results saved to {args.output_dir}")

    # ── Step 2: Prepare data for QSage ───────────────────────────────────────
    # QSage expects embeddings column; fill NaN for runs without embedding label
    if "embeddings" not in model_df.columns:
        model_df["embeddings"] = "none"
    model_df["embeddings"] = model_df["embeddings"].fillna("none").astype(str)

    # ── Step 3: Initialize and train QSage ───────────────────────────────────
    from qbiocode.apps.sage.sage import QuantumSage

    sage = QuantumSage(data_input=model_df)
    sage.set_seed(args.seed)

    log.info(f"Available models   : {sage._available_models}")
    log.info(f"Available metrics  : {sage._available_metrics}")
    log.info(f"Training sage type : {args.sage_type}")

    sage.train_sub_sages(
        test_size=args.test_size,
        sage_type=args.sage_type,
        n_iter=args.n_iter,
        cv=args.cv,
    )

    # ── Step 4: Save sage + summary ──────────────────────────────────────────
    import dill

    sage_pkl = os.path.join(args.output_dir, "trained_sage.pkl")
    with open(sage_pkl, "wb") as f:
        dill.dump(sage, f)
    log.info(f"Trained QSage saved to {sage_pkl}")

    # Save numeric summary
    summary_rows = []
    for metric in sage._available_metrics:
        for m in sage._available_models:
            if metric in sage._results_subsages and m in sage._results_subsages[metric]:
                r = sage._results_subsages[metric][m]
                summary_rows.append(
                    {"model": m, "metric": metric, "mae": r["mae"],
                     "mse": r["mse"], "rmse": r["rmse"], "r2": r["r2"]}
                )
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(args.output_dir, "sage_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    log.info(f"QSage summary:\n{summary_df.to_string(index=False)}")

    # ── Step 5: SHAP analysis ─────────────────────────────────────────────────
    if args.sage_type == "xgboost_optuna":
        log.info("\nRunning SHAP feature importance analysis...")
        run_shap_analysis(sage, args.output_dir)

    # ── Step 6: Plot results ──────────────────────────────────────────────────
    try:
        sage.plot_results(
            saveFile=os.path.join(args.output_dir, "sage_results.pdf")
        )
    except Exception as exc:
        log.warning(f"Plot generation failed (non-fatal): {exc}")

    log.info("\n" + "=" * 60)
    log.info("QSage training complete!")
    log.info(f"  Outputs in: {args.output_dir}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
