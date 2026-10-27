import json
import logging
import os

import numpy as np
import pandas as pd

from joblib import Parallel, delayed

log = logging.getLogger(__name__)

current_dir = os.getcwd()

# Quantum models require one qubit per input feature.
# Statevectors beyond 2**26 complex128 elements (~1 GB) are impractical.
_QML_MAX_QUBITS = 26


def _make_nan_result(y_test, model_name: str, args: dict) -> pd.DataFrame:
    """Return a single-row NaN DataFrame in the same shape model_run returns."""
    import time
    import numpy as _np
    from qbiocode.evaluation.model_evaluation import modeleval
    beg = time.time()
    return modeleval(
        y_test, _np.zeros(len(y_test), dtype=int),
        beg, {}, args, model=model_name, verbose=False,
    )


def model_run(X_train, X_test, y_train, y_test, data_key, args):
    """Run all configured ML models in parallel and return results.

    Args:
        X_train (np.ndarray): Training features.
        X_test (np.ndarray): Test features.
        y_train (np.ndarray): Training labels.
        y_test (np.ndarray): Test labels.
        data_key (str): Identifier for the current dataset/embedding/iteration.
        args (dict): Configuration dict with keys:
            - model (list): Model names to run.
            - n_jobs (int): Parallel job count.
            - grid_search (bool): Run optimized (_opt) model variants.
            - cross_validation: CV strategy for grid search.
            - gridsearch_<model>_args (dict): Grid search kwargs per model.
            - <model>_args (dict): Model kwargs.

    Returns:
        dict: Pivoted results keyed by column name, compatible with
            ``pd.DataFrame.from_dict``.
    """
    from qbiocode.learning.compute_dt import compute_dt, compute_dt_opt
    from qbiocode.learning.compute_lr import compute_lr, compute_lr_opt
    from qbiocode.learning.compute_rf import compute_rf, compute_rf_opt
    from qbiocode.learning.compute_mlp import compute_mlp, compute_mlp_opt
    from qbiocode.learning.compute_xgb import compute_xgb, compute_xgb_opt
    from qbiocode.learning.compute_pqk import compute_pqk
    from qbiocode.learning.compute_qpl import compute_qpl
    from qbiocode.learning.compute_qnn import compute_qnn
    from qbiocode.learning.compute_qsvc import compute_qsvc
    from qbiocode.learning.compute_nb import compute_nb, compute_nb_opt
    from qbiocode.learning.compute_svc import compute_svc, compute_svc_opt
    from qbiocode.learning.compute_vqc import compute_vqc
    from qbiocode.learning.compute_mmelon import compute_mmelon
    from qbiocode.learning.compute_qensemble import compute_qensemble

    compute_ml_dict = {
        "svc_opt": compute_svc_opt,
        "svc": compute_svc,
        "dt_opt": compute_dt_opt,
        "dt": compute_dt,
        "lr_opt": compute_lr_opt,
        "lr": compute_lr,
        "nb_opt": compute_nb_opt,
        "nb": compute_nb,
        "rf_opt": compute_rf_opt,
        "rf": compute_rf,
        "xgb_opt": compute_xgb_opt,
        "xgb": compute_xgb,
        "mlp_opt": compute_mlp_opt,
        "mlp": compute_mlp,
        "qsvc": compute_qsvc,
        "vqc": compute_vqc,
        "qnn": compute_qnn,
        "pqk": compute_pqk,
        "qpl": compute_qpl,
        "mmelon": compute_mmelon,
        "qensemble": compute_qensemble,
    }

    quantum_models = {"qsvc", "qnn", "vqc", "pqk", "qpl", "mmelon", "qensemble"}

    max_qubits = int(args.get("max_qubits", _QML_MAX_QUBITS))
    n_features = X_train.shape[1]
    active_models = list(args["model"])
    skipped_qml: list[str] = []
    if n_features > max_qubits:
        skipped_qml = [m for m in active_models if m in quantum_models]
        if skipped_qml:
            log.warning(
                "QML qubit budget exceeded: n_features=%d > max_qubits=%d. "
                "Skipping quantum models: %s. "
                "Use an embedding with n_components <= max_qubits.",
                n_features, max_qubits, skipped_qml,
            )
            active_models = [m for m in active_models if m not in quantum_models]

    n_jobs = len(args["model"])
    if "n_jobs" in args.keys():
        n_jobs = min(args["n_jobs"], len(args["model"]))

    grid_search = args.get("grid_search", False)

    if grid_search:
        quantum_in_models = [m for m in args["model"] if m in quantum_models]
        if quantum_in_models:
            print(f"WARNING: Grid search enabled with quantum models {quantum_in_models}.")
            print("Quantum models do not support grid search. Use generate_qml_experiment_configs")
            print("to create per-hyperparameter config files and run QProfiler separately.")

    skipped_results = [_make_nan_result(y_test, m, args) for m in skipped_qml]

    if grid_search:
        results = []
        for method in active_models:
            if method in quantum_models:
                result = delayed(compute_ml_dict[method])(
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    args,
                    model=method,
                    data_key=data_key,
                    **args.get(method + "_args", {}),
                    verbose=False,
                )
            else:
                result = delayed(compute_ml_dict[method + "_opt"])(
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    args,
                    model=method + "_opt",
                    cv=args["cross_validation"],
                    **args.get("gridsearch_" + method + "_args", {}),
                    verbose=False,
                )
            results.append(result)
        results = Parallel(n_jobs=n_jobs)(results)
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_ml_dict[method])(
                X_train,
                X_test,
                y_train,
                y_test,
                args,
                model=method,
                data_key=data_key,
                **args[method + "_args"],
                verbose=False,
            )
            for method in active_models
        )

    results = results + skipped_results

    model_total_result = pd.melt(pd.concat(results)).dropna()  # type: ignore
    model_total_result["i"] = 0
    model_total_result = model_total_result.pivot(columns="variable", values="value", index="i")
    return model_total_result.to_dict()
