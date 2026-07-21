# ====== Base class imports ======

import time
from typing import Any, Literal, Optional
import pandas as pd

# ====== Scikit-learn imports ======

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
)

from qbiocode.utils.helper_fn import print_results


def modeleval(
    y_test,
    y_predicted,
    beg_time,
    params,
    args,
    model: str,
    verbose: bool = True,
    average: str = "weighted",
    fitted_model: Optional[Any] = None,
    checkpoint_dir: Optional[str] = None,
    dataset_name: Optional[str] = None,
    split_id: int = 0,
):
    """Evaluate model performance and optionally persist the fitted model.

    Computes accuracy, F1, AUC (and optionally AUPRC, MCC for ADMET tasks)
    and returns a QBioCode-standard results DataFrame.

    When ``fitted_model`` and ``checkpoint_dir`` are both provided, the
    fitted model is serialized via :func:`qbiocode.utils.model_checkpoint.save_checkpoint`
    and the best-model index is updated if this split's F1 is a new best.

    Args:
        y_test (array-like): True labels for the test set.
        y_predicted (array-like): Predicted labels by the model.
        beg_time (float): Start time for measuring execution time.
        params (dict): Model parameters used during training.
        args (dict): Additional arguments, including ``grid_search`` flag.
            Set ``args['admet_metrics'] = True`` to also compute AUPRC and MCC.
        model (str): Name of the model being evaluated.
        verbose (bool): If True, prints the evaluation results.
        average (str): Averaging strategy for F1 score.  Default ``'weighted'``.
        fitted_model (object, optional): Fitted model object to checkpoint.
            Only used when ``checkpoint_dir`` is also provided.
        checkpoint_dir (str, optional): Root directory for model checkpoints.
            When set together with ``fitted_model``, calls
            :func:`~qbiocode.utils.model_checkpoint.save_checkpoint`.
        dataset_name (str, optional): Dataset identifier used as checkpoint key.
            Defaults to ``model`` if not provided.
        split_id (int): Train/valid split iteration, used in checkpoint filename.

    Returns:
        pd.DataFrame: Results DataFrame with accuracy, F1, AUC, and model params.
            Also contains ``auprc`` and ``mcc`` columns when
            ``args.get('admet_metrics')`` is True.
    """
    # ── Core metrics ────────────────────────────────────────────────────────
    auc = roc_auc_score(y_test, y_predicted)
    accuracy = accuracy_score(y_test, y_predicted, normalize=True)
    f1 = f1_score(y_test, y_predicted, average=average)
    compile_time = time.time() - beg_time

    # ── ADMET extended metrics (opt-in) ─────────────────────────────────────
    admet_extras: dict = {}
    if args.get("admet_metrics", False):
        try:
            admet_extras["auprc"] = average_precision_score(y_test, y_predicted)
        except Exception:
            admet_extras["auprc"] = float("nan")
        try:
            admet_extras["mcc"] = matthews_corrcoef(y_test, y_predicted)
        except Exception:
            admet_extras["mcc"] = float("nan")

    if verbose:
        print_results(model, accuracy, f1, compile_time, params)

    # ── Optional checkpoint save ─────────────────────────────────────────────
    if fitted_model is not None and checkpoint_dir is not None:
        try:
            from qbiocode.utils.model_checkpoint import save_checkpoint  # lazy import
            ds_key = dataset_name or model
            save_checkpoint(
                fitted_model, model, ds_key, split_id, float(f1), checkpoint_dir
            )
        except Exception as exc:
            # Non-fatal: log and continue
            import logging
            logging.getLogger(__name__).warning(
                f"Checkpoint save failed for {model}/{dataset_name}: {exc}"
            )

    # ── Build results dict ───────────────────────────────────────────────────
    if args.get("grid_search", False):
        result_dict = {
            "model": model,
            "accuracy": accuracy,
            "f1_score": f1,
            "time": compile_time,
            "auc": auc,
            "BestParams_GridSearch": params,
            **admet_extras,
        }
    else:
        result_dict = {
            "model": model,
            "accuracy": accuracy,
            "f1_score": f1,
            "time": compile_time,
            "auc": auc,
            "Model_Parameters": params,
            **admet_extras,
        }

    return pd.DataFrame(
        {
            "y_test_" + model: [y_test],
            "y_predicted_" + model: [y_predicted],
            "results_" + model: [result_dict],
        }
    )


def evaluation_metrics(predictions, y_test, metrics=["accuracy", "brier"], save=False):
    """
    Calculate evaluation metrics for classification predictions.

    Computes specified metrics for model predictions. Supports accuracy, Brier score,
    F1 score, precision, recall, and AUC-ROC. The Brier score measures the mean
    squared difference between predicted probabilities and actual outcomes, providing
    a measure of calibration quality.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted probabilities, shape (n_samples, n_classes)
    y_test : np.ndarray
        True labels, shape (n_samples,)
    metrics : list of str, optional
        List of metrics to compute. Options: 'accuracy', 'brier', 'f1',
        'precision', 'recall', 'auc' (default: ['accuracy', 'brier'])
    save : bool, optional
        Whether to save results (reserved for future use, default: False)

    Returns
    -------
    tuple or dict
        If metrics=['accuracy', 'brier'] (default): returns (accuracy, brier_score)
        Otherwise: returns dict with requested metrics as keys

    Examples
    --------
    >>> import numpy as np
    >>> from qbiocode.evaluation import evaluation_metrics
    >>>
    >>> # Binary classification example - default metrics
    >>> predictions = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
    >>> y_test = np.array([0, 1, 0])
    >>> accuracy, brier = evaluation_metrics(predictions, y_test)
    >>> print(f"Accuracy: {accuracy:.2f}, Brier Score: {brier:.3f}")
    Accuracy: 1.00, Brier Score: 0.060

    >>> # Multiple metrics
    >>> results = evaluation_metrics(predictions, y_test,
    ...                              metrics=['accuracy', 'brier', 'f1', 'auc'])
    >>> print(results)
    {'accuracy': 1.0, 'brier': 0.06, 'f1': 1.0, 'auc': 1.0}

    Notes
    -----
    - For binary classification, Brier score is computed using the probability
      of the positive class
    - For multi-class classification, the average Brier score across all classes
      is returned
    - F1, precision, and recall use weighted averaging for multi-class
    - AUC uses one-vs-rest for multi-class
    - Lower Brier scores indicate better calibrated probability predictions

    References
    ----------
    Brier, G. W. (1950). "Verification of forecasts expressed in terms of probability".
    Monthly Weather Review, 78(1), 1-3.
    """
    import numpy as np
    from sklearn.metrics import (
        brier_score_loss,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    # Get predicted classes
    y_pred = np.argmax(predictions, axis=1)

    results = {}

    # Calculate requested metrics
    if "accuracy" in metrics:
        results["accuracy"] = accuracy_score(y_test, y_pred)

    if "brier" in metrics:
        if predictions.shape[1] == 2:
            # Binary classification: use probability of positive class
            results["brier"] = brier_score_loss(y_test, predictions[:, 1])
        else:
            # Multi-class: use average Brier score across all classes
            results["brier"] = np.mean(
                [
                    brier_score_loss(y_test == i, predictions[:, i])
                    for i in range(predictions.shape[1])
                ]
            )

    if "f1" in metrics:
        results["f1"] = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    if "precision" in metrics:
        results["precision"] = precision_score(y_test, y_pred, average="weighted", zero_division=0)

    if "recall" in metrics:
        results["recall"] = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    if "auc" in metrics:
        try:
            if predictions.shape[1] == 2:
                # Binary classification
                results["auc"] = roc_auc_score(y_test, predictions[:, 1])
            else:
                # Multi-class: one-vs-rest
                results["auc"] = roc_auc_score(
                    y_test, predictions, multi_class="ovr", average="weighted"
                )
        except ValueError:
            # Handle cases where AUC cannot be computed (e.g., single class in y_test)
            results["auc"] = np.nan

    # For backward compatibility: return tuple if default metrics
    if metrics == ["accuracy", "brier"]:
        return results["accuracy"], results["brier"]

    return results
