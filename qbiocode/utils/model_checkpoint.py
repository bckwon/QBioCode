"""
Model Checkpoint Utilities
===========================

Provides save / load / infer functions for fitted ML model objects produced
by QBioCode's ``compute_*`` learning functions.

Current QBioCode only records scalar metrics (accuracy, F1, AUC) to CSV.
This module adds persistent model storage so that:

1. Every fitted model is serialized to disk after evaluation.
2. A ``best_models.json`` index tracks the best checkpoint per
   ``(dataset, model_name)`` pair by validation F1-score.
3. Any downstream script can call ``infer_from_best()`` to load the
   best checkpoint and run inference — no retraining required.
4. Checkpoints can be served as production predictors in downstream systems.

Serialization uses ``dill`` (already a QBioCode dependency) which correctly
handles Qiskit ML objects, closures, and lambda functions that standard
``pickle`` cannot serialize.

Checkpoint directory layout
---------------------------
::

    {checkpoint_dir}/
    ├── best_models.json          ← index: {dataset: {model: {path, val_f1, split_id}}}
    └── {dataset}/
        ├── {model}_split{n}_f1{score:.4f}.pkl
        └── ...

Usage
-----
>>> from qbiocode.utils.model_checkpoint import save_checkpoint, infer_from_best
>>> # Save after fitting:
>>> save_checkpoint(fitted_model, 'rf', 'hERG_ecfp4', split_id=1,
...                 val_f1=0.8934, checkpoint_dir='results/admet/checkpoints')
>>> # Infer using best saved model:
>>> y_pred = infer_from_best('rf', 'hERG_ecfp4', X_test,
...                          checkpoint_dir='results/admet/checkpoints')
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import numpy as np

log = logging.getLogger(__name__)

_INDEX_FILE = "best_models.json"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_checkpoint(
    model_obj: Any,
    model_name: str,
    dataset_name: str,
    split_id: int,
    val_f1: float,
    checkpoint_dir: str,
) -> str:
    """Serialize a fitted model and update the best-model index.

    If ``val_f1`` is higher than the currently recorded best for this
    ``(dataset_name, model_name)`` pair, the index is updated.

    Parameters
    ----------
    model_obj : any fitted sklearn / Qiskit ML model
        The fitted model object to persist.
    model_name : str
        Short model identifier, e.g. ``'rf'``, ``'qsvc'``, ``'mmelon'``.
    dataset_name : str
        Unique dataset identifier, e.g. ``'hERG_ecfp4'``.
    split_id : int
        Train/valid split iteration number.
    val_f1 : float
        Validation F1-score achieved by this model on this split.
    checkpoint_dir : str
        Root directory for checkpoints.

    Returns
    -------
    str
        Path to the saved checkpoint file.
    """
    import dill  # lazy import — already in requirements-base.txt

    model_dir = os.path.join(checkpoint_dir, _sanitize(dataset_name))
    os.makedirs(model_dir, exist_ok=True)

    filename = f"{_sanitize(model_name)}_split{split_id}_f1{val_f1:.4f}.pkl"
    ckpt_path = os.path.join(model_dir, filename)

    # Atomic write: serialize to tmp file then rename
    tmp_path = ckpt_path + ".tmp"
    with open(tmp_path, "wb") as f:
        dill.dump(model_obj, f)
    os.replace(tmp_path, ckpt_path)

    log.debug(f"Saved checkpoint: {ckpt_path}")

    # Update best-model index
    _update_index(checkpoint_dir, dataset_name, model_name, ckpt_path, val_f1, split_id)

    return ckpt_path


def load_best_checkpoint(
    model_name: str,
    dataset_name: str,
    checkpoint_dir: str,
) -> Any:
    """Load the best fitted model for a given (dataset, model) pair.

    Reads the ``best_models.json`` index to find the checkpoint with the
    highest validation F1, then deserializes and returns it.

    Parameters
    ----------
    model_name : str
        Model identifier, e.g. ``'rf'``.
    dataset_name : str
        Dataset identifier, e.g. ``'hERG_ecfp4'``.
    checkpoint_dir : str
        Root checkpoint directory.

    Returns
    -------
    Fitted model object (type depends on what was saved).

    Raises
    ------
    FileNotFoundError
        If no checkpoint exists for this (dataset, model) pair.
    """
    import dill

    index = _load_index(checkpoint_dir)
    try:
        entry = index[dataset_name][model_name]
    except KeyError:
        raise FileNotFoundError(
            f"No checkpoint found for dataset='{dataset_name}', model='{model_name}' "
            f"in {os.path.join(checkpoint_dir, _INDEX_FILE)}"
        )

    ckpt_path = entry["path"]
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file missing: {ckpt_path}")

    with open(ckpt_path, "rb") as f:
        model_obj = dill.load(f)

    log.info(
        f"Loaded best checkpoint: {ckpt_path} "
        f"(val_f1={entry['val_f1']:.4f}, split={entry['split_id']})"
    )
    return model_obj


def infer_from_best(
    model_name: str,
    dataset_name: str,
    X_test: np.ndarray,
    checkpoint_dir: str,
) -> np.ndarray:
    """Load the best checkpoint and return predictions on ``X_test``.

    Parameters
    ----------
    model_name : str
    dataset_name : str
    X_test : np.ndarray
        Feature matrix for the test set.
    checkpoint_dir : str

    Returns
    -------
    np.ndarray
        Predicted class labels.
    """
    model_obj = load_best_checkpoint(model_name, dataset_name, checkpoint_dir)
    return model_obj.predict(X_test)


def get_best_index(checkpoint_dir: str) -> dict:
    """Return the full best-model index as a Python dict.

    Parameters
    ----------
    checkpoint_dir : str

    Returns
    -------
    dict
        ``{dataset_name: {model_name: {path, val_f1, split_id}}}``
    """
    return _load_index(checkpoint_dir)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sanitize(name: str) -> str:
    """Replace characters that are invalid in directory/file names."""
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_")


def _index_path(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, _INDEX_FILE)


def _load_index(checkpoint_dir: str) -> dict:
    path = _index_path(checkpoint_dir)
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def _save_index(checkpoint_dir: str, index: dict) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    tmp = _index_path(checkpoint_dir) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(index, f, indent=2)
    os.replace(tmp, _index_path(checkpoint_dir))


def _update_index(
    checkpoint_dir: str,
    dataset_name: str,
    model_name: str,
    ckpt_path: str,
    val_f1: float,
    split_id: int,
) -> None:
    """Update best-model index if val_f1 improves over the current best."""
    index = _load_index(checkpoint_dir)
    current_best = index.get(dataset_name, {}).get(model_name, {}).get("val_f1", -1.0)

    if val_f1 > current_best:
        index.setdefault(dataset_name, {})[model_name] = {
            "path": ckpt_path,
            "val_f1": val_f1,
            "split_id": split_id,
        }
        _save_index(checkpoint_dir, index)
        log.debug(
            f"Updated best checkpoint: {dataset_name}/{model_name} "
            f"f1={val_f1:.4f} (was {current_best:.4f})"
        )
