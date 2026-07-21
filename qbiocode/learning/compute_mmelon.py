"""
MMELON Baseline Classifier
===========================

Integrates the IBM Research MMELON foundation model
(``ibm-research/biomed.sm.mv-te-84m``) as a QBioCode classification baseline.

MMELON is a multi-view molecular transformer pretrained on large biomedical
corpora.  This module uses MMELON in **frozen-encoder mode**: SMILES strings
are tokenized and passed through the model to produce pooled CLS embeddings,
which are then fed to a lightweight sklearn Random Forest classification head.

This approach follows the same ``compute_*(X_train, X_test, y_train, y_test,
args, model, data_key, ...)`` signature as all other QBioCode learning modules,
making it a drop-in addition to the ``compute_ml_dict`` in ``model_run.py``.

Important
---------
``X_train`` and ``X_test`` passed to this function are expected to be
**SMILES string arrays** (dtype object / str), not numerical feature arrays.
The function embeds them internally and uses the resulting float32 vectors for
classification.  SMILES embeddings are cached to ``args['mmelon_cache_dir']``
to avoid recomputation across QProfiler iterations.

Usage
-----
>>> results = compute_mmelon(
...     smiles_train, smiles_test, y_train, y_test, args,
...     model='mmelon', data_key='hERG_ecfp4_10_1'
... )
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import Literal, Optional

import numpy as np

from qbiocode.evaluation.model_evaluation import modeleval

log = logging.getLogger(__name__)

#: Default HuggingFace model identifier for MMELON.
_MMELON_MODEL_ID = "ibm-research/biomed.sm.mv-te-84m"


def compute_mmelon(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    args: dict,
    model: str = "mmelon",
    data_key: str = "",
    head: Literal["rf", "lr", "svc"] = "rf",
    model_name: str = _MMELON_MODEL_ID,
    cache_dir: Optional[str] = None,
    batch_size: int = 32,
    max_length: int = 128,
    verbose: bool = False,
) -> "pd.DataFrame":
    """Compute MMELON frozen-embedding + sklearn-head classification.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train,) or (n_train, 1)
        SMILES strings for training samples.
    X_test : np.ndarray, shape (n_test,) or (n_test, 1)
        SMILES strings for test samples.
    y_train : np.ndarray
        Binary integer labels for training samples.
    y_test : np.ndarray
        Binary integer labels for test samples.
    args : dict
        QBioCode config dict (passed through to ``modeleval``).
    model : str
        Model identifier string stored in results CSV.  Default ``'mmelon'``.
    data_key : str
        Unique key for this dataset/split combination, used as cache filename.
    head : {'rf', 'lr', 'svc'}
        Classification head fitted on MMELON embeddings.  Default ``'rf'``.
    model_name : str
        HuggingFace model identifier.  Default ``ibm-research/biomed.sm.mv-te-84m``.
    cache_dir : str, optional
        Directory where embedding ``.npy`` caches are stored.  Falls back to
        ``args.get('mmelon_cache_dir', 'data/mmelon_cache')``.
    batch_size : int
        Number of SMILES to encode per forward pass.  Default 32.
    max_length : int
        Maximum tokenizer sequence length.  Default 128.
    verbose : bool
        If True, print progress information.  Default False.

    Returns
    -------
    pd.DataFrame
        QBioCode-compatible results DataFrame from ``modeleval()``.
    """
    import pandas as pd  # noqa: F401 (used by modeleval return type)

    beg_time = time.time()

    cache_dir = cache_dir or args.get("mmelon_cache_dir", "data/mmelon_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Flatten SMILES arrays (handle both 1-D and 2-D column inputs)
    smiles_train = _flatten_smiles(X_train)
    smiles_test = _flatten_smiles(X_test)

    # 1. Load / compute embeddings (cached)
    emb_train = _get_embeddings(
        smiles_train, model_name, cache_dir, data_key + "_train",
        batch_size, max_length, verbose
    )
    emb_test = _get_embeddings(
        smiles_test, model_name, cache_dir, data_key + "_test",
        batch_size, max_length, verbose
    )

    # 2. Fit classification head
    clf = _build_head(head, seed=args.get("seed", 42))
    clf.fit(emb_train, y_train)
    y_predicted = clf.predict(emb_test)

    # 3. Collect hyperparameters for logging
    model_params = {
        "model_name": model_name,
        "head": head,
        "head_params": clf.get_params(),
        "embedding_dim": emb_train.shape[1],
        "batch_size": batch_size,
        "max_length": max_length,
    }

    if verbose:
        log.info(
            f"MMELON [{head}] — train={len(smiles_train)}, test={len(smiles_test)}, "
            f"emb_dim={emb_train.shape[1]}, elapsed={time.time() - beg_time:.1f}s"
        )

    return modeleval(
        y_test, y_predicted, beg_time, model_params, args, model=model, verbose=verbose
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _flatten_smiles(X: np.ndarray) -> list:
    """Convert numpy array of SMILES strings to a plain Python list."""
    arr = np.asarray(X)
    if arr.ndim == 2:
        arr = arr[:, 0]
    return [str(s) for s in arr.tolist()]


def _cache_key(smiles_list: list, model_name: str, data_key: str) -> str:
    """Deterministic cache filename based on content + model hash."""
    content = model_name + data_key + "".join(smiles_list[:10])
    h = hashlib.md5(content.encode()).hexdigest()[:12]
    safe_key = data_key.replace("/", "_").replace("\\", "_")
    return f"mmelon_{safe_key}_{h}.npy"


def _get_embeddings(
    smiles_list: list,
    model_name: str,
    cache_dir: str,
    data_key: str,
    batch_size: int,
    max_length: int,
    verbose: bool,
) -> np.ndarray:
    """Return CLS-pooled MMELON embeddings, using disk cache when available."""
    fname = _cache_key(smiles_list, model_name, data_key)
    cache_path = os.path.join(cache_dir, fname)

    if os.path.exists(cache_path):
        log.info(f"  MMELON: loading cached embeddings from {cache_path}")
        return np.load(cache_path)

    log.info(f"  MMELON: computing embeddings for {len(smiles_list)} SMILES → {cache_path}")
    embeddings = _encode_smiles(smiles_list, model_name, batch_size, max_length, verbose)
    np.save(cache_path, embeddings)
    return embeddings


def _encode_smiles(
    smiles_list: list,
    model_name: str,
    batch_size: int,
    max_length: int,
    verbose: bool,
) -> np.ndarray:
    """Tokenize SMILES and extract CLS-pooled embeddings from MMELON."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"  MMELON: loading model '{model_name}' on {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name)
    encoder.eval()
    encoder.to(device)

    all_embeddings = []
    n_batches = (len(smiles_list) + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in range(n_batches):
            batch = smiles_list[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            if verbose and batch_idx % 10 == 0:
                log.info(f"    batch {batch_idx + 1}/{n_batches}")

            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = encoder(**inputs)
            # CLS token is the first token of last_hidden_state
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_emb)

    # Free GPU memory
    del encoder
    if device == "cuda":
        torch.cuda.empty_cache()

    return np.vstack(all_embeddings).astype(np.float32)


def _build_head(head: str, seed: int):
    """Instantiate the classification head."""
    if head == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    elif head == "lr":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=5000, random_state=seed, solver="saga")
    elif head == "svc":
        from sklearn.svm import SVC
        return SVC(kernel="rbf", probability=False, random_state=seed)
    raise ValueError(f"Unknown head '{head}'. Choose from: 'rf', 'lr', 'svc'.")
