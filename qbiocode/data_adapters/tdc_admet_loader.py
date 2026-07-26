"""
TDC ADMET Benchmark Loader
==========================

Adapts the Therapeutics Data Commons (TDC) ADMET Benchmark Group into
QBioCode's CSV-based data format.  Each endpoint is downloaded via the
canonical TDC API, optionally binarized (for regression tasks), and
written as ``train.csv``, ``valid.csv``, ``test.csv`` files where the
**last column is always the binary label** — matching QBioCode convention.

A ``metadata.json`` file is written at the root of ``data_dir`` recording
per-endpoint details: task type, binarization threshold, dataset sizes, and
the QML sample cap applied to the training split.

Usage
-----
>>> from qbiocode.data_adapters import TDCAdmetLoader
>>> loader = TDCAdmetLoader(data_dir='data/admet')
>>> loader.prepare_all()
>>> # Or a single endpoint:
>>> loader.prepare_endpoint('hERG')
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Endpoint registry
# ---------------------------------------------------------------------------

#: Canonical TDC ADMET endpoint names + metadata.
#: ``binarize``: whether the original task is regression and must be binarized.
#: ``threshold``: value used for binarization (label = 1 if value >= threshold).
#:               ``None`` means binary already; ``"median"`` uses training-set median.
ADMET_ENDPOINTS: Dict[str, Dict] = {
    # ── Absorption ──────────────────────────────────────────────────────────
    "Caco2_Wang": {
        "tdc_name": "Caco2_Wang",
        "category": "Absorption",
        "original_task": "regression",
        "binarize": True,
        "threshold": -5.15,           # log cm/s; > -5.15 → high permeability
        "label_meaning": "high_permeability",
    },
    "HIA_Hou": {
        "tdc_name": "HIA_Hou",
        "category": "Absorption",
        "original_task": "binary",
        "binarize": False,
        "threshold": None,
        "label_meaning": "absorbed",
    },
    "Pgp_Broccatelli": {
        "tdc_name": "Pgp_Broccatelli",
        "category": "Absorption",
        "original_task": "binary",
        "binarize": False,
        "threshold": None,
        "label_meaning": "pgp_inhibitor",
    },
    "Bioavailability_Ma": {
        "tdc_name": "Bioavailability_Ma",
        "category": "Absorption",
        "original_task": "binary",
        "binarize": False,
        "threshold": None,
        "label_meaning": "bioavailable",
    },
    # ── Distribution ────────────────────────────────────────────────────────
    "Lipophilicity_AstraZeneca": {
        "tdc_name": "Lipophilicity_AstraZeneca",
        "category": "Distribution",
        "original_task": "regression",
        "binarize": True,
        "threshold": 2.0,             # logD; > 2.0 → lipophilic
        "label_meaning": "lipophilic",
    },
    "Solubility_AqSolDB": {
        "tdc_name": "Solubility_AqSolDB",
        "category": "Distribution",
        "original_task": "regression",
        "binarize": True,
        "threshold": -3.0,            # log mol/L; > -3.0 → soluble
        "label_meaning": "soluble",
    },
    "BBB_Martins": {
        "tdc_name": "BBB_Martins",
        "category": "Distribution",
        "original_task": "binary",
        "binarize": False,
        "threshold": None,
        "label_meaning": "bbb_permeable",
    },
    "PPBR_AstraZeneca": {
        "tdc_name": "ppbr_az",
        "category": "Distribution",
        "original_task": "regression",
        "binarize": True,
        "threshold": 90.0,            # % bound; > 90 → highly bound
        "label_meaning": "highly_protein_bound",
    },
    "VDss_Lombardo": {
        "tdc_name": "VDss_Lombardo",
        "category": "Distribution",
        "original_task": "regression",
        "binarize": True,
        "threshold": 0.71,            # L/kg; > 0.71 → high volume of distribution
        "label_meaning": "high_vd",
    },
    # ── Metabolism ───────────────────────────────────────────────────────────
    "CYP2C19_Veith": {
        "tdc_name": "CYP2C19_Veith",
        "category": "Metabolism",
        "original_task": "binary",
        "binarize": False,
        "threshold": None,
        "label_meaning": "cyp2c19_inhibitor",
    },
    "CYP2D6_Veith": {
        "tdc_name": "CYP2D6_Veith",
        "category": "Metabolism",
        "original_task": "binary",
        "binarize": False,
        "threshold": None,
        "label_meaning": "cyp2d6_inhibitor",
    },
    "CYP3A4_Veith": {
        "tdc_name": "CYP3A4_Veith",
        "category": "Metabolism",
        "original_task": "binary",
        "binarize": False,
        "threshold": None,
        "label_meaning": "cyp3a4_inhibitor",
    },
    "CYP1A2_Veith": {
        "tdc_name": "CYP1A2_Veith",
        "category": "Metabolism",
        "original_task": "binary",
        "binarize": False,
        "threshold": None,
        "label_meaning": "cyp1a2_inhibitor",
    },
    "CYP2C9_Veith": {
        "tdc_name": "CYP2C9_Veith",
        "category": "Metabolism",
        "original_task": "binary",
        "binarize": False,
        "threshold": None,
        "label_meaning": "cyp2c9_inhibitor",
    },
    "CYP2C9_Substrate_CarbonMangels": {
        "tdc_name": "CYP2C9_Substrate_CarbonMangels",
        "category": "Metabolism",
        "original_task": "binary",
        "binarize": False,
        "threshold": None,
        "label_meaning": "cyp2c9_substrate",
    },
    "CYP2D6_Substrate_CarbonMangels": {
        "tdc_name": "CYP2D6_Substrate_CarbonMangels",
        "category": "Metabolism",
        "original_task": "binary",
        "binarize": False,
        "threshold": None,
        "label_meaning": "cyp2d6_substrate",
    },
    "CYP3A4_Substrate_CarbonMangels": {
        "tdc_name": "CYP3A4_Substrate_CarbonMangels",
        "category": "Metabolism",
        "original_task": "binary",
        "binarize": False,
        "threshold": None,
        "label_meaning": "cyp3a4_substrate",
    },
    # ── Excretion ────────────────────────────────────────────────────────────
    "Half_Life_Obach": {
        "tdc_name": "Half_Life_Obach",
        "category": "Excretion",
        "original_task": "regression",
        "binarize": True,
        "threshold": "median",        # median-split of training set
        "label_meaning": "long_half_life",
    },
    "Clearance_Hepatocyte_AZ": {
        "tdc_name": "Clearance_Hepatocyte_AZ",
        "category": "Excretion",
        "original_task": "regression",
        "binarize": True,
        "threshold": "median",
        "label_meaning": "high_clearance",
    },
    # ── Toxicity ─────────────────────────────────────────────────────────────
    "hERG": {
        "tdc_name": "hERG",
        "category": "Toxicity",
        "original_task": "binary",
        "binarize": False,
        "threshold": None,
        "label_meaning": "herg_blocker",
    },
    "AMES": {
        "tdc_name": "AMES",
        "category": "Toxicity",
        "original_task": "binary",
        "binarize": False,
        "threshold": None,
        "label_meaning": "mutagenic",
    },
    "DILI": {
        "tdc_name": "DILI",
        "category": "Toxicity",
        "original_task": "binary",
        "binarize": False,
        "threshold": None,
        "label_meaning": "dili_positive",
    },
}


class TDCAdmetLoader:
    """Download and preprocess TDC ADMET benchmark endpoints for QBioCode.

    This class wraps the TDC ``admet_group`` API to download canonical
    train/valid/test splits for all 22 ADMET endpoints, applies binarization
    for regression tasks, and writes QBioCode-compatible CSV files where the
    last column is always the binary label.

    Parameters
    ----------
    data_dir : str
        Root directory where processed data will be written.
        Subdirectories are created as ``{data_dir}/{endpoint}/{featurizer}/``.
    qml_train_cap : int
        Maximum number of training samples used when creating the QML-capped
        training split. Classical models use the full training set.
        Default is 300.
    seed : int
        Random seed for stratified subsampling of the QML training cap.
        Default is 42.
    tdc_data_path : str, optional
        Path where TDC will cache raw downloaded data. If None, uses TDC
        default (``~/data`` or ``./data``).

    Examples
    --------
    >>> loader = TDCAdmetLoader(data_dir='data/admet')
    >>> loader.prepare_all()
    >>> loader.prepare_endpoint('hERG')   # single endpoint
    """

    def __init__(
        self,
        data_dir: str = "data/admet",
        qml_train_cap: int = 300,
        seed: int = 42,
        tdc_data_path: Optional[str] = None,
    ) -> None:
        self.data_dir = data_dir
        self.qml_train_cap = qml_train_cap
        self.seed = seed
        self.tdc_data_path = tdc_data_path or os.path.join(data_dir, "_tdc_raw")
        self._metadata: Dict = {}
        # TDC admet_group requires the path directory to already exist
        os.makedirs(self.tdc_data_path, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare_all(self, endpoints: Optional[list] = None) -> Dict:
        """Prepare all (or a subset of) ADMET endpoints.

        Parameters
        ----------
        endpoints : list of str, optional
            Subset of endpoint names from ``ADMET_ENDPOINTS``.  If None,
            all 22 endpoints are prepared.

        Returns
        -------
        dict
            Metadata dictionary mirroring what is written to ``metadata.json``.
        """
        targets = endpoints or list(ADMET_ENDPOINTS.keys())
        log.info(f"Preparing {len(targets)} ADMET endpoints → {self.data_dir}")
        for name in targets:
            try:
                self.prepare_endpoint(name)
            except Exception as exc:
                log.error(f"Failed to prepare {name}: {exc}")
        self._write_metadata()
        return self._metadata

    def prepare_endpoint(self, endpoint_name: str) -> Dict:
        """Prepare a single ADMET endpoint.

        Downloads raw data via TDC, applies binarization if required,
        creates the QML-capped training split, and writes CSV files.

        Parameters
        ----------
        endpoint_name : str
            Key from ``ADMET_ENDPOINTS``.

        Returns
        -------
        dict
            Metadata entry for this endpoint.
        """
        if endpoint_name not in ADMET_ENDPOINTS:
            raise ValueError(
                f"Unknown endpoint '{endpoint_name}'. "
                f"Valid names: {list(ADMET_ENDPOINTS.keys())}"
            )
        cfg = ADMET_ENDPOINTS[endpoint_name]
        log.info(f"Preparing endpoint: {endpoint_name}")

        # 1. Download via TDC
        train_df, valid_df, test_df = self._download_tdc(cfg["tdc_name"])

        # 2. Binarize if needed
        threshold = cfg["threshold"]
        if cfg["binarize"]:
            if threshold == "median":
                threshold = float(train_df["Y"].median())
                log.info(f"  {endpoint_name}: median threshold = {threshold:.4f}")
            train_df, valid_df, test_df = self._binarize_splits(
                train_df, valid_df, test_df, threshold
            )

        # 3. Ensure labels are int
        for df in (train_df, valid_df, test_df):
            df["Y"] = df["Y"].astype(int)

        # 4. Build QML-capped training split (SMILES + label only; features added later)
        train_qml_df = self._cap_qml_split(train_df)

        # 5. Write raw SMILES splits (featurizer writes feature CSVs on top of these)
        endpoint_dir = os.path.join(self.data_dir, endpoint_name)
        os.makedirs(endpoint_dir, exist_ok=True)
        train_df.to_csv(os.path.join(endpoint_dir, "train_smiles.csv"), index=False)
        valid_df.to_csv(os.path.join(endpoint_dir, "valid_smiles.csv"), index=False)
        test_df.to_csv(os.path.join(endpoint_dir, "test_smiles.csv"), index=False)
        train_qml_df.to_csv(os.path.join(endpoint_dir, "train_qml_smiles.csv"), index=False)

        # 6. Record metadata
        actual_threshold = threshold if cfg["binarize"] else None
        meta = {
            **cfg,
            "n_train": len(train_df),
            "n_valid": len(valid_df),
            "n_test": len(test_df),
            "n_train_qml": len(train_qml_df),
            "class_balance_train": float(train_df["Y"].mean()),
            "actual_threshold": actual_threshold,
        }
        self._metadata[endpoint_name] = meta
        log.info(
            f"  {endpoint_name}: train={len(train_df)}, valid={len(valid_df)}, "
            f"test={len(test_df)}, qml_cap={len(train_qml_df)}, "
            f"pos_rate={meta['class_balance_train']:.3f}"
        )
        return meta

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _download_tdc(self, tdc_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Download TDC canonical splits and return (train, valid, test) DataFrames."""
        from tdc.benchmark_group import admet_group  # lazy import

        group = admet_group(path=self.tdc_data_path)
        benchmark = group.get(tdc_name)
        # TDC returns {'train_val': df, 'test': df}; we further split train_val
        train_val = benchmark["train_val"]
        test_df = benchmark["test"]
        train_df, valid_df = group.get_train_valid_split(
            benchmark=tdc_name, split_type="default", seed=self.seed
        )
        return train_df, valid_df, test_df

    @staticmethod
    def _binarize_splits(
        train: pd.DataFrame,
        valid: pd.DataFrame,
        test: pd.DataFrame,
        threshold: float,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Binarize 'Y' column: 1 if Y >= threshold, else 0."""
        for df in (train, valid, test):
            df["Y"] = (df["Y"] >= threshold).astype(int)
        return train, valid, test

    def _cap_qml_split(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Return a stratified subsample of training data for QML models."""
        if len(train_df) <= self.qml_train_cap:
            return train_df.copy()
        # Stratified sample to preserve class balance
        classes = train_df["Y"].unique()
        n_per_class = self.qml_train_cap // len(classes)
        parts = []
        rng = np.random.RandomState(self.seed)
        for cls in sorted(classes):
            cls_df = train_df[train_df["Y"] == cls]
            n = min(n_per_class, len(cls_df))
            parts.append(cls_df.sample(n=n, random_state=rng.randint(0, 10000)))
        return pd.concat(parts).sample(frac=1, random_state=self.seed).reset_index(drop=True)

    def _write_metadata(self) -> None:
        """Persist metadata dictionary to ``{data_dir}/metadata.json``."""
        os.makedirs(self.data_dir, exist_ok=True)
        meta_path = os.path.join(self.data_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(self._metadata, f, indent=2)
        log.info(f"Metadata written to {meta_path}")
