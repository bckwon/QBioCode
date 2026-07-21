"""
Molecular Featurizers for ADMET Data
=====================================

Converts SMILES strings into numerical feature matrices using three
molecular fingerprinting / descriptor schemes:

- **ECFP4**: 2048-bit Morgan circular fingerprints (radius=2)
- **MACCS**: 167-bit MACCS key fingerprints
- **RDKit200**: 200 RDKit 2D physicochemical descriptors (z-score standardized)

For each endpoint and featurizer, this module writes QBioCode-compatible CSV
files where features occupy all columns except the last, which is the binary
label ``Y``.  Failed SMILES are dropped and logged.

Usage
-----
>>> from qbiocode.data_adapters import MolecularFeaturizer, FEATURIZERS
>>> feat = MolecularFeaturizer(featurizer='ecfp4')
>>> feat.featurize_endpoint('data/admet/hERG')
>>> # Or featurize all at once:
>>> feat.featurize_all_endpoints('data/admet')
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

#: Supported featurizer names.
FEATURIZERS: List[str] = ["ecfp4", "maccs", "rdkit200"]

#: Split filenames that the TDCAdmetLoader writes.
_SPLIT_FILES = {
    "train": "train_smiles.csv",
    "train_qml": "train_qml_smiles.csv",
    "valid": "valid_smiles.csv",
    "test": "test_smiles.csv",
}


class MolecularFeaturizer:
    """Convert SMILES to numerical features and write QBioCode CSV files.

    Parameters
    ----------
    featurizer : str
        One of ``'ecfp4'``, ``'maccs'``, ``'rdkit200'``.
    ecfp4_radius : int
        Morgan fingerprint radius (default 2 → ECFP4).
    ecfp4_nbits : int
        Bit-vector length for Morgan fingerprints (default 2048).
    rdkit_n_components : int
        Number of RDKit descriptors to keep after removing zero-variance ones
        (default 200; actual count may be slightly lower if some descriptors
        cannot be computed for the dataset).

    Examples
    --------
    >>> feat = MolecularFeaturizer('ecfp4')
    >>> feat.featurize_endpoint('data/admet/hERG')
    >>> feat.featurize_all_endpoints('data/admet')
    """

    def __init__(
        self,
        featurizer: str = "ecfp4",
        ecfp4_radius: int = 2,
        ecfp4_nbits: int = 2048,
        rdkit_n_components: int = 200,
    ) -> None:
        if featurizer not in FEATURIZERS:
            raise ValueError(f"featurizer must be one of {FEATURIZERS}, got '{featurizer}'")
        self.featurizer = featurizer
        self.ecfp4_radius = ecfp4_radius
        self.ecfp4_nbits = ecfp4_nbits
        self.rdkit_n_components = rdkit_n_components
        # Fitted scaler for rdkit200 (fit on train, applied to valid/test)
        self._scaler: Optional[StandardScaler] = None
        # Descriptor names retained after variance filtering (rdkit200)
        self._rdkit_cols: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def featurize_endpoint(self, endpoint_dir: str) -> None:
        """Featurize all splits for one endpoint and write CSV files.

        Reads ``{split}_smiles.csv`` files from ``endpoint_dir``, converts
        SMILES to features, and writes ``{featurizer}/{split}.csv`` files.
        The scaler is fit on the training split and applied to valid/test.

        Parameters
        ----------
        endpoint_dir : str
            Path to the endpoint directory produced by ``TDCAdmetLoader``,
            e.g. ``data/admet/hERG``.
        """
        out_dir = os.path.join(endpoint_dir, self.featurizer)
        os.makedirs(out_dir, exist_ok=True)

        # Reset per-endpoint state
        self._scaler = None
        self._rdkit_cols = None

        for split_name, smiles_file in _SPLIT_FILES.items():
            smiles_path = os.path.join(endpoint_dir, smiles_file)
            if not os.path.exists(smiles_path):
                log.debug(f"  {smiles_path} not found, skipping")
                continue

            raw_df = pd.read_csv(smiles_path)
            smiles_col = self._detect_smiles_col(raw_df)
            label_col = "Y"

            X, valid_mask = self._smiles_to_features(
                raw_df[smiles_col].tolist(),
                fit=(split_name == "train"),  # fit scaler on full train split
            )
            y = raw_df[label_col].values[valid_mask]

            out_df = pd.DataFrame(X)
            out_df["Y"] = y  # label always last column
            out_path = os.path.join(out_dir, f"{split_name}.csv")
            out_df.to_csv(out_path, index=False)

            n_dropped = (~valid_mask).sum()
            log.info(
                f"  [{self.featurizer}] {os.path.basename(endpoint_dir)}/{split_name}: "
                f"{X.shape[0]} samples × {X.shape[1]} features "
                f"({n_dropped} SMILES failed)"
            )

    def featurize_all_endpoints(self, data_dir: str) -> None:
        """Featurize all endpoints under ``data_dir``.

        Parameters
        ----------
        data_dir : str
            Root directory containing per-endpoint subdirectories
            (as produced by ``TDCAdmetLoader``).
        """
        endpoint_dirs = [
            os.path.join(data_dir, d)
            for d in sorted(os.listdir(data_dir))
            if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith("_")
        ]
        log.info(f"Featurizing {len(endpoint_dirs)} endpoints with '{self.featurizer}'")
        for ep_dir in endpoint_dirs:
            try:
                self.featurize_endpoint(ep_dir)
            except Exception as exc:
                log.error(f"Failed to featurize {ep_dir}: {exc}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _smiles_to_features(
        self, smiles_list: List[str], fit: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert a list of SMILES to a feature matrix.

        Parameters
        ----------
        smiles_list : list of str
        fit : bool
            If True, fit the scaler (rdkit200) or record column list on this
            split (called once per endpoint for the training split).

        Returns
        -------
        X : np.ndarray, shape (n_valid, n_features)
        valid_mask : np.ndarray of bool, shape (n_smiles,)
            True for SMILES that were successfully converted.
        """
        if self.featurizer == "ecfp4":
            return self._ecfp4(smiles_list)
        elif self.featurizer == "maccs":
            return self._maccs(smiles_list)
        elif self.featurizer == "rdkit200":
            return self._rdkit200(smiles_list, fit=fit)
        raise ValueError(f"Unknown featurizer: {self.featurizer}")

    # ── ECFP4 ──────────────────────────────────────────────────────────────
    def _ecfp4(self, smiles_list: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        rows, valid = [], []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is None:
                valid.append(False)
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=self.ecfp4_radius, nBits=self.ecfp4_nbits
            )
            rows.append(list(fp))
            valid.append(True)
        return np.array(rows, dtype=np.float32), np.array(valid, dtype=bool)

    # ── MACCS ──────────────────────────────────────────────────────────────
    def _maccs(self, smiles_list: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        from rdkit import Chem
        from rdkit.Chem import MACCSkeys

        rows, valid = [], []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is None:
                valid.append(False)
                continue
            fp = MACCSkeys.GenMACCSKeys(mol)
            rows.append(list(fp))
            valid.append(True)
        return np.array(rows, dtype=np.float32), np.array(valid, dtype=bool)

    # ── RDKit200 ────────────────────────────────────────────────────────────
    def _rdkit200(
        self, smiles_list: List[str], fit: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        from rdkit.ML.Descriptors import MoleculeDescriptors

        # All RDKit 2D descriptor names
        desc_names = [name for name, _ in Descriptors.descList]
        calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)

        rows, valid = [], []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is None:
                valid.append(False)
                continue
            descs = np.array(calc.CalcDescriptors(mol), dtype=np.float64)
            rows.append(descs)
            valid.append(True)

        if not rows:
            return np.empty((0, len(desc_names))), np.array(valid, dtype=bool)

        X = np.array(rows)
        # Replace inf / NaN with column mean (or 0 if all-NaN column)
        col_means = np.nanmean(np.where(np.isinf(X), np.nan, X), axis=0)
        col_means = np.nan_to_num(col_means, nan=0.0)
        mask_bad = ~np.isfinite(X)
        X[mask_bad] = np.take(col_means, np.where(mask_bad)[1])

        if fit:
            # Remove near-zero-variance columns, keep top rdkit_n_components
            stds = X.std(axis=0)
            keep_idx = np.argsort(stds)[::-1][: self.rdkit_n_components]
            keep_idx = np.sort(keep_idx)
            self._rdkit_cols = [desc_names[i] for i in keep_idx]
            X = X[:, keep_idx]
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)
        else:
            # Apply previously fit column selection + scaler
            if self._rdkit_cols is None or self._scaler is None:
                raise RuntimeError(
                    "RDKit200 featurizer must be fit on training data first. "
                    "Call featurize_endpoint() which fits on 'train' split."
                )
            col_indices = [desc_names.index(c) for c in self._rdkit_cols]
            X = X[:, col_indices]
            X = self._scaler.transform(X)

        return X.astype(np.float32), np.array(valid, dtype=bool)

    @staticmethod
    def _detect_smiles_col(df: pd.DataFrame) -> str:
        """Auto-detect the SMILES column name in a TDC DataFrame."""
        for candidate in ("Drug", "SMILES", "smiles", "drug", "Drug_SMILES"):
            if candidate in df.columns:
                return candidate
        raise ValueError(
            f"Cannot detect SMILES column. Available columns: {df.columns.tolist()}"
        )
