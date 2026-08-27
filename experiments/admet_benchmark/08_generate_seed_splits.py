#!/usr/bin/env python3
"""
Generate per-seed training subsamples for L1 multi-seed ablation.

For each seed in SEEDS (excluding 42 which already exists as train_qml.csv),
creates data/admet_seeds/seed_{S}/{endpoint}/{feat}/train_qml.csv
by calling cap_qml_split with seed=S.

Test and valid splits are SYMLINKED from the original data/admet/ directory
(never copied — both point to the same TDC canonical files).

Usage:
    .venv/bin/python3 experiments/admet_benchmark/08_generate_seed_splits.py
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

SEEDS = [0, 21, 84, 100]   # seed=42 already in data/admet/{ep}/{feat}/train_qml.csv
QML_CAP = 300
ENDPOINTS = [
    "AMES", "BBB_Martins", "Bioavailability_Ma", "CYP1A2_Veith",
    "CYP2C19_Veith", "CYP2C9_Substrate_CarbonMangels", "CYP2C9_Veith",
    "CYP2D6_Substrate_CarbonMangels", "CYP2D6_Veith",
    "CYP3A4_Substrate_CarbonMangels", "CYP3A4_Veith", "Caco2_Wang",
    "Clearance_Hepatocyte_AZ", "DILI", "HIA_Hou", "Half_Life_Obach",
    "Lipophilicity_AstraZeneca", "PPBR_AstraZeneca", "Pgp_Broccatelli",
    "Solubility_AqSolDB", "VDss_Lombardo", "hERG",
]
FEATS = ["ecfp4", "maccs", "rdkit200"]


def cap_qml_split(train_df: pd.DataFrame, seed: int, cap: int = QML_CAP) -> pd.DataFrame:
    """Stratified subsample — mirrors tdc_admet_loader._cap_qml_split."""
    if len(train_df) <= cap:
        return train_df.copy()
    classes = train_df["Y"].unique()
    n_per_class = cap // len(classes)
    rng = np.random.RandomState(seed)
    parts = []
    for cls in sorted(classes):
        cls_df = train_df[train_df["Y"] == cls]
        n = min(n_per_class, len(cls_df))
        parts.append(cls_df.sample(n=n, random_state=rng.randint(0, 10000)))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def main():
    src_base = REPO / "data" / "admet"
    dst_base = REPO / "data" / "admet_seeds"

    total = 0
    for seed in SEEDS:
        print(f"\n=== seed={seed} ===")
        for ep in ENDPOINTS:
            for feat in FEATS:
                src_dir = src_base / ep / feat
                train_path = src_dir / "train.csv"
                if not train_path.exists():
                    print(f"  SKIP {ep}/{feat}: no train.csv")
                    continue

                dst_dir = dst_base / f"seed_{seed}" / ep / feat
                dst_dir.mkdir(parents=True, exist_ok=True)

                # 1. Generate new train_qml.csv
                train_df = pd.read_csv(train_path)
                qml_df = cap_qml_split(train_df, seed)
                qml_df.to_csv(dst_dir / "train_qml.csv", index=False)

                # 2. Symlink test.csv and valid.csv from original (never copy)
                for fname in ["test.csv", "valid.csv"]:
                    link = dst_dir / fname
                    target = os.path.relpath(src_dir / fname, dst_dir)
                    if link.exists() or link.is_symlink():
                        link.unlink()
                    link.symlink_to(target)

                total += 1
                print(f"  OK  {ep}/{feat}  n_qml={len(qml_df)}")

    print(f"\nGenerated {total} train_qml.csv files across {len(SEEDS)} seeds.")


if __name__ == "__main__":
    main()
