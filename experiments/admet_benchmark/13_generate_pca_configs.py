#!/usr/bin/env python3
"""
Generate per-PCA-dimension YAML configs for the L3 PCA dimension sweep.

Creates:
  qbiocode/apps/qprofiler/configs/admet_qsvc_pca_{K}.yaml    (QSVC)
  qbiocode/apps/qprofiler/configs/admet_cl300_pca_{K}.yaml   (classical@300)

for K in [4, 12, 16, 32]. K=8 already exists in admet_qsvc_config.yaml.

Usage:
    .venv/bin/python3 experiments/admet_benchmark/13_generate_pca_configs.py
"""
import yaml
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DIMS = [4, 12, 16, 32]   # K=8 already run
CFG_DIR = REPO / "qbiocode/apps/qprofiler/configs"

# ── QSVC configs ─────────────────────────────────────────────────────────────
with open(CFG_DIR / "admet_qsvc_config.yaml") as f:
    base_qsvc = yaml.safe_load(f)

for K in DIMS:
    cfg = dict(base_qsvc)
    cfg["n_components"] = K
    cfg["config_file_name"] = f"admet_qsvc_pca_{K}"
    out = CFG_DIR / f"admet_qsvc_pca_{K}.yaml"
    with open(out, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Wrote {out}")

# ── Classical@300 PCA configs ─────────────────────────────────────────────────
with open(CFG_DIR / "admet_classical300_config.yaml") as f:
    base_cl = yaml.safe_load(f)

for K in DIMS:
    cfg = dict(base_cl)
    cfg["n_components"] = K
    cfg["config_file_name"] = f"admet_cl300_pca_{K}"
    out = CFG_DIR / f"admet_cl300_pca_{K}.yaml"
    with open(out, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Wrote {out}")

print(f"\nDone. Wrote {len(DIMS) * 2} PCA config files.")
