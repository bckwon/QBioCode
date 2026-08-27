#!/usr/bin/env python3
"""
Generate per-q_seed YAML configs for the L2 VQC initialisation seed sweep.

Data seed (train subsample) stays at 42 — only the quantum circuit
initialisation seed (q_seed) changes.

Creates:
  qbiocode/apps/qprofiler/configs/admet_vqc_qseed_{S}.yaml

for each q_seed in Q_SEEDS. q_seed=42 is already covered by admet_config.yaml.

Usage:
    .venv/bin/python3 experiments/admet_benchmark/09b_generate_qseed_configs.py
"""
import yaml
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
Q_SEEDS = [0, 7, 21, 73, 84, 100, 123, 200, 314]   # 9 additional; q_seed=42 already run
CFG_DIR = REPO / "qbiocode/apps/qprofiler/configs"

with open(CFG_DIR / "admet_config.yaml") as f:
    base = yaml.safe_load(f)

for qs in Q_SEEDS:
    cfg = dict(base)
    cfg["seed"] = 42            # data seed stays fixed
    cfg["q_seed"] = qs
    cfg["model"] = ["vqc"]      # VQC only
    cfg["config_file_name"] = f"admet_vqc_qseed_{qs}"
    out = CFG_DIR / f"admet_vqc_qseed_{qs}.yaml"
    with open(out, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Wrote {out}")

print(f"\nDone. Wrote {len(Q_SEEDS)} VQC q_seed config files.")
