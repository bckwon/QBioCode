#!/usr/bin/env python3
"""
Generate per-seed YAML configs for the L1 seed ablation.

Creates:
  qbiocode/apps/qprofiler/configs/admet_qsvc_seed_{S}.yaml    (QSVC)
  qbiocode/apps/qprofiler/configs/admet_classical300_seed_{S}.yaml  (classical@300)

for each seed S in [0, 21, 84, 100].

Usage:
    .venv/bin/python3 experiments/admet_benchmark/09_generate_seed_configs.py
"""
import yaml
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SEEDS = [0, 21, 84, 100]
CFG_DIR = REPO / "qbiocode/apps/qprofiler/configs"

# ── QSVC configs ─────────────────────────────────────────────────────────────
with open(CFG_DIR / "admet_qsvc_config.yaml") as f:
    base_qsvc = yaml.safe_load(f)

for seed in SEEDS:
    cfg = dict(base_qsvc)
    cfg["seed"] = seed
    cfg["q_seed"] = seed
    cfg["config_file_name"] = f"admet_qsvc_seed_{seed}"
    out = CFG_DIR / f"admet_qsvc_seed_{seed}.yaml"
    with open(out, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Wrote {out}")

# ── Classical@300 seed configs ────────────────────────────────────────────────
with open(CFG_DIR / "admet_classical300_config.yaml") as f:
    base_cl = yaml.safe_load(f)

for seed in SEEDS:
    cfg = dict(base_cl)
    cfg["seed"] = seed
    cfg["q_seed"] = seed
    cfg["config_file_name"] = f"admet_classical300_seed_{seed}"
    out = CFG_DIR / f"admet_classical300_seed_{seed}.yaml"
    with open(out, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Wrote {out}")

print(f"\nDone. Wrote {len(SEEDS) * 2} config files.")
