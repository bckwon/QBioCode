#!/usr/bin/env python3
"""
Compile key numbers from the ADMET QML benchmark for paper writing.
Prints a JSON object with all figures needed by the write-admet-paper skill.
Run from repo root: python3 .bob/skills/write-admet-paper/compile_results.py
"""
import glob, json, os, re, sys, warnings
warnings.filterwarnings("ignore")

try:
    import pandas as pd
    import yaml
except ImportError:
    sys.exit("pip install pandas pyyaml first")

REPO = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../../..")
EP_RE = re.compile(r"data/admet/([^/]+)/([^/]+?)/?$")
CLASSICAL = {"lr", "rf", "mlp", "xgb", "svc"}
FEATS = {"ecfp4", "maccs", "rdkit200"}


def load_csv_with_ep(pattern, ep_re=EP_RE):
    rows = []
    for f in glob.glob(pattern, recursive=True):
        cfg = os.path.join(os.path.dirname(f), ".hydra", "config.yaml")
        if not os.path.exists(cfg):
            continue
        try:
            c = yaml.safe_load(open(cfg))
            fp = c.get("folder_path", "") if isinstance(c, dict) else ""
        except Exception:
            raw = open(cfg).read()
            m0 = re.search(r"folder_path:\s*(.+)", raw)
            fp = m0.group(1).strip() if m0 else ""
        m = ep_re.search(fp)
        if not m:
            continue
        ep, feat = m.group(1), m.group(2)
        if feat not in FEATS:
            continue
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        df["endpoint"] = ep
        df["featurizer"] = feat
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).rename(
        columns={"auc": "auroc", "f1_score": "f1"}
    )


def best(df, model_filter, emb_filter):
    sub = df.copy()
    if model_filter:
        sub = sub[sub["model"].isin(model_filter)]
    if emb_filter:
        sub = sub[sub["embeddings"] == emb_filter]
    if sub.empty:
        return None, 0
    sub = sub.sort_values("auroc", ascending=False).drop_duplicates(
        ["endpoint", "featurizer", "model"]
    )
    return round(sub["auroc"].mean(), 4), len(sub)


# ── Condition A: cl@full/none ──────────────────────────────────────────────
main_test = load_csv_with_ep(
    f"{REPO}/results/admet_config/dataset=test.csv/**/ModelResults.csv"
)

cl_full_none_auc, cl_full_none_n = best(main_test, CLASSICAL, "none")
cl_full_pca_auc,  cl_full_pca_n  = best(main_test, CLASSICAL, "pca")
qsvc_full_pca_auc, qsvc_full_pca_n = best(main_test, {"qsvc"}, "pca")

# ── Condition B: cl@300/none and cl@300/pca ───────────────────────────────
cl300_test = load_csv_with_ep(
    f"{REPO}/results/admet_classical300_config/dataset=test.csv/**/ModelResults.csv"
)
cl300_none_auc, cl300_none_n = best(cl300_test, CLASSICAL, "none")
cl300_pca_auc,  cl300_pca_n  = best(cl300_test, CLASSICAL, "pca")

# ── Condition C: QSVC@300/pca (train_qml split) ───────────────────────────
qsvc300 = load_csv_with_ep(
    f"{REPO}/results/admet_qsvc_config/dataset=train_qml.csv/**/ModelResults.csv"
)
qsvc300_pca_auc, qsvc300_pca_n = best(qsvc300, {"qsvc"}, "pca")

# ── Gap decomposition ──────────────────────────────────────────────────────
if all(v is not None for v in [cl_full_none_auc, cl300_none_auc,
                                cl300_pca_auc, qsvc300_pca_auc]):
    data_starvation     = round(cl_full_none_auc - cl300_none_auc, 4)
    feature_compression = round(cl300_none_auc   - cl300_pca_auc,  4)
    model_penalty       = round(cl300_pca_auc    - qsvc300_pca_auc, 4)
    total_gap           = round(cl_full_none_auc - qsvc300_pca_auc, 4)
else:
    data_starvation = feature_compression = model_penalty = total_gap = None

# ── Full model ranking from precomputed summary ────────────────────────────
summary_csv = f"{REPO}/results/admet_benchmark/tables/performance_summary_model.csv"
model_ranking = []
if os.path.exists(summary_csv):
    sm = pd.read_csv(summary_csv)
    auroc_sm = sm[sm["metric"] == "AUROC"][["model", "model_display", "model_type",
                                              "mean", "std", "n_endpoints"]]
    auroc_sm = auroc_sm.sort_values("mean", ascending=False)
    model_ranking = auroc_sm.to_dict(orient="records")

# ── QML wins ──────────────────────────────────────────────────────────────
qml_vs = f"{REPO}/results/admet_benchmark/tables/qml_vs_classical.csv"
qml_wins_list = []
mean_delta = None
if os.path.exists(qml_vs):
    qv = pd.read_csv(qml_vs)
    auroc_rows = qv[qv.get("metric", pd.Series(["AUROC"]*len(qv))) == "AUROC"] \
        if "metric" in qv.columns else qv
    if "qml_wins" in auroc_rows.columns:
        wins = auroc_rows[auroc_rows["qml_wins"] == True]
        qml_wins_list = wins["endpoint"].tolist()
    if "delta_qml_minus_classical" in auroc_rows.columns:
        mean_delta = round(auroc_rows["delta_qml_minus_classical"].mean(), 4)

# ── Per-category AUROC ────────────────────────────────────────────────────
cat_csv = f"{REPO}/results/admet_benchmark/tables/performance_by_category.csv"
category_auroc = {}
if os.path.exists(cat_csv):
    cat = pd.read_csv(cat_csv)
    for _, row in cat.iterrows():
        category_auroc[row["category"]] = {
            "best_classical": round(max(
                row.get("LR", 0), row.get("MLP", 0),
                row.get("RF", 0), row.get("XGBoost", 0)), 4),
            "qsvc": round(row.get("QSVC", 0), 4),
        }

out = {
    "ablation": {
        "cl_full_none":  {"auroc": cl_full_none_auc,  "n": cl_full_none_n},
        "cl_300_none":   {"auroc": cl300_none_auc,    "n": cl300_none_n},
        "cl_300_pca":    {"auroc": cl300_pca_auc,     "n": cl300_pca_n},
        "qsvc_300_pca":  {"auroc": qsvc300_pca_auc,   "n": qsvc300_pca_n},
    },
    "gap_decomposition": {
        "total_gap":           total_gap,
        "data_starvation":     data_starvation,
        "feature_compression": feature_compression,
        "model_penalty":       model_penalty,
    },
    "model_ranking_auroc": model_ranking,
    "qml_wins": {
        "n_endpoints": len(qml_wins_list),
        "total_endpoints": 22,
        "endpoints": qml_wins_list,
        "mean_delta_auroc": mean_delta,
    },
    "category_auroc": category_auroc,
}

print(json.dumps(out, indent=2))
