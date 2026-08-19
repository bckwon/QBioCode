---
name: write-admet-paper
description: Use when the user asks to write, update, or revise the QML-ADMET benchmark paper (paper/summary.md). Compiles live results from the experiment tree, identifies research gaps from the literature, and produces a concise empirical paper draft.
---

# Write QML-ADMET Benchmark Paper

Follow these steps every time this skill activates.

## Step 1 — Compile live results

Run the supporting script to extract the four ablation numbers and the full model-ranking table:

```
execute_command: python3 .bob/skills/write-admet-paper/compile_results.py
```

Capture its JSON output. Do not hallucinate numbers — use only what the script returns.

## Step 2 — Identify research gaps

Read the following files to ground the literature context:

- `paper/summary.md` (if it already exists — note what needs updating)
- `qbiocode/apps/qprofiler/configs/admet_qsvc_config.yaml` (understand the experimental setup)
- `results/admet_benchmark/tables/performance_summary_model.csv`
- `results/admet_benchmark/tables/qml_vs_classical.csv`

The key gaps this work addresses (use these verbatim as anchor points):

1. **No standardised benchmark** — prior QML-ADMET studies each use different endpoints, splits, and
   featurisers, making cross-study comparison impossible.
2. **Conflated confounders** — data starvation, feature compression, and model penalty are never
   disentangled; it is unknown whether QML loses because of small training sets, dimensionality
   reduction, or intrinsic model capacity.
3. **Optimistic hyperparameters** — many published QSVC evaluations use default C=0.01 without
   MinMaxScaler pre-scaling, which collapses the kernel matrix; results are not reproducible.
4. **Missing baselines** — QML is rarely compared to all five classical baselines (LR, RF, MLP,
   XGBoost, SVC) on the same splits.

## Step 3 — Write or overwrite `paper/summary.md`

Use `write_file` to produce the paper. The structure must be exactly:

```
# Title
## Abstract        (~100 words)
## Introduction    (motivation + gaps, ~200 words)
## Methods         (experimental design + justification, ~300 words)
## Results         (numbers from Step 1 + table + interpretation, ~300 words)
## Discussion      (implications, limitations, future work, ~200 words)
## References      (5–8 key citations, formatted as markdown list)
```

Rules:
- Use **bold** for numbers that matter; keep prose tight.
- Every claim must be supported by a number from Step 1 or a named reference.
- No marketing language ("groundbreaking", "novel", "state-of-the-art").
- The ablation table must appear in Results as a markdown table with the four conditions.
- Total word count target: 900–1200 words (excluding references).

## Step 4 — Validate

After writing, count the sections and confirm the ablation table renders as a markdown table
(four rows, four columns minimum). Report the word count to the user.
