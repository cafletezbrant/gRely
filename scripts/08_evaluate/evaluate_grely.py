#!/usr/bin/env python
"""Evaluate gRely on the held-out cross-validation predictions.

Stage 08 of the pipeline. Scores every tuple out-of-fold (each held-out
prediction is scored by the model that held it out), then computes the reliability
metrics behind the paper's Fig 1 / Table 1 as source-data CSVs:

  coverage_accuracy_grely.csv       accuracy + gene coverage vs. gRely threshold
  coverage_accuracy_magnitude.csv   the same, ranking by |avg_logSum| (baseline)
  average_precision_by_fold.csv     AP per fold (+ mean / std) on sign concordance
  threshold_table.csv               |VEP| thresholding vs. gRely quantile filtering
  per_tissue_accuracy.csv           baseline / top / bottom quantile per tissue
  effect_size_correlation.csv       predicted VEP vs. observed eQTL beta (all / top)
  mean_grely_per_tissue.csv         mean gRely score vs. concordance per tissue

Example
-------
    python scripts/08_evaluate/evaluate_grely.py \
        --matrix data/task_sign_matching_with_features.parquet \
        --model-dir models --out-dir analysis/evaluation
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from grely import apply as gapply      # noqa: E402
from grely import data as gdata         # noqa: E402
from grely import evaluate as geval     # noqa: E402
from grely.paths import load_paths      # noqa: E402


def main():
    PATHS = load_paths()
    ap = argparse.ArgumentParser(description="Evaluate gRely (coverage / accuracy / per-tissue).")
    ap.add_argument("--matrix", default=PATHS['tables']['feature_matrix'],
                    help="Compiled feature matrix parquet.")
    ap.add_argument("--model-dir", default=PATHS['models']['models_dir'],
                    help="Directory with trained fold models.")
    ap.add_argument("--out-dir", default=PATHS['analysis']['evaluation_dir'],
                    help="Directory for the evaluation CSVs.")
    ap.add_argument("--config", default=None, help="Path to grely.yaml.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = gdata.load_config(args.config)

    df = pd.read_parquet(args.matrix)
    df, features, _ = gdata.prepare(df, cfg)

    # Pooled held-out predictions, each scored by the model that held it out.
    scored = gapply.predict_out_of_fold(df, features, model_dir=args.model_dir, config=cfg)
    print(f"Evaluating {len(scored)} held-out predictions across "
          f"{scored['fold'].nunique()} folds.")

    def out(name):
        return os.path.join(args.out_dir, name)

    # Coverage vs. accuracy — gRely score and the effect-size baseline.
    geval.coverage_curve(scored, rank_col=geval.SCORE_COL).to_csv(
        out("coverage_accuracy_grely.csv"), index=False)
    geval.coverage_curve(scored, rank_col=geval.MAG_COL, abs_rank=True).to_csv(
        out("coverage_accuracy_magnitude.csv"), index=False)

    # Precision / recall on the sign-concordance task.
    ap_by_fold = geval.average_precision_by_fold(scored)
    ap_by_fold.to_csv(out("average_precision_by_fold.csv"), index=False)

    # Magnitude thresholding vs. gRely quantile filtering (Table 1).
    table = geval.threshold_table(scored)
    table.to_csv(out("threshold_table.csv"), index=False)

    # Per-tissue reliability.
    geval.per_tissue_quantile_accuracy(scored).to_csv(out("per_tissue_accuracy.csv"), index=False)

    # Effect-size correlation (predicted VEP vs. observed beta).
    if geval.BETA_COL in scored.columns:
        esc = geval.effect_size_correlation(scored)
        pd.DataFrame([esc]).to_csv(out("effect_size_correlation.csv"), index=False)

    # Mean gRely per tissue vs. observed concordance.
    per_tissue, rho = geval.mean_grely_per_tissue(scored)
    per_tissue.to_csv(out("mean_grely_per_tissue.csv"), index=False)

    # Console summary.
    mean_ap = ap_by_fold.loc[ap_by_fold["fold"] == "mean", "average_precision"].iloc[0]
    std_ap = ap_by_fold.loc[ap_by_fold["fold"] == "std", "average_precision"].iloc[0]
    all_acc = table.loc[table["filter"] == "All", "accuracy"].iloc[0]
    top20 = table.loc[table["filter"] == "gRely top 20%", "accuracy"].iloc[0]
    print(f"\nMean average precision: {mean_ap:.3f} ± {std_ap:.3f}")
    print(f"Sign concordance — all: {all_acc:.3f}  |  gRely top 20%: {top20:.3f}")
    print(f"Mean gRely per tissue vs. observed concordance: Spearman rho = {rho:.3f}")
    print(f"\nWrote evaluation outputs -> {args.out_dir}")


if __name__ == "__main__":
    main()
