#!/usr/bin/env python
"""Apply a trained gRely model to another S2F model's VEPs and evaluate transfer.

Stage 06 of the pipeline. The Borzoi-trained gRely fold models are applied,
without retraining, to a table that contains the gRely features **and** another
model's per-tuple sign-concordance column (e.g. produced by
``alphagenome_vep.py`` or ``enformer_vep.py``). It attaches the reliability score
and reports overall vs top-20% vs bottom-20% sign concordance.

Example
-------
    python scripts/06_apply_other_models/apply_grely.py \
        --matrix data/task_sign_matching_with_features.parquet \
        --other-model-csv alphagenome_pred.csv \
        --sign-col alphagenome_sign_concordant \
        --join-on variant,gene_id,target_tissue \
        --model-dir models
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from grely import apply as gapply   # noqa: E402
from grely import data as gdata     # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Apply gRely to another model's VEPs.")
    ap.add_argument("--matrix", required=True,
                    help="Feature matrix parquet (same features used to train gRely).")
    ap.add_argument("--other-model-csv", default=None,
                    help="CSV with the other model's sign-concordance column; merged onto --matrix.")
    ap.add_argument("--sign-col", required=True,
                    help="Boolean column: is the other model's VEP sign-concordant with the eQTL?")
    ap.add_argument("--join-on", default="variant,gene_id,target_tissue",
                    help="Comma-separated keys to merge --other-model-csv onto the matrix.")
    ap.add_argument("--model-dir", default="models", help="Directory with trained fold models.")
    ap.add_argument("--config", default=None, help="Path to grely.yaml.")
    ap.add_argument("--out-csv", default=None, help="Optional path to write the scored table.")
    args = ap.parse_args()

    cfg = gdata.load_config(args.config)
    df = pd.read_parquet(args.matrix)
    df, features, _ = gdata.prepare(df, cfg)

    if args.other_model_csv:
        keys = [k.strip() for k in args.join_on.split(",")]
        other = pd.read_csv(args.other_model_csv)
        keep = keys + [args.sign_col]
        df = df.merge(other[keep], on=keys, how="left")

    if args.sign_col not in df.columns:
        ap.error(f"--sign-col '{args.sign_col}' not found after merge. "
                 f"Provide --other-model-csv or ensure the column exists.")

    scored = gapply.score_by_fold(df, features, model_dir=args.model_dir, config=cfg)
    scored = scored.dropna(subset=["grely_prob", args.sign_col])

    res = gapply.transfer_eval(scored, sign_col=args.sign_col, prob_col="grely_prob")
    print("=" * 60)
    print(f"n = {res['n']}")
    print(f"Overall sign concordance : {res['overall_acc']:.3f}")
    print(f"Top 20%  (prob >= {res['top_threshold']:.3f}) : {res['top_acc']:.3f}  (n={res['n_top']})")
    print(f"Bottom 20% (prob <= {res['bot_threshold']:.3f}): {res['bot_acc']:.3f}  (n={res['n_bot']})")
    print("=" * 60)

    if args.out_csv:
        scored.to_csv(args.out_csv, index=False)
        print(f"Wrote scored table -> {args.out_csv}")


if __name__ == "__main__":
    main()
