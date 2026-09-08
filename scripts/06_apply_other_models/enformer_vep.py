#!/usr/bin/env python
"""Compute Enformer variant effect predictions (VEPs) for the gRely test tuples.

Part of stage 06 ("apply gRely to other models"). This produces the per-tuple
Enformer VEP and its sign concordance against the fine-mapped eQTL, in the same
logSUM formulation used for Borzoi, so that a trained gRely model can then score
these predictions via ``apply_grely.py``.

Enformer per-allele predictions are expected as ``{pred_dir}/{i}_ref.npy`` and
``{i}_alt.npy`` of shape (896, 5313) (896 output bins x Enformer human tracks),
where ``i`` is the row index into the test-tuple table. Running Enformer itself
is out of scope here (use the published Enformer model/weights).

The tissue -> CAGE track map and VEP loop are copied verbatim from the original
Enformer analysis; the gene-slice geometry is imported from ``grely.other_models``.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from grely import data as gdata                              # noqa: E402
from grely.other_models import (                             # noqa: E402
    get_gene_slice_enformer,
    load_exons_from_gtf,
)

# Enformer CAGE output-track index per GTEx tissue (verbatim from the analysis).
tissue_to_cage_idx = {
    # ---- set1 ----
    "brain_amygdala":              5095,  # amygdala, adult
    "brain_cerebellar_hemisphere": 4979,  # cerebellum, adult, pool1
    "brain_cortex":                5103,  # occipital cortex, adult   (see note)
    "nerve_tibial":                5078,  # Schwann Cells             (peripheral-nerve cell)
    "pituitary":                   5093,  # pituitary gland, adult
    "adrenal_gland":               4977,  # adrenal gland, adult, pool1
    "thyroid":                     4696,  # thyroid, adult, pool1

    # ---- set2 ----
    "artery_aorta":                4949,  # aorta, adult, pool1
    "artery_tibial":               5164,  # artery, adult            (generic; only muscular-artery bulk track)
    "heart_atrial_appendage":      4975,  # left atrium, adult        (see note)
    "heart_left_ventricle":        4974,  # left ventricle, adult
    "muscle":                      4691,  # skeletal muscle, adult, pool1
    "lung":                        4687,  # lung, adult, pool1
    "kidney_cortex":               4685,  # kidney, adult, pool1

    # ---- set3 ----
    "esophagus_gej":               4683,  # esophagus, adult, pool1   (see note)
    "esophagus_mucosa":            4683,  # esophagus, adult, pool1   (see note)
    "stomach":                     4959,  # stomach, FETAL            (no adult bulk track exists — see note)
    "small_intestine":             4692,  # small intestine, adult, pool1
    "colon_sigmoid":               4682,  # colon, adult, pool1       (see note)
    "colon_transverse":            4682,  # colon, adult, pool1       (see note)
    "liver":                       4686,  # liver, adult, pool1
    "pancreas":                    4946,  # pancreas, adult

    # ---- set4 ----
    "ovary":                       4688,  # ovary, adult, pool1
    "testis":                      4694,  # testis, adult, pool1
    "uterus":                      4910,  # uterus, adult, pool1
    "vagina":                      5175,  # vagina, adult
    "prostate":                    4690,  # prostate, adult, pool1
    "breast":                      4976,  # breast, adult
    "minor_salivary_gland":        4911,  # salivary gland, adult, pool1

    # ---- set5 ----
    "blood":                       4950,  # blood, adult, pool1
    "LCL":                         5110,  # B lymphoblastoid cell line: GM12878
    "spleen":                      4693,  # spleen, adult, pool1
    "skin_not_sun_exposed":        4970,  # skin, adult
    "skin_sun_exposed":            4970,  # skin, adult
    "fibroblast":                  4908,  # Fibroblast - Dermal
    "adipose_subcutaneous":        4902,  # Adipocyte - subcutaneous
    "adipose_visceral":            4795,  # Adipocyte - omental
}


def build_test_tuples(matrix_path, cfg):
    """Test tuples = chr1-3 ∩ each fold's held-out tissues, tagged with fold."""
    df = pd.read_parquet(matrix_path)
    df = gdata.add_chrom(df)
    df = gdata.filter_center_window(df, cfg)
    folds = gdata.get_folds(cfg)
    _, test_chroms = gdata.get_chroms(cfg)

    base = df[df["chrom"].isin(test_chroms)]
    parts = []
    for fold_name, test_tissues in folds:
        part = base[base["target_tissue"].isin(test_tissues)].copy()
        part["fold"] = fold_name
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    print(out.shape, out["fold"].value_counts().to_dict())
    return out


def main():
    ap = argparse.ArgumentParser(description="Compute Enformer VEPs for gRely test tuples.")
    ap.add_argument("--matrix", required=True, help="Feature matrix parquet (test tuples drawn from it).")
    ap.add_argument("--pred-dir", required=True, help="Directory of Enformer {i}_ref.npy / {i}_alt.npy.")
    ap.add_argument("--gtf", required=True, help="GENCODE v41 GTF (gencode41_basic_nort.gtf).")
    ap.add_argument("--out-csv", default="enformer_pred.csv")
    ap.add_argument("--config", default=None, help="Path to grely.yaml.")
    args = ap.parse_args()

    cfg = gdata.load_config(args.config)
    df = build_test_tuples(args.matrix, cfg)
    gene_to_exons = load_exons_from_gtf(args.gtf)

    vep_scores, sign_matches, gene_slice_lens = [], [], []
    for i in tqdm(range(len(df)), desc="Computing Enformer VEPs"):
        row      = df.iloc[i]
        gene_id  = row['gene_id']
        chrom    = row.get('chrom', None)
        center   = int(row['pos'])               # 1-based
        beta     = row['beta']
        cage_idx = tissue_to_cage_idx[row['target_tissue']]

        ref_pred = np.load(f'{args.pred_dir}/{i}_ref.npy')   # (896, 5313)
        alt_pred = np.load(f'{args.pred_dir}/{i}_alt.npy')   # (896, 5313)

        gene_slice = get_gene_slice_enformer(
            gene_to_exons, gene_id,
            center_pos=center - 1,             # 1-based -> 0-based
            expected_chrom=chrom,
        )
        if gene_slice is None:
            vep_scores.append(np.nan); sign_matches.append(False); gene_slice_lens.append(0)
            continue

        ref_track = ref_pred[gene_slice, cage_idx].astype(np.float32)
        alt_track = alt_pred[gene_slice, cage_idx].astype(np.float32)

        # logSUM, matching the Borzoi/AG formulation
        ref_log = np.log2(ref_track + 1.0)
        alt_log = np.log2(alt_track + 1.0)
        logsum  = float(alt_log.sum() - ref_log.sum())

        vep_scores.append(logsum)
        sign_matches.append(np.sign(logsum) == np.sign(beta))
        gene_slice_lens.append(len(gene_slice))

    df['enformer_vep']             = np.asarray(vep_scores)
    df['enformer_sign_concordant'] = np.asarray(sign_matches)
    df['enf_gene_slice_bins']      = np.asarray(gene_slice_lens)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}  (n={len(df)})")


if __name__ == "__main__":
    main()
