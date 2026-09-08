"""Gene-level sequence / chromatin features.

For each gene body (union of exon spans, `union_start`..`union_end`), computes GC
content and, from the panel of EpiMap "Other" average chromatin tracks (ATAC,
DNase, H3K27ac, H3K4me1, H3K4me3, H3K9ac, H3K9me3, H3K27me3, H3K36me3, CTCF,
POLR2A), per-track summary statistics across the span: mean, median, max, min,
std, 25th/75th percentiles, fraction non-zero, fraction missing, and total
signal. Writes one parquet per gene (numeric feature columns only) to
`outputs.gene_sequence_features`, which `compile_final_feature_set.py` reads and
prefixes with `gene_sequence_`.

Run as a SLURM array job: shard index = sys.argv[1] (100 genes per shard).
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig
import pyfaidx
from Bio.SeqUtils import gc_fraction

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from grely.paths import load_paths

P = load_paths()


def gc_content_span(fasta, chrom, start, end):
    seq = fasta[chrom][int(start):int(end)].seq.upper()
    return gc_fraction(seq) if len(seq) > 0 else float('nan')


def extract_bigwig_span(bw_path, chrom, start, end):
    bw = pyBigWig.open(bw_path)
    try:
        values = bw.values(chrom, int(start), int(end))
        values = np.array([v for v in values if v is not None])
    except RuntimeError:
        values = np.array([])
    bw.close()

    if len(values) == 0:
        return {
            'mean': float('nan'), 'median': float('nan'), 'max': float('nan'),
            'min': float('nan'), 'std': float('nan'), 'q25': float('nan'),
            'q75': float('nan'), 'nonzero_frac': float('nan'), 'nan_frac': 1.0,
            'sum': float('nan'),
        }

    return {
        'mean': float(np.mean(values)),
        'median': float(np.median(values)),
        'max': float(np.max(values)),
        'min': float(np.min(values)),
        'std': float(np.std(values)),
        'q25': float(np.percentile(values, 25)),
        'q75': float(np.percentile(values, 75)),
        'nonzero_frac': float(np.mean(values > 0)),
        'nan_frac': float(np.isnan(values).mean()),
        'sum': float(np.sum(values)),
    }


fasta = pyfaidx.Fasta(P['reference']['fasta'])

genes = pd.read_csv(P['tables']['all_genes'], sep='\t')
job_id = int(sys.argv[1])
genes = genes.iloc[job_id * 100:job_id * 100 + 100].reset_index(drop=True)

bigwig_dir = P['bigwig_dir']
bigwig_paths = {
    'ATAC': os.path.join(bigwig_dir, 'average_ATAC-seq_observed_Other.bigWig'),
    'H3K27ac': os.path.join(bigwig_dir, 'average_H3K27ac_observed_Other.bigWig'),
    'H3K4me1': os.path.join(bigwig_dir, 'average_H3K4me1_observed_Other.bigWig'),
    'H3K9me3': os.path.join(bigwig_dir, 'average_H3K9me3_observed_Other.bigWig'),
    'CTCF': os.path.join(bigwig_dir, 'average_CTCF_observed_Other.bigWig'),
    'H3K27me3': os.path.join(bigwig_dir, 'average_H3K27me3_observed_Other.bigWig'),
    'H3K4me3': os.path.join(bigwig_dir, 'average_H3K4me3_observed_Other.bigWig'),
    'POLR2A': os.path.join(bigwig_dir, 'average_POLR2A_observed_Other.bigWig'),
    'DNase': os.path.join(bigwig_dir, 'average_DNase-seq_observed_Other.bigWig'),
    'H3K36me3': os.path.join(bigwig_dir, 'average_H3K36me3_observed_Other.bigWig'),
    'H3K9ac': os.path.join(bigwig_dir, 'average_H3K9ac_observed_Other.bigWig'),
}

out_dir = P['outputs']['gene_sequence_features']
os.makedirs(out_dir, exist_ok=True)

for _, row in genes.iterrows():
    gene_id = row['gene_id']
    chrom = row['chrom']
    start = int(row['union_start'])
    end = int(row['union_end'])

    out_path = os.path.join(out_dir, f'{gene_id}.parquet')
    if os.path.exists(out_path):
        try:
            pd.read_parquet(out_path)
            continue
        except Exception as e:
            print(f"Skipping: corrupted file.\n - {out_path}\nError: {e}")
            pass

    features = {'GC_content': gc_content_span(fasta, chrom, start, end)}
    for track, bw_path in bigwig_paths.items():
        stats = extract_bigwig_span(bw_path, chrom, start, end)
        for stat_name, value in stats.items():
            features[f"{track}_{stat_name}"] = value

    pd.DataFrame([features]).to_parquet(out_path, index=False)
