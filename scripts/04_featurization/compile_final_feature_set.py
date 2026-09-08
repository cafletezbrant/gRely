import json
import os
import pickle
import sys
import time
import warnings
from multiprocessing import Pool, cpu_count
import bioframe as bf
import h5py
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfaidx
import pyranges as pr
import pysam
import seaborn as sns
import tensorflow as tf
from scipy.stats import (entropy, kurtosis, pearsonr, skew, spearmanr,
                           variation)
from collections import Counter

from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
from grely.paths import load_paths
P = load_paths()
sys.path.append(P['roots']['borzoi_examples'])
from baskerville import dna
from baskerville import gene as bgene
from baskerville import seqnn
from borzoi_helpers import *
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from scipy.special import rel_entr
import numpy as np
from scipy.stats import variation
import re
from pathlib import Path

def nanpercentile(a, q):
    finite_vals = a[np.isfinite(a)]
    return np.percentile(finite_vals, q) if finite_vals.size > 0 else np.nan

def summarize_vector(v, prefix: str):
    v = np.asarray(v, dtype=float)
    n = v.size
    av = np.abs(v)
    s1 = float(av.sum())

    feats = {}

    # Moments / norms / range
    feats[f"{prefix}_mean"]   = float(v.mean())
    feats[f"{prefix}_std"]    = float(v.std(ddof=0))
    feats[f"{prefix}_median"] = float(np.median(v))
    feats[f"{prefix}_mad"]    = float(1.4826 * np.median(np.abs(v - np.median(v))))  # robust scale
    feats[f"{prefix}_l1"]     = float(av.sum())
    feats[f"{prefix}_l2"]     = float(np.sqrt(np.dot(v, v)))
    feats[f"{prefix}_linf"]   = float(av.max(initial=0.0))
    feats[f"{prefix}_min"]    = float(v.min(initial=0.0))
    feats[f"{prefix}_max"]    = float(v.max(initial=0.0))

    # Quantiles (robust summaries)
    qs = np.quantile(v, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    for qv, qn in zip(qs, ("q01","q05","q25","q50","q75","q95","q99")):
        feats[f"{prefix}_{qn}"] = float(qv)

    # Sign structure
    pos = v[v > 0]; neg = v[v < 0]
    feats[f"{prefix}_frac_pos"] = float(pos.size / n)
    feats[f"{prefix}_frac_neg"] = float(neg.size / n)
    feats[f"{prefix}_mean_pos"] = float(pos.mean()) if pos.size else 0.0
    feats[f"{prefix}_mean_neg"] = float(neg.mean()) if neg.size else 0.0
    feats[f"{prefix}_l1_pos"]   = float(pos.sum()) if pos.size else 0.0
    feats[f"{prefix}_l1_neg"]   = float((-neg).sum()) if neg.size else 0.0  # absolute of negatives

    # Near-zero sparsity at multiple thresholds
    for eps in (1e-8, 1e-6, 1e-4):
        feats[f"{prefix}_frac_abs_lt_{eps:g}"] = float((av < eps).mean())

    # Gini coefficient (concentration of |v| across tracks)
    if s1 > 0.0:
        y_sorted = np.sort(av)
        i = np.arange(1, n + 1)
        gini = 1.0 + 1.0/n - 2.0 * np.sum((n + 1 - i) * y_sorted) / (n * s1)
    else:
        gini = 0.0
    feats[f"{prefix}_gini_abs"] = float(gini)

    # Entropy of |v| mass (base-2); higher = more diffuse across tracks
    if s1 > 0.0:
        p = av / s1
        nz = p > 0
        H = -float(np.sum(p[nz] * np.log2(p[nz])))
    else:
        H = 0.0
    feats[f"{prefix}_entropy_abs_bits"] = H

    # Tail mass concentration: fraction of |v| in top k% tracks
    for frac in (0.01, 0.05, 0.10):
        k = max(1, int(np.ceil(frac * n)))
        if s1 > 0.0:
            idx = np.argpartition(av, -k)[-k:]
            feats[f"{prefix}_top{int(frac*100)}pct_mass_abs"] = float(av[idx].sum() / s1)
        else:
            feats[f"{prefix}_top{int(frac*100)}pct_mass_abs"] = 0.0

    return feats

def load_vector_nan_safe(path):
    """Load a numpy vector, convert inf to nan, keep dtype float."""
    arr = np.asarray(np.load(path), dtype=np.float64)
    arr[~np.isfinite(arr)] = np.nan  # turn ±inf into NaN
    return arr

def get_phylop_window_mean(bw_obj, chrom, center_pos, window_size, neutral_score=0.0):
    """Return mean phyloP score in ±window_bp region, replacing missing with neutral_score."""
    flank_bp = window_size//2
    start = max(center_pos - flank_bp - 1, 0)  # 0-based inclusive
    end = center_pos + flank_bp               # 1-based exclusive
    vals = bw_obj.values(chrom, start, end)
    clean_vals = [
        v if (v is not None and not (isinstance(v, float) and v != v)) else neutral_score
        for v in vals
    ]
    return np.mean(clean_vals)

def compute_tau(x):
    """Tissue-specificity index τ from Yanai et al., 2005"""
    if np.max(x) == 0:
        return 0.0
    x_norm = x / np.max(x)
    return np.sum(1 - x_norm) / (len(x_norm) - 1)

def compute_gini(x):
    """Compute Gini coefficient"""
    if np.amin(x) < 0:
        x = x - np.amin(x)
    x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * x)) / (n * np.sum(x) + 1e-8)

def _slug(x):
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(x).strip())

def compute_specificity_metrics(x, i, epsilon=1e-8):
    others = np.delete(x, i)
    z = (x[i] - np.mean(others)) / (np.std(others) + epsilon)
    fc = (x[i] + epsilon) / (np.mean(others) + epsilon)
    frac = x[i] / (np.sum(x) + epsilon)
    rel_expr = others / (x[i] + epsilon)
    spec = 1 - np.mean(rel_expr)
    return z, fc, frac, spec


def extract_target_features(gene_id, target, gene_slice, sparsity_threshold=1e-3):
    result = {'gene_id': gene_id}
    eps = 1e-8

    target_all = target  # shape: (num_cells, num_bins)
    target_exons = target[:, gene_slice] if len(gene_slice) > 0 else None

    # Flatten for summary stats
    flat_all = target_all.flatten()

    # --- Basic stats over all bins ---
    result.update({
        'sum_all': np.sum(flat_all),
        'mean_all': np.mean(flat_all),
        'median_all': np.median(flat_all),
        'max_all': np.max(flat_all),
        'min_all': np.min(flat_all),
        'std_all': np.std(flat_all),
        'q25_all': np.percentile(flat_all, 25),
        'q75_all': np.percentile(flat_all, 75),
        'skew_all': skew(flat_all),
        'kurtosis_all': kurtosis(flat_all),
        'gini_all': compute_gini(flat_all),
    })

    # Variance and range across bins (per cell)
    var_all = np.var(target_all, axis=0)
    range_all = np.max(target_all, axis=0) - np.min(target_all, axis=0)
    result.update({
        'var_mean_all': np.mean(var_all),
        'var_max_all': np.max(var_all),
        'range_mean_all': np.mean(range_all),
        'range_max_all': np.max(range_all),
    })

    # Variance across bins (per cell)
    var_cells_all = np.var(target_all, axis=1)
    result.update({
        'cell_variance_mean_all': np.mean(var_cells_all),
        'cell_variance_max_all': np.max(var_cells_all),
    })

    # Entropy
    prob_all = target_all / (np.sum(target_all, axis=0, keepdims=True) + eps)
    entropy_all = entropy(prob_all + eps, axis=0)
    result.update({
        'entropy_mean_all': np.mean(entropy_all),
        'entropy_median_all': np.median(entropy_all),
        'entropy_max_all': np.max(entropy_all),
    })

    # Sparsity
    result['sparsity_all'] = np.mean(flat_all < sparsity_threshold)

    # Peakiness (bins > 95th percentile)
    thr_all = np.percentile(flat_all, 95)
    result['num_peaks_95p_all'] = np.sum(flat_all > thr_all)

    # --- Features over exon bins ---
    if target_exons is not None and target_exons.shape[1] > 0:
        flat_exon = target_exons.flatten()

        result.update({
            'sum_exon': np.sum(flat_exon),
            'mean_exon': np.mean(flat_exon),
            'median_exon': np.median(flat_exon),
            'max_exon': np.max(flat_exon),
            'min_exon': np.min(flat_exon),
            'std_exon': np.std(flat_exon),
            'q25_exon': np.percentile(flat_exon, 25),
            'q75_exon': np.percentile(flat_exon, 75),
            'skew_exon': skew(flat_exon),
            'kurtosis_exon': kurtosis(flat_exon),
            'gini_exon': compute_gini(flat_exon),
        })

        var_exon = np.var(target_exons, axis=0)
        range_exon = np.max(target_exons, axis=0) - np.min(target_exons, axis=0)
        result.update({
            'var_mean_exon': np.mean(var_exon),
            'var_max_exon': np.max(var_exon),
            'range_mean_exon': np.mean(range_exon),
            'range_max_exon': np.max(range_exon),
        })

        var_cells_exon = np.var(target_exons, axis=1)
        result.update({
            'cell_variance_mean_exon': np.mean(var_cells_exon),
            'cell_variance_max_exon': np.max(var_cells_exon),
        })

        prob_exon = target_exons / (np.sum(target_exons, axis=0, keepdims=True) + eps)
        entropy_exon = entropy(prob_exon + eps, axis=0)
        result.update({
            'entropy_mean_exon': np.mean(entropy_exon),
            'entropy_median_exon': np.median(entropy_exon),
            'entropy_max_exon': np.max(entropy_exon),
        })

        result['sparsity_exon'] = np.mean(flat_exon < sparsity_threshold)

        thr_exon = np.percentile(flat_exon, 95)
        result['num_peaks_95p_exon'] = np.sum(flat_exon > thr_exon)

    return result

def extract_celltype_features(target, gene_slice, sparsity_threshold=1e-3):
    result = {}
    num_cells, num_bins = target.shape
    target_exons = target[:, gene_slice] if len(gene_slice) > 0 else None

    # --- Full sequence features per cell type ---
    cell_means_all = np.mean(target, axis=1)
    cell_sums_all = np.sum(target, axis=1)
    cell_max_all = np.max(target, axis=1)
    cell_std_all = np.std(target, axis=1)
    cell_sparsity_all = np.mean(target < sparsity_threshold, axis=1)

    # --- Global cell-type specificity metrics ---
    # Safer CV: std/mean with stabilization to avoid inf/NaN
    result['celltype_cv_all'] = float(np.nanstd(cell_means_all) / (np.nanmean(cell_means_all) + 1e-8))
    result['celltype_tau_all'] = compute_tau(cell_means_all)
    result['celltype_gini_all'] = compute_gini(cell_means_all)
    result['celltype_top1_frac_all'] = np.max(cell_means_all) / (np.sum(cell_means_all) + 1e-8)
    result['celltype_top1_index_all'] = int(np.argmax(cell_means_all))

    for i in range(num_cells):
        result[f'gtex_{i}_mean_all'] = cell_means_all[i]
        result[f'gtex_{i}_sum_all'] = cell_sums_all[i]
        result[f'gtex_{i}_max_all'] = cell_max_all[i]
        result[f'gtex_{i}_std_all'] = cell_std_all[i]
        result[f'gtex_{i}_sparsity_all'] = cell_sparsity_all[i]

        # Specificity of cell i compared to all others
        z, fc, frac, spec = compute_specificity_metrics(cell_means_all, i)
        result[f'gtex_{i}_zscore_all'] = z
        result[f'gtex_{i}_fc_all'] = fc
        result[f'gtex_{i}_frac_all'] = frac
        result[f'gtex_{i}_specidx_all'] = spec

    # --- Exon features per cell type ---
    if target_exons is not None and target_exons.shape[1] > 0:
        cell_means_exon = np.mean(target_exons, axis=1)
        cell_sums_exon = np.sum(target_exons, axis=1)
        cell_max_exon = np.max(target_exons, axis=1)
        cell_std_exon = np.std(target_exons, axis=1)
        cell_sparsity_exon = np.mean(target_exons < sparsity_threshold, axis=1)

        result['celltype_cv_exon'] = float(np.nanstd(cell_means_exon) / (np.nanmean(cell_means_exon) + 1e-8))
        result['celltype_tau_exon'] = compute_tau(cell_means_exon)
        result['celltype_gini_exon'] = compute_gini(cell_means_exon)
        result['celltype_top1_frac_exon'] = np.max(cell_means_exon) / (np.sum(cell_means_exon) + 1e-8)
        result['celltype_top1_index_exon'] = int(np.argmax(cell_means_exon))

        for i in range(num_cells):
            result[f'gtex_{i}_mean_exon'] = cell_means_exon[i]
            result[f'gtex_{i}_sum_exon'] = cell_sums_exon[i]
            result[f'gtex_{i}_max_exon'] = cell_max_exon[i]
            result[f'gtex_{i}_std_exon'] = cell_std_exon[i]
            result[f'gtex_{i}_sparsity_exon'] = cell_sparsity_exon[i]

            # Specificity for exon-only signal
            z, fc, frac, spec = compute_specificity_metrics(cell_means_exon, i)
            result[f'gtex_{i}_zscore_exon'] = z
            result[f'gtex_{i}_fc_exon'] = fc
            result[f'gtex_{i}_frac_exon'] = frac
            result[f'gtex_{i}_specidx_exon'] = spec

    return result

df=pd.read_csv(P['tables']['all_curated'], sep='\t')
print(len(df))
df = df.drop(columns=['rsid'])
df = df.drop_duplicates()
df = df.reset_index(drop=True)

transcriptome = bgene.Transcriptome(P['reference']['gencode_gtf'])
genes = [gene_key for gene_key in transcriptome.genes.keys()]
base_gene_ids = [gene.split('.')[0] for gene in genes]
df=df[df['gene_id'].isin(base_gene_ids)]

borzoi_track_map=pd.read_csv(P['tables']['borzoi_tracks_matched'])
df = df.merge(
    borzoi_track_map[['gtex', 'borzoi_tracks']],
    how='left',
    left_on='target_tissue',
    right_on='gtex'
)
tss_df=pd.read_csv(P['tables']['gene_to_tss'], sep='\t')
df = df.merge(
    tss_df[['gene_id', 'TSS', 'strand']],
    how='left',
    on='gene_id',
)

job_id = int(sys.argv[1])
df=df.iloc[job_id*1000:job_id*1000+1000].reset_index(drop=True)


df_seq = pd.read_csv(P['reference']['sequences_bed_gz'], sep='\t', header=None)
df_seq.columns=['chrom','start','end','fold']
df_seq['split'] = df_seq['fold'].map({
    'fold3': 'test',
    'fold4': 'val'
}).fillna('train')
df_seq_train=df_seq[df_seq['split']=='train']

bw = pyBigWig.open(P['reference']['phylop_bw'])
df_genes = pd.read_csv(P['tables']['all_genes'], sep='\t')
df_pos=df[df['type']=='positive']

# Resolve output / intermediate directories once (used inside the loop below).
SEQFEAT_DIR   = P['outputs']['sequence_features_new']
GENESEQ_DIR   = P['outputs']['gene_sequence_features']
GENEDATA_DIR  = P['outputs']['gene_data_features']
GENEPERF_DIR  = P['outputs']['borzoi_gene_performance']
VARPERF_DIR   = P['outputs']['borzoi_performance']
PRED_DIR      = P['outputs']['borzoi_predictions']
GENEPRED_DIR  = P['outputs']['borzoi_gene_predictions']
FINAL_DIR     = P['outputs']['final_feature_set']

centromere_telomere_info = {
'chr1':  (125000000, 248956422),
'chr2':  (93300000,  242193529),
'chr3':  (91000000,  198295559),
'chr4':  (50400000,  190214555),
'chr5':  (48400000,  181538259),
'chr6':  (61000000,  170805979),
'chr7':  (59900000,  159345973),
'chr8':  (45600000,  145138636),
'chr9':  (49000000,  138394717),
'chr10': (40200000,  133797422),
'chr11': (53700000,  135086622),
'chr12': (35800000,  133275309),
'chr13': (17900000,  114364328),
'chr14': (17600000,  107043718),
'chr15': (19000000,  101991189),
'chr16': (36600000,   90338345),
'chr17': (24000000,   83257441),
'chr18': (17200000,   80373285),
'chr19': (26500000,   58617616),
'chr20': (27500000,   64444167),
'chr21': (13200000,   46709983),
'chr22': (14700000,   50818468),
'chrX':  (60600000,  156040895),
}

for seq_id in range(len(df)):
    try:
        row=df.iloc[seq_id]
        seq_len=524288
        search_gene = row['gene_id']
        center_pos = row['pos']
        chrom = row['chrom']
        variant = row['variant']
        poses = [center_pos]
        start = center_pos - seq_len // 2
        end = center_pos + seq_len // 2
        target_tissue = row['target_tissue']

        filename = f"{_slug(variant)}_{_slug(search_gene)}_{_slug(target_tissue)}.parquet"
        out_dir = Path(FINAL_DIR)
        os.makedirs(out_dir, exist_ok=True)
        out_path = out_dir / filename

        if out_path.exists():
            print(f"Exists — skipping: {out_path}")
            continue

        flanking_bins = 16352-6144
        start_bin=flanking_bins//2
        end_bin=16352-flanking_bins//2

        gene_keys = [gene_key for gene_key in transcriptome.genes.keys() if search_gene in gene_key]
        gene = transcriptome.genes[gene_keys[0]]

        seq_out_start = start + 32*16
        seq_out_len = 32*16352

        sparsity_threshold=1e-3

        gene_slice = gene.output_slice(seq_out_start, seq_out_len, 32, False)
        gene_slice = gene_slice[(gene_slice >= start_bin) & (gene_slice < end_bin)]

        basic_features = {
        'fine_map_pip': float(row['pip']),
        'credible_set_size': float(row['cs_size']),
        'credible_set_min_r2': float(row['cs_min_r2']),
        'eqtl_study_sample_size': float(row['sample_size']),
        'eqtl_pvalue': float(row['pvalue']),
        'eqtl_se': float(row['se']),
        'minor_allele_freq': float(row['maf']),
        'alt_allele_count': int(row['ac']),          # assumes AC = alternate-allele count
        'total_called_alleles': int(row['an']),      # AN
        'minor_allele_sample_count': int(row['ma_samples']),
        'gene_median_tpm': float(row['median_tpm']),
        'gene_tss_position': int(row['TSS']),
        'variant_position': int(row['pos']),
        'gene_num_exons': len(gene_slice),}

        basic_features['variant_to_tss_distance'] = abs(int(row['pos']) - int(row['TSS']))
        basic_features['gene_strand_plus'] = int(row['strand'] == '+')  # 1 for '+', 0 for '-'


        chrom = row['chrom']
        pos_1based = int(row['pos'])

        # Single base score
        phyloP_score = bw.values(chrom, pos_1based - 1, pos_1based)[0]
        if phyloP_score is None or (isinstance(phyloP_score, float) and phyloP_score != phyloP_score):
            phyloP_score = 0.0  # neutral score

        # Surrounding regions
        phyloP_mean_100bp = get_phylop_window_mean(bw, chrom, pos_1based, 100, neutral_score=0.0)
        phyloP_mean_1000bp = get_phylop_window_mean(bw, chrom, pos_1based, 1000, neutral_score=0.0)

        # Add to features
        basic_features['phyloP'] = phyloP_score
        basic_features['phyloP_mean_100bp_window'] = phyloP_mean_100bp
        basic_features['phyloP_mean_1000bp_window'] = phyloP_mean_1000bp

        chr_row = row['chrom']
        pos_row = int(row['pos'])

        chr_series = df_pos['chrom']
        pos_series = df_pos['pos'].astype('int64')

        # ----------------- windows -----------------
        # ±1b
        window_bp = 1
        lo = max(1, pos_row - window_bp)
        hi = pos_row + window_bp
        mask = (chr_series == chr_row) & (pos_series >= lo) & (pos_series <= hi)
        eqtls_1bp = df_pos.loc[mask]

        # ±1 kb
        window_bp = 1_000
        lo = max(1, pos_row - window_bp)
        hi = pos_row + window_bp
        mask = (chr_series == chr_row) & (pos_series >= lo) & (pos_series <= hi)
        eqtls_1kb = df_pos.loc[mask]

        # ±10 kb
        window_bp = 10_000
        lo = max(1, pos_row - window_bp)
        hi = pos_row + window_bp
        mask = (chr_series == chr_row) & (pos_series >= lo) & (pos_series <= hi)
        eqtls_10kb = df_pos.loc[mask]

        # Row counts (all tissues) — as you already had
        n_1bp = len(eqtls_1bp)
        n_1kb = len(eqtls_1kb)
        n_10kb = len(eqtls_10kb)

        # ----------------- variant counts across ALL tissues -----------------
        # unique variants defined by (chrom, pos)
        n_1bp_variants_all  = int(eqtls_1bp[['chrom','pos']].drop_duplicates().shape[0])
        n_1kb_variants_all  = int(eqtls_1kb[['chrom','pos']].drop_duplicates().shape[0])
        n_10kb_variants_all = int(eqtls_10kb[['chrom','pos']].drop_duplicates().shape[0])

        # ----------------- target tissue subsets -----------------
        eqtls_1bp_target  = eqtls_1bp[eqtls_1bp['target_tissue'] == target_tissue]
        eqtls_1kb_target  = eqtls_1kb[eqtls_1kb['target_tissue'] == target_tissue]
        eqtls_10kb_target = eqtls_10kb[eqtls_10kb['target_tissue'] == target_tissue]

        # Row counts within target tissue
        n_1bp_rows_target  = int(len(eqtls_1bp_target))
        n_1kb_rows_target  = int(len(eqtls_1kb_target))
        n_10kb_rows_target = int(len(eqtls_10kb_target))

        # Variant counts within target tissue
        n_1bp_variants_target  = int(eqtls_1bp_target[['chrom','pos']].drop_duplicates().shape[0])
        n_1kb_variants_target  = int(eqtls_1kb_target[['chrom','pos']].drop_duplicates().shape[0])
        n_10kb_variants_target = int(eqtls_10kb_target[['chrom','pos']].drop_duplicates().shape[0])

        # ----------------- attach to features (names chosen to be explicit) -----------------
        basic_features['num_pos_eqtl_all_tissue_1bp_flank_all_genes'] = n_1bp
        basic_features['num_pos_eqtl_all_tissue_1kb_flank_all_genes'] = n_1kb
        basic_features['num_pos_eqtl_all_tissue_10kb_flank_all_genes'] = n_10kb
        basic_features['num_pos_eqtl_all_tissue_1bp_flank_unique_variants'] = n_1bp_variants_all
        basic_features['num_pos_eqtl_all_tissue_1kb_flank_unique_variants'] = n_1kb_variants_all
        basic_features['num_pos_eqtl_all_tissue_10kb_flank_unique_variants'] = n_10kb_variants_all

        basic_features['num_pos_eqtl_target_tissue_1bp_flank_all_genes'] = n_1bp_rows_target
        basic_features['num_pos_eqtl_target_tissue_1kb_flank_all_genes'] = n_1kb_rows_target
        basic_features['num_pos_eqtl_target_tissue_10kb_flank_all_genes'] = n_10kb_rows_target
        basic_features['num_pos_eqtl_target_tissue_1bp_flank_unique_variants'] = n_1bp_variants_target
        basic_features['num_pos_eqtl_target_tissue_1kb_flank_unique_variants'] = n_1kb_variants_target
        basic_features['num_pos_eqtl_target_tissue_10kb_flank_unique_variants'] = n_10kb_variants_target

        print(
            "ALL tissues — rows (1bp, 1kb,10kb):", n_1bp, n_1kb, n_10kb,
            "| variants (1bp, 1kb,10kb):", n_1bp_variants_all, n_1kb_variants_all, n_10kb_variants_all
        )
        print(
            f"TARGET={target_tissue} — rows (1kb,10kb):", n_1kb_rows_target, n_10kb_rows_target,
            "| variants (1kb,10kb):", n_1bp_variants_target, n_1kb_variants_target, n_10kb_variants_target
        )

        chrom = row['chrom']
        pos = int(row['pos'])

        centromere_bp, chrom_len_bp = centromere_telomere_info[chrom]
        dist_centromere_bp = abs(pos - centromere_bp)
        dist_telomere_bp   = min(pos, chrom_len_bp - pos)

        basic_features['variant_to_centromere_distance_bp'] = dist_centromere_bp
        basic_features['variant_to_telomere_distance_bp']   = dist_telomere_bp

        chr_row = row['chrom']
        pos_row = int(row['pos'])

        train_chr = df_seq_train[(df_seq_train['split'] == 'train') & (df_seq_train['chrom'] == chr_row)]
        in_train = ((train_chr['start'] <= pos_row) & (pos_row <= train_chr['end'])).any()
        basic_features['variant_overlaps_any_training_region'] = int(bool(in_train))
        centers = ((train_chr['start'].to_numpy(np.int64) + train_chr['end'].to_numpy(np.int64)) // 2)
        dist_nearest = int(np.min(np.abs(centers - pos_row)))
        basic_features['variant_distance_to_nearest_training_region_center'] = dist_nearest

        file_path = os.path.join(SEQFEAT_DIR, f"{variant}.parquet")
        vf_df = pd.read_parquet(file_path).iloc[:, 5:]
        variant_sequence_features = vf_df.add_prefix('variant_sequence_').iloc[0].to_dict()

        file_path=os.path.join(GENESEQ_DIR, f'{search_gene}.parquet')
        vf_df = pd.read_parquet(file_path)
        gene_sequence_features = vf_df.add_prefix('gene_sequence_').iloc[0].to_dict()

        tracks_to_use=[int(i)-7522 for i in row['borzoi_tracks'].split(';') if i.strip().isdigit()]
        gtex_index=tracks_to_use[0]



        # file_path = os.path.join(GENEDATA_DIR, f"{search_gene}.pkl")
        # with open(file_path, "rb") as file:
        #     gene_data_features = pickle.load(file)
        # gtex_prefix = f"gtex_{gtex_index}"
        # filtered_gene_data_features = {}
        # for key, value in gene_data_features.items():
        #     if key.startswith("gtex_"):
        #         if key.startswith(gtex_prefix):
        #             feature_suffix = key[len(gtex_prefix):]
        #             new_key = f"gtex{feature_suffix}"
        #             filtered_gene_data_features[new_key] = value
        #     else:
        #         filtered_gene_data_features[key] = value
        # keys_to_remove = ["gene_id", "celltype_top1_index_all", "celltype_top1_index_exon"]
        # for key in keys_to_remove:
        #     filtered_gene_data_features.pop(key, None)
        # prefix = "gene_observed_data_"
        # for k in list(filtered_gene_data_features.keys()):
        #     filtered_gene_data_features[prefix+k] = filtered_gene_data_features.pop(k)

        file_path = os.path.join(GENEDATA_DIR, f"{search_gene}.pkl")
        with open(file_path, "rb") as file:
            gene_data_features = pickle.load(file)

        # Keep only gtex_{gtex_index}_* features and drop the index → gtex_*
        pat = re.compile(rf"^gtex_{int(gtex_index)}_(.+)$")

        filtered_gene_data_features = {}
        for key, value in gene_data_features.items():
            m = pat.match(key)
            if m:
                new_key = f"gtex_{m.group(1)}"          # e.g., gtex_13_sum_all → gtex_sum_all
                filtered_gene_data_features[new_key] = value
            elif not key.startswith("gtex_"):           # keep non-gtex features as-is
                filtered_gene_data_features[key] = value

        # Drop unwanted keys
        for key in ["gene_id", "celltype_top1_index_all", "celltype_top1_index_exon"]:
            filtered_gene_data_features.pop(key, None)

        # Add prefix
        prefix = "gene_observed_data_"
        filtered_gene_data_features = {prefix + k: v for k, v in filtered_gene_data_features.items()}


        num_reps = 4
        seq_list = []
        track_list = []

        for rep_ix in range(1, num_reps + 1):
            path_seq = os.path.join(GENEPERF_DIR, f"model_{rep_ix}_{search_gene}_along_sequence.npy")
            path_track = os.path.join(GENEPERF_DIR, f"model_{rep_ix}_{search_gene}_along_track.npy")
            seq_list.append(load_vector_nan_safe(path_seq))
            track_list.append(load_vector_nan_safe(path_track))

        along_sequence = np.nanmean(np.stack(seq_list, axis=0), axis=0)
        along_track = np.nanmean(np.stack(track_list, axis=0), axis=0)
        gene_performance_features = {}

        gene_performance_features['gene_performance_track_mean'] = np.nanmean(along_sequence)
        gene_performance_features['gene_performance_track_std'] = np.nanstd(along_sequence)
        gene_performance_features['gene_performance_track_max'] = np.nanmax(along_sequence)
        gene_performance_features['gene_performance_track_min'] = np.nanmin(along_sequence)
        gene_performance_features['gene_performance_track_median'] = np.nanmedian(along_sequence)
        gene_performance_features['gene_performance_track_p90'] = nanpercentile(along_sequence, 90)
        gene_performance_features['gene_performance_track_above_0.8'] = np.nanmean(along_sequence > 0.8)

        gene_performance_features['gene_performance_seqbin_mean'] = np.nanmean(along_track)
        gene_performance_features['gene_performance_seqbin_std'] = np.nanstd(along_track)
        gene_performance_features['gene_performance_seqbin_max'] = np.nanmax(along_track)
        gene_performance_features['gene_performance_seqbin_min'] = np.nanmin(along_track)
        gene_performance_features['gene_performance_seqbin_median'] = np.nanmedian(along_track)
        gene_performance_features['gene_performance_seqbin_p90'] = nanpercentile(along_track, 90)
        gene_performance_features['gene_performance_seqbin_above_0.8'] = np.nanmean(along_track > 0.8)

        diff = np.diff(along_track)
        diff = diff[np.isfinite(diff)]
        gene_performance_features['gene_performance_seqbin_diff_std'] = np.nanstd(diff) if diff.size else np.nan
        gene_performance_features['gene_performance_seqbin_diff_mean'] = np.nanmean(diff) if diff.size else np.nan

        all_cell_types = along_sequence[7522:7611]
        other_cell_types = np.delete(all_cell_types, gtex_index)

        gene_performance_features['gene_performance_cell_type_performance'] = along_sequence[7522+gtex_index]
        gene_performance_features['gene_performance_celltype_mean'] = np.nanmean(all_cell_types)
        gene_performance_features['gene_performance_celltype_std'] = np.nanstd(all_cell_types)
        gene_performance_features['gene_performance_celltype_diff_from_mean'] = (
            gene_performance_features['gene_performance_cell_type_performance'] - gene_performance_features['gene_performance_celltype_mean']
        )
        gene_performance_features['gene_performance_celltype_zscore'] = (
            (gene_performance_features['gene_performance_cell_type_performance'] - gene_performance_features['gene_performance_celltype_mean']) /
            (gene_performance_features['gene_performance_celltype_std'] if gene_performance_features['gene_performance_celltype_std'] != 0 else np.nan)
        )
        gene_performance_features['gene_performance_celltype_diff_from_others_mean'] = (
            gene_performance_features['gene_performance_cell_type_performance'] - np.nanmean(other_cell_types)
        )

        num_reps=4
        seq_list = []
        track_list = []

        for rep_ix in range(1, num_reps + 1):
            path_seq = os.path.join(VARPERF_DIR, f"model_{rep_ix}_{variant}_along_sequence.npy")
            path_track = os.path.join(VARPERF_DIR, f"model_{rep_ix}_{variant}_along_track.npy")

            seq_list.append(np.load(path_seq))
            track_list.append(np.load(path_track))

        along_sequence = np.nanmean(np.stack(seq_list, axis=0), axis=0)
        along_track = np.nanmean(np.stack(track_list, axis=0), axis=0)

        performance_features = {}

        performance_features['track_mean'] = float(np.nanmean(along_sequence))
        performance_features['track_std'] = float(np.nanstd(along_sequence))
        performance_features['track_max'] = float(np.nanmax(along_sequence))
        performance_features['track_min'] = float(np.nanmin(along_sequence))
        performance_features['track_median'] = float(np.nanmedian(along_sequence))
        performance_features['track_p90'] = float(nanpercentile(along_sequence, 90))
        performance_features['track_above_0.8'] = float(np.nanmean(along_sequence > 0.8))

        performance_features['seqbin_mean'] = float(np.nanmean(along_track))
        performance_features['seqbin_std'] = float(np.nanstd(along_track))
        performance_features['seqbin_max'] = float(np.nanmax(along_track))
        performance_features['seqbin_min'] = float(np.nanmin(along_track))
        performance_features['seqbin_median'] = float(np.nanmedian(along_track))
        performance_features['seqbin_p90'] = float(nanpercentile(along_track, 90))
        performance_features['seqbin_above_0.8'] = float(np.nanmean(along_track > 0.8))

        diff = np.diff(along_track)
        diff = diff[np.isfinite(diff)]
        performance_features['seqbin_diff_std'] = float(np.nanstd(diff)) if diff.size else np.nan
        performance_features['seqbin_diff_mean'] = float(np.nanmean(diff)) if diff.size else np.nan

        all_cell_types = along_sequence[7522:7611]  # inclusive of 7610
        other_cell_types = np.delete(all_cell_types, gtex_index)
        performance_features['cell_type_performance']= along_sequence[7522+gtex_index]
        performance_features['celltype_mean'] = float(np.nanmean(all_cell_types))
        performance_features['celltype_std'] = float(np.nanstd(all_cell_types))
        performance_features['celltype_diff_from_mean'] = (
            performance_features['cell_type_performance'] - performance_features['celltype_mean']
        )
        performance_features['celltype_zscore'] = (
            (performance_features['cell_type_performance'] - performance_features['celltype_mean']) /
            (performance_features['celltype_std'] if performance_features['celltype_std'] != 0 else np.nan)
        )

        performance_features['celltype_diff_from_others_mean'] = (
            performance_features['cell_type_performance'] - other_cell_types.mean()
        )
        prefix = "variant_performance_"
        for k in list(performance_features.keys()):
            performance_features[prefix+k] = performance_features.pop(k)

        NUM_REPLICATES = 4
        ref_list, alt_list = [], []

        for rep_ix in range(1, NUM_REPLICATES + 1):
            ref_pred = np.load(os.path.join(PRED_DIR, f'model_{rep_ix}_ref__{variant}.npy'))[0][0]
            alt_pred = np.load(os.path.join(PRED_DIR, f'model_{rep_ix}_alt__{variant}.npy'))[0][0]
            
            ref_pred = ref_pred.reshape(-1, 89).astype(np.float32).T  # shape: (89, bins)
            alt_pred = alt_pred.reshape(-1, 89).astype(np.float32).T
            
            ref_list.append(ref_pred[:, start_bin:end_bin])  # crop to central region
            alt_list.append(alt_pred[:, start_bin:end_bin])

        ref_all = np.stack(ref_list, axis=0)
        alt_all = np.stack(alt_list, axis=0)

        ref = np.mean(ref_all, axis=0)
        alt = np.mean(alt_all, axis=0)

        l1_diff = np.abs(alt - ref)
        d1 = np.mean(np.sum(l1_diff, axis=1))
        l2_diff = alt - ref
        d2 = np.mean(np.linalg.norm(l2_diff, ord=2, axis=1))

        max_diff_indices = np.argmax(np.abs(alt - ref), axis=1)
        sax = np.mean([alt[i, idx] - ref[i, idx] for i, idx in enumerate(max_diff_indices)])

        # Uncertainty metrics

        features={}
        ref_std = np.std(ref_all, axis=0)
        alt_std = np.std(alt_all, axis=0)
        delta_all = alt_all - ref_all
        delta_std = np.std(delta_all, axis=0)

        # === Standard deviation-based statistics ===
        features["ref_prediction_std"] = float(np.mean(ref_std))
        features["alt_prediction_std"] = float(np.mean(alt_std))
        features["delta_prediction_std"] = float(np.mean(delta_std))

        features["ref_prediction_std_max"] = float(np.max(ref_std))
        features["alt_prediction_std_max"] = float(np.max(alt_std))
        features["delta_prediction_std_max"] = float(np.max(delta_std))

        features["ref_prediction_std_std"] = float(np.std(ref_std))
        features["alt_prediction_std_std"] = float(np.std(alt_std))
        features["delta_prediction_std_std"] = float(np.std(delta_std))

        # === Coefficient of variation (CV) ===
        ref_mean = np.mean(ref_all, axis=0)
        alt_mean = np.mean(alt_all, axis=0)
        delta_mean = np.mean(delta_all, axis=0)

        ref_cv = np.mean(ref_std / (ref_mean + 1e-6))
        alt_cv = np.mean(alt_std / (alt_mean + 1e-6))
        delta_cv = np.mean(delta_std / (np.abs(delta_mean) + 1e-6))

        features["ref_prediction_cv"] = float(ref_cv)
        features["alt_prediction_cv"] = float(alt_cv)
        features["delta_prediction_cv"] = float(delta_cv)

        # === Agreement fraction (bins with low replicate variance) ===
        epsilon = 0.05
        features["ref_prediction_agreement_frac"] = float(np.mean(ref_std < epsilon))
        features["alt_prediction_agreement_frac"] = float(np.mean(alt_std < epsilon))

        # === Entropy across replicates ===
        # Apply per cell type (axis=0), then average over cell types
        ref_entropy = np.mean([entropy(ref_all[:, i, :], base=2).mean() for i in range(ref_all.shape[1])])
        alt_entropy = np.mean([entropy(alt_all[:, i, :], base=2).mean() for i in range(alt_all.shape[1])])

        features["ref_prediction_entropy"] = float(ref_entropy)
        features["alt_prediction_entropy"] = float(alt_entropy)

        ref_list, alt_list = [], []
        for rep_ix in range(1, NUM_REPLICATES + 1):
            ref_pred = np.load(os.path.join(PRED_DIR, f'model_{rep_ix}_ref__{variant}.npy'))[0][0]
            alt_pred = np.load(os.path.join(PRED_DIR, f'model_{rep_ix}_alt__{variant}.npy'))[0][0]
            
            ref_pred = ref_pred.reshape(-1, 89).astype(np.float32).T  # shape: (89, bins)
            alt_pred = alt_pred.reshape(-1, 89).astype(np.float32).T
            
            ref_list.append(ref_pred)  # crop to central region
            alt_list.append(alt_pred)

        # Stack predictions: shape = (num_reps, 89, 6144)
        ref_all = np.stack(ref_list, axis=0)
        alt_all = np.stack(alt_list, axis=0)

        # Compute averages across replicates
        ref = np.mean(ref_all, axis=0)
        alt = np.mean(alt_all, axis=0)

        # Gene slice computations
        ref_gene = ref[:, gene_slice]
        alt_gene = alt[:, gene_slice]
        ref_means = np.mean(ref_gene, axis=1)
        alt_means = np.mean(alt_gene, axis=1)
        delta_means = alt_means - ref_means

        delta_tau = compute_tau(np.abs(delta_means))
        delta_gini = compute_gini(np.abs(delta_means))
        delta_cv = variation(delta_means)
        delta_top1_frac = np.max(np.abs(delta_means)) / (np.sum(np.abs(delta_means)) + 1e-8)

        # Track subset
        tracks_to_use = [int(i) - 7522 for i in row['borzoi_tracks'].split(';') if i.strip().isdigit()]
        gtex_index = tracks_to_use[0]

        alt_preds = np.mean(alt[tracks_to_use, :][:, gene_slice], axis=0).reshape(-1, 1)
        ref_preds = np.mean(ref[tracks_to_use, :][:, gene_slice], axis=0).reshape(-1, 1)

        ref_log = np.log2(ref_preds + 1)
        alt_log = np.log2(alt_preds + 1)
        ref_sqrt = np.sqrt(ref_preds)
        alt_sqrt = np.sqrt(alt_preds)

        # Aggregate over bins
        ref_sum = np.sum(ref_preds)
        alt_sum = np.sum(alt_preds)
        log_ref_sum = np.sum(ref_log)
        log_alt_sum = np.sum(alt_log)
        sqrt_ref_sum = np.sum(ref_sqrt)
        sqrt_alt_sum = np.sum(alt_sqrt)

        log_ref_sed = np.mean(np.log2(np.sum(ref_preds, axis=1) + 1))
        log_alt_sed = np.mean(np.log2(np.sum(alt_preds, axis=1) + 1))

        # Main features
        features.update({
            "delta_l1_mean": float(d1),
            "delta_l2_mean": float(d2),
            "delta_sax": float(sax),
            "absdelta_tau": float(delta_tau),
            "absdelta_gini": float(delta_gini),
            "delta_cv": float(delta_cv),
            "absdelta_top1_frac": float(delta_top1_frac),
            "ref_sum": float(ref_sum),
            "alt_sum": float(alt_sum),
            "ref_logsum": float(log_ref_sum),
            "alt_logsum": float(log_alt_sum),
            "ref_sqrtsum": float(sqrt_ref_sum),
            "alt_sqrtsum": float(sqrt_alt_sum),
            "ref_logsed": float(log_ref_sed),
            "alt_logsed": float(log_alt_sed),
            "delta_sum": float(alt_sum - ref_sum),
            "delta_logsum": float(log_alt_sum - log_ref_sum),
            "delta_sqrtsum": float(sqrt_alt_sum - sqrt_ref_sum),
            "delta_logsed": float(log_alt_sed - log_ref_sed),
        })

        # Extract cell-type and target-level features using averaged predictions
        features_ref_target = extract_target_features(search_gene, ref, gene_slice, sparsity_threshold)
        features_ref_celltype = extract_celltype_features(ref, gene_slice, sparsity_threshold)

        combined_features = {
            **features_ref_target,
            **features_ref_celltype,
        }
        # gtex_prefix = f"gtex_{gtex_index}"
        # filtered_features = {
        #     key: value for key, value in combined_features.items()
        #     if not key.startswith("gtex_") or key.startswith(gtex_prefix)
        # }
        # ref_features_named = {f"ref_{key}": value for key, value in filtered_features.items()}
        pat = re.compile(rf"^gtex_{int(gtex_index)}_(.+)$")
        filtered_features = {}
        for key, value in combined_features.items():
            m = pat.match(key)
            if m:
                new_key = f"gtex_{m.group(1)}"          # e.g., gtex_13_sum_all → gtex_sum_all
                filtered_features[new_key] = value
            elif not key.startswith("gtex_"):           # keep non-gtex features as-is
                filtered_features[key] = value
        ref_features_named = {f"ref_{key}": value for key, value in filtered_features.items()}

        features_alt_target = extract_target_features(search_gene, alt, gene_slice, sparsity_threshold)
        features_alt_celltype = extract_celltype_features(alt, gene_slice, sparsity_threshold)

        combined_features = {
            **features_alt_target,
            **features_alt_celltype
        }
        # filtered_features = {
        #     key: value for key, value in combined_features.items()
        #     if not key.startswith("gtex_") or key.startswith(gtex_prefix)
        # }
        # alt_features_named = {f"alt_{key}": value for key, value in filtered_features.items()}
        pat = re.compile(rf"^gtex_{int(gtex_index)}_(.+)$")
        filtered_features = {}
        for key, value in combined_features.items():
            m = pat.match(key)
            if m:
                new_key = f"gtex_{m.group(1)}"          # e.g., gtex_13_sum_all → gtex_sum_all
                filtered_features[new_key] = value
            elif not key.startswith("gtex_"):           # keep non-gtex features as-is
                filtered_features[key] = value
                
        alt_features_named = {f"alt_{key}": value for key, value in filtered_features.items()}

        # Final feature aggregation
        model_prediction_variant_features = {
            **features,
            **ref_features_named,
            **alt_features_named
        }
        keys_to_remove = ["ref_gene_id", "alt_gene_id"]
        for key in keys_to_remove:
            model_prediction_variant_features.pop(key, None)

        # for key in list(model_prediction_variant_features.keys()):
        #     if ("gtex_" in key) and (gtex_prefix not in key):
        #         continue
        #     new_key = re.sub(r'(?<=gtex_)\d+_', '', key)
        #     if new_key != key:
        #         model_prediction_variant_features[new_key] = model_prediction_variant_features.pop(key)
        prefix = "model_predictions_"
        for k in list(model_prediction_variant_features.keys()):
            model_prediction_variant_features[prefix+k] = model_prediction_variant_features.pop(k)


        log_feats = []
        sum_feats = []

        for rep_ix in range(1, 5):  # Models 1 to 4
            log_path = os.path.join(PRED_DIR, f"model_{rep_ix}_log_score__{variant}.npy")
            sum_path = os.path.join(PRED_DIR, f"model_{rep_ix}_sum_score_{variant}.npy")

            log_feat = np.load(log_path)  # shape: (7611,)
            sum_feat = np.load(sum_path)  # shape: (7611,)

            log_feats.append(log_feat)
            sum_feats.append(sum_feat)

        avg_log_feat = np.nanmean(log_feats, axis=0)
        avg_sum_feat = np.nanmean(sum_feats, axis=0)

        x = np.asarray(avg_sum_feat, dtype=float)
        y = np.asarray(avg_log_feat, dtype=float)
        n = x.size

        model_prediction_diff_all_tracks_feats = {}
        model_prediction_diff_all_tracks_feats.update(summarize_vector(x, "sum"))
        model_prediction_diff_all_tracks_feats.update(summarize_vector(y, "log"))

        xc = x - np.nanmean(x); yc = y - np.nanmean(y)
        den_corr = np.linalg.norm(xc) * np.linalg.norm(yc)
        model_prediction_diff_all_tracks_feats["cross_pearson"] = float(np.dot(xc, yc) / den_corr) if den_corr > 0 else 0.0

        den_cos = np.linalg.norm(x) * np.linalg.norm(y)
        model_prediction_diff_all_tracks_feats["cross_cosine"] = float(np.dot(x, y) / den_cos) if den_cos > 0 else 0.0

        # Sign agreement over non-zero entries in either vector
        sx = np.sign(x); sy = np.sign(y)
        mask_nz = (sx != 0) | (sy != 0)
        model_prediction_diff_all_tracks_feats["cross_frac_same_sign"] = float((sx[mask_nz] == sy[mask_nz]).mean()) if mask_nz.any() else 1.0

        # Scale ratios (guarded)
        model_prediction_diff_all_tracks_feats["cross_l1_ratio_sum_over_log"] = float(np.sum(np.abs(x)) / (np.sum(np.abs(y)) + 1e-12))
        model_prediction_diff_all_tracks_feats["cross_l2_ratio_sum_over_log"] = float(np.linalg.norm(x) / (np.linalg.norm(y) + 1e-12))

        prefix = "model_predictions_difference_all_tracks_"
        for k in list(model_prediction_diff_all_tracks_feats.keys()):
            model_prediction_diff_all_tracks_feats[prefix+k] = model_prediction_diff_all_tracks_feats.pop(k)

        seq_len = 524288
        gene_id = search_gene
        matches = df_genes[df_genes['gene_id'] == gene_id]
        if matches.empty:
            raise ValueError(f"gene_id '{gene_id}' not found in all_genes.csv")
        gene_row = matches.iloc[0]  # Series
        gene_chrom = gene_row['chrom']
        gene_center_pos = int(gene_row['center'])

        gene_start = int(gene_center_pos - seq_len // 2)
        gene_end   = int(gene_center_pos + seq_len // 2)


        flanking_bins = 16352 - 6144
        start_bin = flanking_bins // 2
        end_bin   = 16352 - flanking_bins // 2  # exclusive upper bound

        gene_keys = [k for k in transcriptome.genes.keys() if gene_id in k]
        if len(gene_keys) == 0:
            raise ValueError(f"No transcriptome gene key contains '{gene_id}'")
        gene_obj = transcriptome.genes[gene_keys[0]]

        seq_out_start = int(gene_start + 32 * 16)
        seq_out_len   = int(32 * 16352)

        gene_slice = gene_obj.output_slice(seq_out_start, seq_out_len, 32, False)
        gene_slice = gene_slice[(gene_slice >= start_bin) & (gene_slice < end_bin)]

        NUM_REPLICATES = 4
        gene_list = []

        for rep_ix in range(1, NUM_REPLICATES + 1):
            arr = np.load(os.path.join(GENEPRED_DIR, f'model_{rep_ix}_{gene_id}.npy'))[0][0]
            arr = arr.reshape(-1, 89).astype(np.float32).T  # (89, bins)
            arr[~np.isfinite(arr)] = np.nan                 # sanitize non-finite
            gene_list.append(arr)

        gene_all = np.stack(gene_list, axis=0)
        gene_pred = np.nanmean(gene_all, axis=0)

        gene_std = np.nanstd(gene_all, axis=0)       # (89, bins)
        gene_mean_rep = np.nanmean(gene_all, axis=0) # (89, bins)

        features_gene = {}
        features_gene["gene_prediction_std"] = float(np.nanmean(gene_std))
        features_gene["gene_prediction_std_max"] = float(np.nanmax(gene_std))
        features_gene["gene_prediction_std_std"] = float(np.nanstd(gene_std))

        # CV with stabilization
        features_gene["gene_prediction_cv"] = float(np.nanmean(gene_std / (np.abs(gene_mean_rep) + 1e-6)))

        # Agreement fraction (low replicate variance)
        epsilon = 0.05
        features_gene["gene_prediction_agreement_frac"] = float(np.nanmean(gene_std < epsilon))

        # Entropy across replicates per track, then average across tracks
        track_entropies = []
        for i in range(gene_all.shape[1]):  # 89 tracks
            X = gene_all[:, i, :]  # (num_reps, bins)
            X = np.where(np.isfinite(X), X, np.nan)
            # shift to nonnegative
            X_min = np.nanmin(X, axis=0, keepdims=True)
            X_shift = X - np.where(np.isfinite(X_min), X_min, 0.0)
            X_shift = np.where(np.isfinite(X_shift), X_shift, 0.0)
            # normalize across replicates for each bin (distribution over replicates)
            col_sums = np.sum(X_shift, axis=0, keepdims=True) + 1e-12
            P = X_shift / col_sums
            track_entropies.append(float(np.nanmean(entropy(P, base=2, axis=0))))
        features_gene["gene_prediction_entropy"] = float(np.nanmean(track_entropies))

        gtex_index = tracks_to_use[0]

        # Mean over selected tracks, then restrict to gene_slice → (bins, 1)
        gene_preds = np.nanmean(gene_pred[tracks_to_use, :][:, gene_slice], axis=0).reshape(-1, 1)
        gene_preds[~np.isfinite(gene_preds)] = np.nan

        # Transforms
        gene_log  = np.log2(np.clip(gene_preds, a_min=0.0, a_max=None) + 1.0)
        gene_sqrt = np.sqrt(np.clip(gene_preds, a_min=0.0, a_max=None))

        # Aggregate over bins
        gene_sum     = np.nansum(gene_preds)
        gene_logsum  = np.nansum(gene_log)
        gene_sqrtsum = np.nansum(gene_sqrt)
        gene_logsed  = float(np.log2(np.nansum(gene_preds, axis=0) + 1.0).mean())

        # Position of maximum within gene_slice
        gene_max_pos = int(np.nanargmax(gene_preds)) if np.isfinite(gene_preds).any() else np.nan

        # Descriptive stats
        features_gene["gene_sum"]      = float(gene_sum)
        features_gene["gene_logsum"]   = float(gene_logsum)
        features_gene["gene_sqrtsum"]  = float(gene_sqrtsum)
        features_gene["gene_logsed"]   = float(gene_logsed)
        features_gene["gene_mean"]     = float(np.nanmean(gene_preds))
        features_gene["gene_std"]      = float(np.nanstd(gene_preds))
        features_gene["gene_min"]      = float(np.nanmin(gene_preds))
        features_gene["gene_max"]      = float(np.nanmax(gene_preds))
        features_gene["gene_median"]   = float(np.nanmedian(gene_preds))
        features_gene["gene_p90"]      = float(np.nanpercentile(gene_preds, 90))
        features_gene["gene_above_0.8"]= float(np.nanmean(gene_preds > 0.8))

        # -------------------------
        # Track-level concentration metrics over gene_slice
        # -------------------------
        track_means = np.nanmean(gene_pred[:, gene_slice], axis=1)  # (89,)
        track_means = np.where(np.isfinite(track_means), np.abs(track_means), np.nan)

        # Tau (Yanai)
        if np.all(np.isnan(track_means)) or np.nanmax(track_means) <= 0:
            features_gene["gene_tau"] = np.nan
        else:
            m = np.nanmax(track_means)
            n = int(np.sum(~np.isnan(track_means)))
            tau_num = np.nansum(1.0 - (track_means / (m + 1e-12)))
            features_gene["gene_tau"] = float(tau_num / max(n - 1, 1))

        # Gini
        x = np.sort(track_means[np.isfinite(track_means)])
        if x.size == 0 or np.nanmean(x) == 0:
            features_gene["gene_gini"] = np.nan
        else:
            n = x.size
            cum = np.cumsum(x)
            gini = (n + 1 - 2 * np.sum(cum / cum[-1])) / n
            features_gene["gene_gini"] = float(gini)

        # CV and Top-1 fraction over track_means
        features_gene["gene_cv"] = float(np.nanstd(track_means) / (np.nanmean(track_means) + 1e-6))
        abs_sum = np.nansum(track_means)
        features_gene["gene_top1_frac"] = float((np.nanmax(track_means) / (abs_sum + 1e-8)) if abs_sum > 0 else np.nan)

        features_gene_target = extract_target_features(search_gene, gene_pred, gene_slice, sparsity_threshold)
        features_gene_celltype = extract_celltype_features(gene_pred, gene_slice, sparsity_threshold)

        model_predicted_gene_features = {
            **features_gene_target,
            **features_gene_celltype,
            **features_gene
        }

        # model_predicted_gene_features = {
        #     key: value for key, value in model_predicted_gene_features.items()
        #     if not key.startswith("gtex_") or key.startswith(gtex_prefix)
        # }
        pat = re.compile(rf"^gtex_{int(gtex_index)}_(.+)$")
        filtered_features = {}
        for key, value in model_predicted_gene_features.items():
            m = pat.match(key)
            if m:
                new_key = f"gtex_{m.group(1)}"          # e.g., gtex_13_sum_all → gtex_sum_all
                filtered_features[new_key] = value
            elif not key.startswith("gtex_"):           # keep non-gtex features as-is
                filtered_features[key] = value
        model_predicted_gene_features = {f"model_predictions_gene_{key}": value for key, value in filtered_features.items()}

        # for key in list(model_predicted_gene_features.keys()):
        #     if ("gtex_" in key) and (gtex_prefix not in key):
        #         continue
        #     new_key = re.sub(r'(?<=gtex_)\d+_', '', key)
        #     if new_key != key:
        #         model_predicted_gene_features[new_key] = model_predicted_gene_features.pop(key)

        keys_to_remove = ["model_predictions_gene_gene_id"]
        for key in keys_to_remove:
            model_predicted_gene_features.pop(key, None)

        dicts = [
            basic_features,
            variant_sequence_features,
            gene_sequence_features,
            filtered_gene_data_features,
            gene_performance_features,
            performance_features,
            model_prediction_variant_features,
            model_prediction_diff_all_tracks_feats,
            model_predicted_gene_features,
        ]

        all_keys = [k for d in dicts for k in d.keys()]
        dups = [k for k, c in Counter(all_keys).items() if c > 1]
        if dups:
            raise ValueError(f"Duplicate keys across dictionaries (n={len(dups)}), e.g.: {dups[:10]}")

        # 3) Combine into a single mapping
        combined = {}
        for d in dicts:
            combined.update(d)  # safe because we asserted no duplicates

        # 4) Build a one-row DataFrame (sorted columns for reproducibility)
        df_out = pd.DataFrame([combined])
        df_out = df_out.replace([np.inf, -np.inf], np.nan)
        df_out = df_out.reindex(sorted(df_out.columns), axis=1)

        # 4b) Ensure a stable column schema across files to avoid NaNs when aggregating later
        schema_path = out_dir / "_schema.json"
        if schema_path.exists():
            try:
                with open(schema_path, "r") as f:
                    existing_schema = json.load(f)
                # Fill any structurally missing columns with 0
                missing_cols = [c for c in existing_schema if c not in df_out.columns]
                if missing_cols:
                    for c in missing_cols:
                        df_out[c] = 0
                # Reorder to match schema first, then append any new columns deterministically
                extra_cols = [c for c in df_out.columns if c not in existing_schema]
                ordered_cols = list(existing_schema) + sorted(extra_cols)
                df_out = df_out.reindex(columns=ordered_cols)
                # If new columns appeared, update schema on disk
                if extra_cols:
                    with open(schema_path, "w") as f:
                        json.dump(ordered_cols, f)
            except Exception:
                # If schema read fails for any reason, fall back to writing current schema
                with open(schema_path, "w") as f:
                    json.dump(list(df_out.columns), f)
        else:
            # First file in this directory establishes the schema
            with open(schema_path, "w") as f:
                json.dump(list(df_out.columns), f)

        # 5) Safe file name: variant_gene_target_tissue.parquet
        filename = f"{_slug(variant)}_{_slug(search_gene)}_{_slug(target_tissue)}.parquet"

        # 5b) NaN audit: store a compact report of any NaN columns to help debugging
        try:
            nan_mask = df_out.isna().iloc[0]
            if nan_mask.any():
                nan_cols = [c for c, is_nan in nan_mask.items() if is_nan]
                report = {
                    "variant": variant,
                    "gene_id": search_gene,
                    "target_tissue": target_tissue,
                    "num_nan_cols": int(len(nan_cols)),
                    "nan_columns": nan_cols[:200]  # cap to avoid huge files
                }
                with open(out_dir / f"{_slug(variant)}_{_slug(search_gene)}_{_slug(target_tissue)}.nan_report.json", "w") as f:
                    json.dump(report, f)
        except Exception:
            pass

        df_out.to_parquet(out_path, index=False)

    except:
        pass