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
from tqdm import tqdm
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from grely.paths import load_paths
P = load_paths()
sys.path.append(P['roots']['borzoi_examples'])
from baskerville import dna
from baskerville import gene as bgene
from baskerville import seqnn
from borzoi_helpers import *
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

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

def compute_specificity_metrics(x, i, epsilon=1e-8):
    others = np.delete(x, i)
    z = (x[i] - np.mean(others)) / (np.std(others) + epsilon)
    fc = (x[i] + epsilon) / (np.mean(others) + epsilon)
    frac = x[i] / (np.sum(x) + epsilon)
    rel_expr = others / (x[i] + epsilon)
    spec = 1 - np.mean(rel_expr)
    return z, fc, frac, spec

feature_map = {
    "target": tf.io.FixedLenFeature([], tf.string),
    "sequence": tf.io.FixedLenFeature([], tf.string),
}


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
    result['celltype_cv_all'] = variation(cell_means_all)
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

        result['celltype_cv_exon'] = variation(cell_means_exon)
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

file_path = P['reference']['borzoi_sequences_bed']
seq_mappingF = pd.read_csv(file_path, sep='\t', header=None, names=['chrom', 'start', 'end', 'fold'])
seq_mappingF['tf_label'] = seq_mappingF.groupby('fold').cumcount() // 16
seq_mappingF['tfrecord'] = seq_mappingF.apply(lambda row: f"{row['fold']}-{row['tf_label']}.tfr", axis=1)
seq_mappingF['example_number'] = seq_mappingF.groupby('fold').cumcount() % 16
transcriptome = bgene.Transcriptome(P['reference']['gencode_gtf'])

gene_feat_dir = P['outputs']['gene_data_features']
os.makedirs(gene_feat_dir, exist_ok=True)
df=pd.read_csv(P['tables']['all_genes'], sep='\t')
job_id=int(sys.argv[1])
df=df.iloc[job_id*100:job_id*100+100].reset_index(drop=True)

for seq_id in tqdm(range(len(df))):
    try:
        gene_id=df.iloc[seq_id]['gene_id']
        chrom = df.iloc[seq_id]['chrom']
        query_start = df.iloc[seq_id]['union_start']
        query_end = df.iloc[seq_id]['union_end']
        query_center = (query_start + query_end) // 2
        
        query_interval = pd.DataFrame({
            'chrom': [chrom],
            'start': [query_start],
            'end': [query_end],
            'strand': ['+'],
        })
        
        overlapping_intervals = bf.overlap(query_interval, seq_mappingF, how='inner', suffixes=('_input', ''))
        
        overlapping_intervals['overlap_start'] = overlapping_intervals[['start', 'start_input']].max(axis=1)
        overlapping_intervals['overlap_end'] = overlapping_intervals[['end', 'end_input']].min(axis=1)
        overlapping_intervals['overlap_length'] = (overlapping_intervals['overlap_end'] - overlapping_intervals['overlap_start']).clip(lower=0)
        
        overlapping_intervals['interval_center'] = ((overlapping_intervals['start'] + overlapping_intervals['end']) // 2)
        overlapping_intervals['distance_to_center'] = (overlapping_intervals['interval_center'] - query_center).abs()
        
        valid = overlapping_intervals[overlapping_intervals['overlap_length'] > 0]
        
        selected_record = valid.sort_values(by=['overlap_length', 'distance_to_center'], ascending=[False, True]).iloc[0]
        
        feature_map = {
            "target": tf.io.FixedLenFeature([], tf.string),
            "sequence": tf.io.FixedLenFeature([], tf.string),
        }
        
        dataset = tf.data.TFRecordDataset(f"{P['reference']['borzoi_tfrecords']}/{selected_record.tfrecord}", compression_type="ZLIB")
        i = selected_record.example_number
        
        counter = 0
        
        for raw_record in dataset:
            parsed_record = tf.io.parse_example(raw_record, feature_map)
            sequence = tf.io.decode_raw(parsed_record["sequence"], tf.uint8)
            if counter == i:
                break
            counter += 1
        
        t= tf.io.decode_raw(parsed_record["target"], tf.float16)
        target = tf.reshape(t, (-1, 7611)).numpy().astype(np.float32).T[7522:,:]

        gene_keys = [gk for gk in transcriptome.genes.keys() if gene_id in gk]
        if not gene_keys:
            continue
        gene = transcriptome.genes[gene_keys[0]]

        seq_len = 6144 * 32
        center_pos = query_center
        start = center_pos - seq_len // 2
        seq_out_start = start + 32 * 16
        seq_out_len = 32 * 6144
        gene_slice = gene.output_slice(seq_out_start, seq_out_len, 32, False)

        features = extract_target_features(gene_id, target, gene_slice)
        celltype_features = extract_celltype_features(target, gene_slice)
        features.update(celltype_features)

        with open(os.path.join(gene_feat_dir, f"{gene_id}.pkl"), 'wb') as f:
            pickle.dump(features, f)    
    
    except Exception as e:
        print(f"Skipping {gene_id} due to error: {e}")
        continue