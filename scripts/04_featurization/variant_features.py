import pyfaidx
import pyBigWig
import numpy as np
from Bio.SeqUtils import gc_fraction
from collections import defaultdict, Counter
from tqdm import tqdm
import pickle
import pandas as pd
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from grely.paths import load_paths
P = load_paths()

def get_centered_sequence(fasta, chrom, pos, window_size):
    half = window_size // 2
    start = pos - half
    end = pos + half + 1
    seq = fasta[chrom][start:end].seq
    return seq.upper()

def get_alt_sequence(ref_seq, alt):
    k = len(ref_seq)
    center = k // 2
    return ref_seq[:center] + alt.upper() + ref_seq[center+1:]

def gc_content(seq):
    return gc_fraction(seq)
    
def count_mer(mer, df, fasta):
    mer = mer.upper()
    k = len(mer)
    count = 0

    for row in tqdm(df.itertuples(index=False), total=len(df)):
        chrom = row.chrom
        start = int(row.start)
        end = int(row.end)
        seq = fasta[chrom][start:end].seq.upper()
        if 'N' not in mer:
            count += seq.count(mer)
    return count


def extract_bigwig_features(bw_path, chrom, pos, window=500):
    flank = window // 2
    bw = pyBigWig.open(bw_path)
    try:
        values = bw.values(chrom, pos - flank, pos + flank)
        values = np.array([v for v in values if v is not None])
    except RuntimeError:
        values = np.array([])

    bw.close()

    if len(values) == 0:
        return {
            'mean': float('nan'),
            'median': float('nan'),
            'max': float('nan'),
            'min': float('nan'),
            'std': float('nan'),
            'q25': float('nan'),
            'q75': float('nan'),
            'nonzero_frac': float('nan'),
            'nan_frac': 1.0,
            'sum': float('nan')
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
        'sum': float(np.sum(values))
    }

fasta = pyfaidx.Fasta(P['reference']['fasta'])

seqs=pd.read_csv(P['tables']['all_sequences'], sep='\t')
job_id=int(sys.argv[1])
seqs=seqs.iloc[job_id*100:job_id*100+100].reset_index(drop=True)

df = pd.read_csv(P['reference']['sequences_bed_gz'], sep='\t', header=None)
df.columns=['chrom','start','end','fold']
df['split'] = df['fold'].map({
    'fold3': 'test',
    'fold4': 'val'
}).fillna('train')
df = df[df['split']=='train']


mer_sizes = [11,21,41]
window_sizes = [11, 21, 41, 101, 1001]
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

out_dir = P['outputs']['sequence_features_new']
os.makedirs(out_dir, exist_ok=True)

for i, row in seqs.iterrows():

    features = {}
    chrom = row['chrom']
    pos = int(row['pos']) - 1
    ref = row['ref']
    alt = row['alt']
    variant = row['variant']

    features['variant'] = variant
    features['chrom'] = chrom
    features['pos'] = pos
    features['ref'] = ref
    features['alt'] = alt

    whole_ref_seq = get_centered_sequence(fasta, chrom, pos, window_size=196609)

    out_path = os.path.join(out_dir, f'{variant}.parquet')
    if os.path.exists(out_path):
        try:
            pd.read_parquet(out_path)
            continue
        except Exception as e:
            print(f"Skipping: One of the files is corrupted.\n - {out_path}\nError: {e}")
            pass


    # K-mer features (11, 21, 41)
    for w in mer_sizes:
        ref_seq = get_centered_sequence(fasta, chrom, pos, window_size=w)
        alt_seq = get_alt_sequence(ref_seq, alt)

        ref_count = count_mer(ref_seq, df, fasta)
        alt_count = count_mer(alt_seq, df, fasta)

        features[f"ref_{w}mer_count"] = ref_count
        features[f"alt_{w}mer_count"] = alt_count

        features[f"ref_{w}mer_seen"] = int(ref_count > 0)
        features[f"alt_{w}mer_seen"] = int(alt_count > 0)

        features[f"ref_{w}mer_has_N"] = int('N' in ref_seq)
        features[f"alt_{w}mer_has_N"] = int('N' in alt_seq)

    # GC content features
    for w in window_sizes:
        seq = get_centered_sequence(fasta, chrom, pos, window_size=w)
        features[f'GC_content_{w}bp'] = gc_content(seq)

    # chromatin marks
    for track, path in bigwig_paths.items():
        for w in window_sizes:
            stats = extract_bigwig_features(path, chrom, pos, window=w)
            for stat_name, value in stats.items():
                features[f"{track}_{stat_name}_{w}bp"] = value

    features_df = pd.DataFrame([features])
    features_df.to_parquet(os.path.join(out_dir, f"{variant}.parquet"), index=False)