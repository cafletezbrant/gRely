# import tensorflow.compat.v2 as tf
# import tensorflow_hub as hub
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
import numpy as np
from tqdm import tqdm
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from grely.paths import load_paths
P = load_paths()

df = pd.read_parquet(P['tables']['feature_matrix'])
features=list(df.columns[40:]) + ['avg_logSum']

cols_to_drop= ['fine_map_pip', 'credible_set_size', 'credible_set_min_r2', 'eqtl_study_sample_size', 'eqtl_pvalue', 'eqtl_se', 'minor_allele_freq', 'alt_allele_count',
               'total_called_alleles', 'minor_allele_sample_count']

features = [f for f in features if f not in cols_to_drop]

df['chrom'] = [x.split('_')[0] for x in df.variant]
df = df[df['variant_to_tss_distance']<196608//2].copy()
set1 = [
    "brain_amygdala",
    "brain_cerebellar_hemisphere",
    "brain_cortex",
    "nerve_tibial",
    "pituitary",
    "adrenal_gland",
    "thyroid",
]

set2 = [
    "artery_aorta",
    "artery_tibial",
    "heart_atrial_appendage",
    "heart_left_ventricle",
    "muscle",
    "lung",
    "kidney_cortex",
]

set3 = [
    "esophagus_gej",
    "esophagus_mucosa",
    "stomach",
    "small_intestine",
    "colon_sigmoid",
    "colon_transverse",
    "liver",
    "pancreas",
]

set4 = [
    "ovary",
    "testis",
    "uterus",
    "vagina",
    "prostate",
    "breast",
    "minor_salivary_gland",
]

set5 = [
    "blood",
    "LCL",
    "spleen",
    "skin_not_sun_exposed",
    "skin_sun_exposed",
    "fibroblast",
    "adipose_subcutaneous",
    "adipose_visceral",
]

train_chroms = ['chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 
                'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15',
                'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21',
                'chr22', 'chrX']

test_chroms = ['chr1', 'chr2', 'chr3']
folds = [
    ("set1", set1),
    ("set2", set2),
   ("set3", set3),
    ("set4", set4),
    ("set5", set5),
]

base = df.copy()
base['chrom'] = [x.split('_')[0] for x in base.variant]
base = base[base['variant_to_tss_distance'] < 196608 // 2]
base = base[base['chrom'].isin(test_chroms)]

parts = []
for fold_name, test_tissues in folds:
    part = base[base['target_tissue'].isin(test_tissues)].copy()
    part['fold'] = fold_name
    parts.append(part)

df_combined = pd.concat(parts, ignore_index=True)
print(df_combined.shape, df_combined['fold'].value_counts().to_dict())

tissue_to_rna_idx = {
    "blood":                       661,   # Whole_Blood
    "LCL":                         567,   # Cells_EBV-transformed_lymphocytes
    "spleen":                      625,   # Spleen
    "skin_not_sun_exposed":        666,   # Skin_Not_Sun_Exposed_Suprapubic
    "skin_sun_exposed":            643,   # Skin_Sun_Exposed_Lower_leg
    "fibroblast":                  570,   # Cells_Cultured_fibroblasts
    "adipose_subcutaneous":        632,   # Adipose_Subcutaneous
    "adipose_visceral":            658,   # Adipose_Visceral_Omentum
    "brain_amygdala":              618,   # Brain_Amygdala
    "brain_cerebellar_hemisphere": 634,   # Brain_Cerebellar_Hemisphere
    "brain_cortex":                615,   # Brain_Cortex
    "nerve_tibial":                609,   # Nerve_Tibial
    "pituitary":                   588,   # Pituitary
    "adrenal_gland":               637,   # Adrenal_Gland
    "thyroid":                     624,   # Thyroid
    "artery_aorta":                611,   # Artery_Aorta
    "artery_tibial":               653,   # Artery_Tibial
    "heart_atrial_appendage":      651,   # Heart_Atrial_Appendage
    "heart_left_ventricle":        650,   # Heart_Left_Ventricle
    "muscle":                      659,   # Muscle_Skeletal
    "lung":                        655,   # Lung
    "kidney_cortex":               606,   # Kidney_Cortex
    "esophagus_gej":               645,   # Esophagus_Gastroesophageal_Junction
    "esophagus_mucosa":            652,   # Esophagus_Mucosa
    "stomach":                     593,   # Stomach
    "small_intestine":             604,   # Small_Intestine_Terminal_Ileum
    "colon_sigmoid":               603,   # Colon_Sigmoid
    "colon_transverse":            602,   # Colon_Transverse
    "liver":                       600,   # Liver
    "pancreas":                    601,   # Pancreas
    "ovary":                       596,   # Ovary
    "testis":                      591,   # Testis
    "uterus":                      598,   # Uterus
    "vagina":                      599,   # Vagina
    "prostate":                    636,   # Prostate
    "breast":                      654,   # Breast_Mammary_Tissue
    "minor_salivary_gland":        648,   # Minor_Salivary_Gland
}

df=df_combined.copy()

import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

# ---- AlphaGenome geometry ----
SEQ_LEN_AG = 1_048_576   # 1 Mb input centered on the variant


def load_exons_from_gtf(gtf_path):
    """
    Parse a GENCODE GTF and return {gene_id_unversioned: (chrom, np.ndarray of [start, end))}
    where intervals are 0-based, half-open, and merged across all transcripts of the gene.
    """
    # Read only exon rows; GTF is 1-based, inclusive on both ends.
    cols = ['chrom', 'source', 'feature', 'start', 'end',
            'score', 'strand', 'frame', 'attribute']
    df = pd.read_csv(
        gtf_path, sep='\t', comment='#', header=None, names=cols,
        dtype={'chrom': str, 'start': np.int64, 'end': np.int64, 'attribute': str},
        usecols=['chrom', 'feature', 'start', 'end', 'attribute'],
    )
    df = df[df['feature'] == 'exon']

    # Extract gene_id from the attribute string (fast vectorized regex).
    df['gene_id'] = df['attribute'].str.extract(r'gene_id "([^"]+)"', expand=False)
    df['gene_id'] = df['gene_id'].str.split('.').str[0]   # strip version suffix
    df = df.dropna(subset=['gene_id'])

    # Convert GTF (1-based inclusive) -> 0-based half-open: start-1, end unchanged.
    df['start0'] = df['start'] - 1
    df['end0']   = df['end']

    gene_to_exons = {}
    for gene_id, sub in df.groupby('gene_id', sort=False):
        chrom = sub['chrom'].iloc[0]
        intervals = sub[['start0', 'end0']].to_numpy()
        intervals = _merge_intervals(intervals)
        gene_to_exons[gene_id] = (chrom, intervals)
    return gene_to_exons


def _merge_intervals(intervals):
    """Merge overlapping/adjacent [start, end) intervals. Returns sorted np.ndarray."""
    if len(intervals) == 0:
        return intervals
    intervals = intervals[np.argsort(intervals[:, 0])]
    merged = [intervals[0].tolist()]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return np.asarray(merged, dtype=np.int64)


def get_gene_slice_ag(gene_to_exons, gene_id, center_pos, n_bins,
                      seq_len=SEQ_LEN_AG, expected_chrom=None):
    """
    Indices (into the n_bins-long output axis) of bins overlapping any exon of `gene_id`,
    for an AlphaGenome window of length `seq_len` centered at `center_pos`.

    A bin is included if it overlaps any exon by at least 1 bp.
    """
    bin_size = seq_len // n_bins   # 1 for 1bp tracks, 128 for 128bp tracks

    gene_id_key = gene_id.split('.')[0]
    if gene_id_key not in gene_to_exons:
        return None
    chrom, exons = gene_to_exons[gene_id_key]
    if expected_chrom is not None and chrom != expected_chrom:
        return None

    seq_start = center_pos - seq_len // 2   # 0-based genomic start of the output window
    seq_end   = seq_start + seq_len

    # Keep only exons that overlap the window, then clip into window coords.
    overlap = (exons[:, 1] > seq_start) & (exons[:, 0] < seq_end)
    if not overlap.any():
        return None
    clipped = exons[overlap].copy()
    clipped[:, 0] = np.maximum(clipped[:, 0], seq_start) - seq_start   # local 0-based
    clipped[:, 1] = np.minimum(clipped[:, 1], seq_end)   - seq_start

    # Convert basepair intervals -> bin indices (any-overlap rule).
    bin_idx_set = set()
    for s, e in clipped:
        first_bin = s // bin_size
        last_bin  = (e - 1) // bin_size   # inclusive
        bin_idx_set.update(range(first_bin, last_bin + 1))

    if not bin_idx_set:
        return None
    out = np.fromiter(bin_idx_set, dtype=np.int64)
    out.sort()
    out = out[(out >= 0) & (out < n_bins)]
    return out if len(out) > 0 else None


# -------- Main loop --------
ag_pred_dir = P['other_models']['alphagenome_predictions']
gene_to_exons = load_exons_from_gtf(P['reference']['gencode_gtf'])

vep_scores, sign_matches, gene_slice_lens = [], [], []
for i in tqdm(range(len(df)), desc="Computing AlphaGenome VEPs"):
    row     = df.iloc[i]
    gene_id = row['gene_id']
    chrom   = row.get('chrom', None)         # optional, used as a sanity check
    center  = int(row['pos'])                # 1-based GTEx coord
    beta    = row['beta']
    rna_idx = tissue_to_rna_idx[row['target_tissue']]

    ref_pred = np.load(f'{ag_pred_dir}/{i}_ref.npy')   # (bins, 768)
    alt_pred = np.load(f'{ag_pred_dir}/{i}_alt.npy')   # (bins, 768)

    # ref_pred = np.load(f'torch_predictions/{i}_ref.npy')   # (bins, 768)
    # alt_pred = np.load(f'torch_predictions/{i}_alt.npy')   # (bins, 768)

    n_bins   = ref_pred.shape[0]

    # NB: convert 1-based VCF/GTEx pos -> 0-based for slicing math.
    gene_slice = get_gene_slice_ag(
        gene_to_exons, gene_id, center_pos=center - 1,
        n_bins=n_bins, expected_chrom=chrom,
    )
    if gene_slice is None:
        vep_scores.append(np.nan); sign_matches.append(False); gene_slice_lens.append(0)
        continue

    ref_track = ref_pred[gene_slice, rna_idx].astype(np.float32)
    alt_track = alt_pred[gene_slice, rna_idx].astype(np.float32)

    # ref_log = np.log2(ref_track + 1.0)
    # alt_log = np.log2(alt_track + 1.0)
    # logsum  = float(alt_log.sum() - ref_log.sum())

    eps = 1e-3
    ref_mean = ref_track.mean()
    alt_mean = alt_track.mean()
    logsum = float(np.log2((alt_mean + eps) / (ref_mean + eps)))


    vep_scores.append(logsum)
    sign_matches.append(np.sign(logsum) == np.sign(beta))
    gene_slice_lens.append(len(gene_slice))



df['alphagenome_vep']             = np.asarray(vep_scores)
df['alphagenome_sign_concordant'] = np.asarray(sign_matches)
df['alphagenome_gene_slice_bins']      = np.asarray(gene_slice_lens)

df.to_csv(P['other_models']['alphagenome_pred_csv'], index=False)
