"""Gene-slice geometry for computing other models' variant effect predictions.

To apply gRely to a different S2F model you first need that model's VEP for each
(variant, gene, tissue) tuple, computed the same way as Borzoi's: sum the model's
predicted signal over the output bins that overlap the target gene's exons, for
the matching output track, and take the log-ratio (logSUM) between alt and ref.

These helpers map a gene's exons to the model's output-bin indices for a window
centered on the variant. ``load_exons_from_gtf`` / ``_merge_intervals`` are
shared; ``get_gene_slice_ag`` (AlphaGenome) and ``get_gene_slice_enformer``
(Enformer) differ only in the model geometry. All function bodies are copied
verbatim from the original ``alphagenome.py`` and the Enformer notebook.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---- AlphaGenome geometry ----
SEQ_LEN_AG = 1_048_576   # 1 Mb input centered on the variant

# ---- Enformer geometry ----
SEQ_LEN_ENF       = 393_216    # input window
SEQ_OUT_LEN_ENF   = 114_688    # output window (central crop)
N_BINS_ENF        = 896
BIN_SIZE_ENF      = 128
SEQ_OUT_OFFSET_ENF = (SEQ_LEN_ENF - SEQ_OUT_LEN_ENF) // 2   # 139,264 bp on each side


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


def get_gene_slice_enformer(gene_to_exons, gene_id, center_pos,
                            expected_chrom=None):
    """
    Indices (into the 896-long output axis) of bins overlapping any exon of `gene_id`,
    for an Enformer window centered at `center_pos` (0-based).

    A bin is included if it overlaps any exon by at least 1 bp.
    """
    gene_id_key = gene_id.split('.')[0]
    if gene_id_key not in gene_to_exons:
        return None
    chrom, exons = gene_to_exons[gene_id_key]
    if expected_chrom is not None and chrom != expected_chrom:
        return None

    # Output window covers the CENTRAL 114,688 bp of the input.
    seq_start = center_pos - SEQ_LEN_ENF // 2          # input window start
    out_start = seq_start + SEQ_OUT_OFFSET_ENF         # output window start
    out_end   = out_start + SEQ_OUT_LEN_ENF            # output window end

    overlap = (exons[:, 1] > out_start) & (exons[:, 0] < out_end)
    if not overlap.any():
        return None
    clipped = exons[overlap].copy()
    clipped[:, 0] = np.maximum(clipped[:, 0], out_start) - out_start
    clipped[:, 1] = np.minimum(clipped[:, 1], out_end)   - out_start

    bin_idx_set = set()
    for s, e in clipped:
        first_bin = s // BIN_SIZE_ENF
        last_bin  = (e - 1) // BIN_SIZE_ENF
        bin_idx_set.update(range(first_bin, last_bin + 1))

    if not bin_idx_set:
        return None
    out = np.fromiter(bin_idx_set, dtype=np.int64)
    out.sort()
    out = out[(out >= 0) & (out < N_BINS_ENF)]
    return out if len(out) > 0 else None
