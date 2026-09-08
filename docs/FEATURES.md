# gRely feature families

gRely's meta-model consumes ~1,161 continuous features per (variant, gene,
tissue) tuple (paper Methods → "gRely features"; the Fig 1C legend says 1121).
Crucially, **gene and tissue identity are never encoded as categoricals** — they
are parameterized entirely by continuous functional properties, which is what
lets gRely score genes/tissues it never saw in training. The blocks below map to
the code in `scripts/04_featurization/compile_final_feature_set.py`.

## 1. Variant-level
- **Sequence–chromatin** (`variant_features.py`): GC content in 11/21/41/101/1001 bp
  windows; 11/21/41-mer occurrence counts in the training set; EpiMap "Other"
  average tracks (ATAC, DNase, H3K27ac, H3K4me1, H3K4me3, H3K9ac, H3K9me3,
  H3K27me3, H3K36me3, CTCF, POLR2A) summarized (mean/median/max/min/std/q25/q75/
  nonzero-frac/nan-frac/sum) per window.
- **Conservation**: phyloP at the variant and mean over ±100 bp / ±1000 bp.
- **Genomic context**: distance to TSS, centromere, telomere; overlap with /
  distance to nearest Borzoi training region.
- **Local eQTL density**: positive-eQTL counts in ±1 bp / ±1 kb / ±10 kb, across
  all tissues and within the target tissue, rows and unique variants.

## 2. Gene-level
- **Sequence–chromatin** (gene body): GC + the same EpiMap track summaries over
  the gene span (`scripts/04_featurization/gene_sequence_features.py`).
- **Observed expression** (`gene_data_features.py`): distributional stats of the
  observed TFRecord targets globally and over exon bins; across-cell/across-bin
  variability; Shannon entropy; sparsity/peakiness; per-GTEx-track means, sums,
  z-scores, fold-changes, specificity indices, τ, Gini, top-1 fraction.
- **Model performance on the gene**: Pearson (along tracks / along bins)
  summaries; the target-tissue track's performance and its z-score vs other tissues.

## 3. Tissue context
Derived from the matching Borzoi track, **not** a tissue label: observed
expression of the target gene in that track, Borzoi's accuracy on that track,
the predicted signal over the gene, tissue-specificity of the prediction, etc.

## 4. Base-model behavior
- **Effect size across alleles**: L1/L2 distance, max absolute change, signed
  ref/alt sums, log-sums, sqrt-sums and their deltas over exon bins.
- **Replicate uncertainty**: per-bin replicate std (mean/max/std), coefficient of
  variation, agreement fraction (std < 0.05), replicate entropy — for ref, alt,
  and (alt−ref).
- **All-tracks difference vector** (7611): distributional summary of the average
  `sum` and `log` score vectors (moments, norms, quantiles, sign structure,
  sparsity, Gini, entropy, tail-mass concentration) plus cross-vector
  Pearson/cosine/sign-agreement/scale-ratios (`summarize_vector` in `grely.stats`).
- **Model-predicted gene features**: the observed-expression feature set applied
  to the predicted (rather than observed) gene matrix, plus gene-level
  uncertainty.

## What gRely learns (background)
The predicted-effect-magnitude features carry the most weight, followed by a long
tail of chromatin-context, replicate-agreement, and gene-expression features.
gRely uses signals that magnitude alone does not carry: correct low-magnitude
calls tend to be supported by open chromatin (ATAC/POLR2A) at the variant, while
incorrect high-magnitude calls tend to have high replicate dispersion and large
distance to training regions. This is why gRely stays discriminative in the
low-|VEP| regime that effect-size filtering discards.
