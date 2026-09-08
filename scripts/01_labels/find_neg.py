import pandas as pd
import pyranges as pr
import os
from tqdm import tqdm
import sys

job_id = int(sys.argv[1])
base_path = '/gstore/data/humgenet/ref/eqtl_catalogue/susie/'
output_dir = '../data/neg_eqtls'
os.makedirs(output_dir, exist_ok=True)

final_qtls = pd.read_csv('../data/borzoi_gtex_common.csv', sep='\t')
study_id = final_qtls.iloc[job_id]['study_id']
dataset_id = final_qtls.iloc[job_id]['dataset_id']

file_path = f'/gstore/data/humgenet/ref/eqtl_catalogue/sumstats/{study_id}/{dataset_id}/{dataset_id}.all.tsv.gz'
sumstats=pd.read_csv(file_path, sep='\t', compression='gzip')

### filter for maf

sumstats= sumstats[sumstats['maf']>0.01]

### remove all credible variants from all gtex datasets

all_variants_list = []

datasets = os.listdir(os.path.join(base_path, study_id))
for dataset in tqdm(datasets, desc="Loading GTEx Datasets"):
    filepath = os.path.join(base_path, study_id, dataset, f'{dataset}.credible_sets.tsv.gz')
    df = pd.read_csv(filepath, sep='\t', compression='gzip', usecols=['variant'])
    all_variants_list.append(df)

all_variants_pool_df = pd.concat(all_variants_list, ignore_index=True)

split_variants_pool = all_variants_pool_df['variant'].str.split('_', expand=True)
all_variants_pool_df['chr'] = split_variants_pool[0]
all_variants_pool_df['pos'] = pd.to_numeric(split_variants_pool[1])
all_variants_pool_df['ref'] = split_variants_pool[2]
all_variants_pool_df['alt'] = split_variants_pool[3]

credible_variants_set = set(all_variants_pool_df['variant'])
sumstats = sumstats[~sumstats['variant'].isin(credible_variants_set)]

### keep only snvs

split_variants = sumstats['variant'].str.split('_', expand=True)
sumstats['chrom'] = split_variants[0]
sumstats['pos'] = split_variants[1]
sumstats['ref'] = split_variants[2]
sumstats['alt'] = split_variants[3]
is_snp_mask = (sumstats['ref'].str.len() == 1) & (sumstats['alt'].str.len() == 1)
sumstats = sumstats[is_snp_mask]

### remove blacklist

sumstats['pos'] = pd.to_numeric(sumstats['pos'], errors='coerce')
if sumstats['pos'].isna().any():
    print(f"Warning: {sumstats['pos'].isna().sum()} rows have non-numeric positions and will be excluded.")
    sumstats = sumstats.dropna(subset=['pos'])

blacklist = pd.read_csv('../data/hg38-blacklist.v2.bed', sep='\t', header=None)
blacklist.columns = ['Chromosome', 'Start', 'End', 'Type']
blacklist['Chromosome'] = blacklist['Chromosome'].astype(str)
blacklist_pr = pr.PyRanges(blacklist[['Chromosome', 'Start', 'End']])

variants_df = pd.DataFrame({
    'Chromosome': sumstats['chrom'].astype(str),
    'Start': sumstats['pos'].astype(int) - 1,
    'End': sumstats['pos'].astype(int),
    'orig_idx': sumstats.index })

variants_pr = pr.PyRanges(variants_df)
overlap = variants_pr.join(blacklist_pr)
overlapping_idx = overlap.df['orig_idx'].unique()
sumstats.drop(index=overlapping_idx, inplace=True)

### keep only genes with pos qtls

filepath = os.path.join(base_path, study_id, dataset_id, f'{dataset_id}.credible_sets.tsv.gz')
susie_df = pd.read_csv(filepath, sep='\t', compression='gzip')

df_pip_filtered = susie_df[susie_df['pip'] > 0.9].copy()
split_variants = df_pip_filtered['variant'].str.split('_', expand=True)
df_pip_filtered['chrom'] = split_variants[0]
df_pip_filtered['pos'] = split_variants[1]
df_pip_filtered['ref'] = split_variants[2]
df_pip_filtered['alt'] = split_variants[3]
is_snp_mask = (df_pip_filtered['ref'].str.len() == 1) & (df_pip_filtered['alt'].str.len() == 1)
df_snps_only = df_pip_filtered[is_snp_mask].copy()
for col in final_qtls.columns:
    df_snps_only.loc[:, col] = final_qtls.iloc[job_id][col]

columns_to_drop = ['ftp_path', 'ftp_cs_path', 'ftp_lbf_path']
df_snps_only = df_snps_only.drop(columns=columns_to_drop, errors='ignore')

blacklist = pd.read_csv('../data/hg38-blacklist.v2.bed', sep='\t', header=None)
blacklist.columns = ['Chromosome', 'Start', 'End', 'Type']
blacklist['Chromosome'] = blacklist['Chromosome'].astype(str)
blacklist_pr = pr.PyRanges(blacklist[['Chromosome', 'Start', 'End']])

variants_df = pd.DataFrame({
    'Chromosome': df_snps_only['chrom'].astype(str),
    'Start': df_snps_only['pos'].astype(int) - 1,
    'End': df_snps_only['pos'].astype(int),
    'orig_idx': df_snps_only.index })

variants_pr = pr.PyRanges(variants_df)
overlap = variants_pr.join(blacklist_pr)
overlapping_idx = overlap.df['orig_idx'].unique()
df_snps_only.drop(index=overlapping_idx, inplace=True)
df_snps_only.reset_index(drop=True, inplace=True)

sumstats = sumstats[sumstats['gene_id'].isin(df_snps_only['gene_id'])]

# match distance with pos eqtls

df_pr = pr.PyRanges(pd.DataFrame({
    'Chromosome': df_snps_only['chrom'].astype(str),
    'Start': df_snps_only['pos'].astype(int) - 1000,
    'End': df_snps_only['pos'].astype(int) + 1000
}))

sumstats_pr_df = pd.DataFrame({
    'Chromosome': sumstats['chrom'].astype(str),
    'Start': sumstats['pos'].astype(int) - 1,
    'End': sumstats['pos'].astype(int),
    'orig_idx': sumstats.index
})

sumstats_pr = pr.PyRanges(sumstats_pr_df)
overlap = sumstats_pr.join(df_pr)
matching_idx = overlap.df['orig_idx'].unique()
sumstats = sumstats.loc[matching_idx].copy()

output_file = os.path.join(output_dir, f'{dataset_id}_neg_eqtls.tsv.gz')
sumstats.to_csv(output_file, sep='\t', index=False, compression='gzip')