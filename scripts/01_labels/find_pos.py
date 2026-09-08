import os
import pandas as pd
from tqdm import tqdm
import sys
from pathlib import Path
import pyranges as pr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from grely.paths import load_paths
P = load_paths()

final_qtls = pd.read_csv(P['tables']['borzoi_gtex_common'], sep='\t')
job_id = int(sys.argv[1])

base_path = P['eqtl']['susie_dir']
output_dir = P['outputs']['pos_eqtls']
os.makedirs(output_dir, exist_ok=True)

study_id = final_qtls.iloc[job_id]['study_id']
dataset_id = final_qtls.iloc[job_id]['dataset_id']
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

blacklist = pd.read_csv(P['reference']['blacklist'], sep='\t', header=None)
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

file_path = os.path.join(P['eqtl']['sumstats_dir'], study_id, dataset_id, f'{dataset_id}.all.tsv.gz')
sumstats=pd.read_csv(file_path, sep='\t', compression='gzip')

merged_df = df_snps_only.merge(sumstats, on='variant', how='left')
merged_df=merged_df[merged_df['maf']>0.01]

output_file = os.path.join(output_dir, f'{dataset_id}_pos_eqtls.tsv.gz')
merged_df.to_csv(output_file, sep='\t', index=False, compression='gzip')