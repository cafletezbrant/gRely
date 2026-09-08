import json
import pandas as pd
import pyfaidx
from baskerville import seqnn
import pysam
import sys
sys.path.append('/home/rafia1/scratch/borzoi/examples')
from borzoi_helpers import *
import numpy as np
import os

# read in the sequences
df=pd.read_csv('/home/rafia1/scratch/data/all_sequences.csv', sep='\t')
job_id = int(sys.argv[1])
df=df.iloc[job_id*100:job_id*100+100].reset_index(drop=True)

params_file = '/home/rafia1/scratch/borzoi/examples/params_pred.json'
with open(params_file) as params_open :
    params = json.load(params_open)
    params_model = params['model']

targets_file = '/home/rafia1/scratch/data/targets_human.txt.gz'
targets_df = pd.read_csv(targets_file, index_col=0, sep='\t')
target_index = targets_df.index

fasta_path = '/home/rafia1/scratch/data/hg38.fa'
pyfaidx.Faidx(fasta_path)
fasta_open = pysam.Fastafile(fasta_path)

seq_len = 524288

models = {}
for rep_ix in range(1, 5):
    model_file = "/home/rafia1/scratch/data/model_weights/model0_best_" + str(rep_ix) + ".h5"
    seqnn_model = seqnn.SeqNN(params_model)
    seqnn_model.restore(model_file, 0)
    seqnn_model.build_slice(target_index)
    models[rep_ix] = seqnn_model

for seq_id in range(len(df)):
    variant=df.iloc[seq_id]['variant']
    chrom=df.iloc[seq_id]['chrom']
    center_pos=df.iloc[seq_id]['pos']
    poses = [center_pos]
    alts = [df.iloc[seq_id]['alt']]

    start = center_pos - seq_len // 2
    end = center_pos + seq_len // 2

    sequence_one_hot_wt = process_sequence(fasta_open, chrom, start, end)
    sequence_one_hot_mut = np.copy(sequence_one_hot_wt)
    for pos, alt in zip(poses, alts) :
        alt_ix = -1
        if alt == 'A' :
            alt_ix = 0
        elif alt == 'C' :
            alt_ix = 1
        elif alt == 'G' :
            alt_ix = 2
        elif alt == 'T' :
            alt_ix = 3

        sequence_one_hot_mut[pos-start-1] = 0.
        sequence_one_hot_mut[pos-start-1, alt_ix] = 1.

    for rep_ix, model in models.items():
        sum_score_path = f'/home/rafia1/scratch/data/borzoi_predictions/model_{rep_ix}_sum_score_{variant}.npy'
        log_score_path = f'/home/rafia1/scratch/data/borzoi_predictions/model_{rep_ix}_log_score__{variant}.npy'

        if os.path.exists(sum_score_path) and os.path.exists(log_score_path):
            try:
                np.load(sum_score_path, allow_pickle=False)
                np.load(log_score_path, allow_pickle=False)
                continue
            except Exception as e:
                print(f"Corrupt or unreadable file detected for variant {variant}, rep {rep_ix}: {e}")
                pass

        y_wt = model(sequence_one_hot_wt[None, ...])[:, None, ...].astype("float16")
        y_mut = model(sequence_one_hot_mut[None, ...])[:, None, ...].astype("float16")

        y_wt_arr = y_wt[0][0]
        y_mut_arr = y_mut[0][0]

        sum_score = np.sum(y_mut_arr - y_wt_arr, axis=0)
        log_score = np.sqrt(np.sum((np.log2(1 + y_mut_arr) - np.log2(1 + y_wt_arr))**2, axis=0))

        np.save(f'/home/rafia1/scratch/data/borzoi_predictions/model_{rep_ix}_sum_score_{variant}.npy', sum_score)
        np.save(f'/home/rafia1/scratch/data/borzoi_predictions/model_{rep_ix}_log_score__{variant}.npy', log_score)