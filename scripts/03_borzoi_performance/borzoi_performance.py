import json
import pandas as pd
import pyfaidx
from baskerville import seqnn
import pysam
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from grely.paths import load_paths
P = load_paths()
sys.path.append(P['roots']['borzoi_examples'])
from borzoi_helpers import *
import numpy as np
import os
import pandas as pd
import numpy as np
import bioframe as bf
import tensorflow as tf
from matplotlib import pyplot as plt
import json
import pandas as pd
import pyfaidx
from baskerville import seqnn
import pysam
from scipy.stats import pearsonr


# read in the sequences
perf_dir = P['outputs']['borzoi_performance']
os.makedirs(perf_dir, exist_ok=True)
df=pd.read_csv(P['tables']['all_sequences'], sep='\t')
job_id = int(sys.argv[1])
df=df.iloc[job_id*100:job_id*100+100].reset_index(drop=True)

params_file = P['borzoi']['params_pred']
with open(params_file) as params_open :
    params = json.load(params_open)
    params_model = params['model']

targets_file = P['borzoi']['targets_human']
targets_df = pd.read_csv(targets_file, index_col=0, sep='\t')
target_index = targets_df.index

fasta_path = P['reference']['fasta']
pyfaidx.Faidx(fasta_path)
fasta_open = pysam.Fastafile(fasta_path)

file_path = P['reference']['borzoi_sequences_bed']
seq_mappingF = pd.read_csv(file_path, sep='\t', header=None, names=['chrom', 'start', 'end', 'fold'])
seq_mappingF['tf_label'] = seq_mappingF.groupby('fold').cumcount() // 16
seq_mappingF['tfrecord'] = seq_mappingF.apply(lambda row: f"{row['fold']}-{row['tf_label']}.tfr", axis=1)
seq_mappingF['example_number'] = seq_mappingF.groupby('fold').cumcount() % 16

seq_len = 524288

models = {}
for rep_ix in range(1, 5):
    model_file = P['borzoi']['model_weights'].format(rep=rep_ix)
    seqnn_model = seqnn.SeqNN(params_model)
    seqnn_model.restore(model_file, 0)
    seqnn_model.build_slice(target_index)
    models[rep_ix] = seqnn_model

for seq_id in range(len(df)):
    variant=df.iloc[seq_id]['variant']
    chrom = df.iloc[seq_id]['chrom']
    query_start = df.iloc[seq_id]['pos']
    query_end = df.iloc[seq_id]['pos']+1
    query_center = df.iloc[seq_id]['pos']

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
    if valid.empty:
        print(f"No overlapping record for variant {variant}, skipping...")
        continue

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
    target = tf.reshape(t, (-1,7611)).numpy() # 7611
    target = target.T

    chrom=selected_record['chrom']
    current_start = selected_record['start']
    current_end = selected_record['end']
    center= (current_start+current_end)//2
    center_pos=center
    poses = [center_pos]

    start = center_pos - seq_len // 2
    end = center_pos + seq_len // 2

    sequence_one_hot_wt = process_sequence(fasta_open, chrom, start, end)

    for rep_ix, model in models.items():
        path1 = os.path.join(perf_dir, f'model_{rep_ix}_{variant}_along_sequence.npy')
        path2 = os.path.join(perf_dir, f'model_{rep_ix}_{variant}_along_track.npy')

        if os.path.exists(path1) and os.path.exists(path2):
            try:
                np.load(path1, allow_pickle=False)
                np.load(path2, allow_pickle=False)
                continue
            except Exception as e:
                print(f"Skipping: One of the files is corrupted.\n - {path1}\n - {path2}\nError: {e}")
                pass

        y_wt = model(sequence_one_hot_wt[None, ...])[:, None, ...].astype("float16")[0][0]

        start_idx = (16352 - 6144) // 2
        end_idx = start_idx + 6144
        y_wt_centered = y_wt[start_idx:end_idx, :]

        pearsons=[]
        for i in range(7611):
            pearsons.append(pearsonr(y_wt_centered[:,i], target.T[:,i])[0])
        pearsons_seq = np.array(pearsons)

        pearsons=[]
        for i in range(6144):
            pearsons.append(pearsonr(y_wt_centered[i,:], target.T[i,:])[0])
        pearsons_track = np.array(pearsons)

        np.save(os.path.join(perf_dir, f'model_{rep_ix}_{variant}_along_sequence.npy'), pearsons_seq)
        np.save(os.path.join(perf_dir, f'model_{rep_ix}_{variant}_along_track.npy'), pearsons_track)