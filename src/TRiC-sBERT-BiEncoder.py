import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import f1_score, roc_curve
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer

import argparse
parser = argparse.ArgumentParser(prog='sBERT', description='Extract sBERT embeddings')
parser.add_argument('--add_tags', action='store_true')
parser.add_argument('-p', '--data_path', type=str, default='')
parser.add_argument('-m', '--model', type=str, default='all-MiniLM-L6-v2')
parser.add_argument('-b', '--batch_size', type=int, default=32)
parser.add_argument('-d', '--device', type=str, default='cuda', choices=['cuda', 'cpu'])
parser.add_argument('-k', '--k_fold', type=str, default="1")
args = parser.parse_args()

def set_threshold(y_true, y, average='weighted', pos_label=1):
    fpr, tpr, thresholds = roc_curve(y_true, y)

    scores = []
    for thresh in thresholds:
        scores.append(f1_score(y_true, [m <= thresh for m in y], average=average, pos_label=pos_label))

    scores = np.array(scores)
    max_score = scores.max()
    max_score_threshold = thresholds[scores.argmax()]

    return max_score.round(3), max_score_threshold.round(3)

# Parameters
model_name = args.model
batch_size = args.batch_size
device = args.device
k_fold = args.k_fold

# sBERT model
model = SentenceTransformer(model_name)

# wrapper
scores = defaultdict(list)
labels = defaultdict(list)
sentences = defaultdict(list)
mask_sentences = defaultdict(list)
embeddings = dict()
mask_embeddings = dict()
distances = defaultdict(list)
mask_distances = defaultdict(list)

def load_sentence(row, add_tags):
    if not add_tags:
        return row['context']
    else:
        start, end = [int(i) for i in row['indices_target_token'].split(':')]
        return row['context'][:start] + '<t>' + row['context'][start:end] + '</t>' + row['context'][end:]


data_sets_evaluation = ['train', 'test', 'test.iov', 'test.oov', 'dev', 'dev.iov', 'dev.oov']
for data_set in data_sets_evaluation:
    lines = open(f'{args.data_path}TRoTR/datasets/FOLD_{k_fold}/line-by-line/{data_set}.ranking.jsonl', mode='r', encoding='utf-8').readlines()
    for i, row in enumerate(open(f'{args.data_path}TRoTR/datasets/FOLD_{k_fold}/line-by-line/{data_set}.binary.jsonl', mode='r', encoding='utf-8')):
        row = json.loads(row)
        start, end = [int(i) for i in row['indices_target_token'].split(':')]
        sentences[data_set].append(load_sentence(row, args.add_tags)) #row['context'])
        mask_sentences[data_set].append(row['context'][:start] + ' - ' + row['context'][end:])
        if i % 2 == 0:
            labels[data_set].append(float(row['label']))
            scores[data_set].append(float(json.loads(lines[i])['label']))

    embeddings[data_set] = model.encode(sentences[data_set], batch_size=batch_size, device=device)
    mask_embeddings[data_set] = model.encode(mask_sentences[data_set], batch_size=batch_size, device=device)

    for i in range(0, embeddings[data_set].shape[0] - 1, 2):
        emb1 = embeddings[data_set][i]
        emb2 = embeddings[data_set][i + 1]
        mask_emb1 = mask_embeddings[data_set][i]
        mask_emb2 = mask_embeddings[data_set][i + 1]
        distances[data_set].append(cosine(emb1, emb2))
        mask_distances[data_set].append(cosine(mask_emb1, mask_emb2))

spearman_corr, spearman_pvalue = list(), list()
pearson_corr, pearson_pvalue = list(), list()
mask_spearman_corr, mask_spearman_pvalue = list(), list()
mask_pearson_corr, mask_pearson_pvalue = list(), list()

for data_set in data_sets_evaluation:
    corr, pvalue = spearmanr(scores[data_set], -np.array(distances[data_set])) # distance -> similarity
    spearman_corr.append(round(corr,3))
    spearman_pvalue.append(round(pvalue, 3))
    corr, pvalue = pearsonr(scores[data_set], -np.array(distances[data_set])) # distance -> similarity
    pearson_corr.append(round(corr, 3))
    pearson_pvalue.append(round(pvalue, 3))

    corr, pvalue = spearmanr(scores[data_set], -np.array(mask_distances[data_set])) # distance -> similarity
    mask_spearman_corr.append(round(corr, 3))
    mask_spearman_pvalue.append(round(pvalue, 3))
    corr, pvalue = pearsonr(scores[data_set], -np.array(mask_distances[data_set])) # distance -> similarity
    mask_pearson_corr.append(round(corr, 3))
    mask_pearson_pvalue.append(round(pvalue, 3))

_, thr = set_threshold(labels['dev'], mask_distances['dev'])
_, mask_thr = set_threshold(labels['dev'], mask_distances['dev'])

f1_scores = list()
mask_f1_scores = list()
f1_scores_label1 = list()
mask_f1_scores_label1 = list()
f1_scores_label0 = list()
mask_f1_scores_label0 = list()
recall_label1 = list()
mask_recall_label1 = list()
recall_label0 = list()
mask_recall_label0 = list()
precision_label1 = list()
mask_precision_label1 = list()
precision_label0 = list()
mask_precision_label0 = list()

for data_set in data_sets_evaluation:
    f1 = f1_score(labels[data_set], [m <= thr for m in distances[data_set]], average='weighted')
    f1_scores.append(f1)
    f1 = f1_score(labels[data_set], [m <= mask_thr for m in mask_distances[data_set]], average='weighted')
    mask_f1_scores.append(f1)

    #f1 = f1_score(labels[data_set], [m <= thr for m in distances[data_set]], average='binary', pos_label=1)
    #f1_scores_label1.append(f1)
    #f1 = f1_score(labels[data_set], [m <= mask_thr for m in mask_distances[data_set]], average='binary', pos_label=1)
    #mask_f1_scores_label1.append(f1)

    #f1 = f1_score(labels[data_set], [m <= thr for m in distances[data_set]], average='binary', pos_label=0)
    #f1_scores_label0.append(f1)
    #f1 = f1_score(labels[data_set], [m <= mask_thr for m in mask_distances[data_set]], average='binary', pos_label=0)
    #mask_f1_scores_label0.append(f1)

    #rec = recall_score(labels[data_set], [m <= thr for m in distances[data_set]])
    #recall_scores.append(rec)
    #rec = recall_score(labels[data_set], [m <= mask_thr for m in mask_distances[data_set]])
    #mask_recall_scores.append(rec)

    rep = classification_report(labels[data_set], [m <= thr for m in distances[data_set]], output_dict=True)
    f1_scores_label1.append(rep['1.0']['f1-score'])
    recall_label1.append(rep['1.0']['recall'])
    precision_label1.append(rep['1.0']['precision'])
    f1_scores_label0.append(rep['0.0']['f1-score'])
    recall_label0.append(rep['0.0']['recall'])
    precision_label0.append(rep['0.0']['precision'])
    rep = classification_report(labels[data_set], [m <= mask_thr for m in mask_distances[data_set]], output_dict=True)
    mask_f1_scores_label1.append(rep['1.0']['f1-score'])
    mask_recall_label1.append(rep['1.0']['recall'])
    mask_precision_label1.append(rep['1.0']['precision'])
    mask_f1_scores_label0.append(rep['0.0']['f1-score'])
    mask_recall_label0.append(rep['0.0']['recall'])
    mask_precision_label0.append(rep['0.0']['precision'])
    #f1 = f1_score(labels[data_set], [m <= mask_thr for m in mask_distances[data_set]], average='binary', pos_label=1)
    #mask_f1_scores_label1.append(f1)
    

header = ['model', 'k_fold'] + [f'{data_set}-{column}' for data_set in data_sets_evaluation for column in ['spearman_corr', 'spearman_pvalue', 'pearson_corr', 'pearson_pvalue', 'f1_score', 'f1_scores_label1', 'f1_scores_label0', 'recall_label1', 'recall_label0', 'precision_label1', 'precision_label0']] + ['thr']
header = "\t".join(header)

stats_file = "TRiC-stats.tsv"
if not Path(stats_file).is_file():
    lines = [header+'\n']
else:
    lines = open(stats_file, mode='r',encoding='utf-8').readlines()

lines.append(f'{model_name}\t{k_fold}\t' + "\t".join([f'{spearman_corr[i]}\t{spearman_pvalue[i]}\t{pearson_corr[i]}\t{pearson_pvalue[i]}\t{f1_scores[i]}\t{f1_scores_label1[i]}\t{f1_scores_label0[i]}\t{recall_label1[i]}\t{recall_label0[i]}\t{precision_label1[i]}\t{precision_label0[i]}' for i in range(len(data_sets_evaluation))]) + f'\t{thr}\n')
lines.append(f'{model_name}_mask\t{k_fold}\t' + "\t".join([f'{mask_spearman_corr[i]}\t{mask_spearman_pvalue[i]}\t{mask_pearson_corr[i]}\t{mask_pearson_pvalue[i]}\t{mask_f1_scores[i]}\t{mask_f1_scores_label1[i]}\t{mask_f1_scores_label0[i]}\t{mask_recall_label1[i]}\t{mask_recall_label0[i]}\t{mask_precision_label1[i]}\t{mask_precision_label0[i]}' for i in range(len(data_sets_evaluation))]) + f'\t{mask_thr}\n')

with open(stats_file, mode='w', encoding='utf-8') as f:
    f.writelines(lines)
