import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import classification_report
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import f1_score, roc_curve

from sentence_transformers import CrossEncoder

import argparse

parser = argparse.ArgumentParser(prog='BERT', description='Extract BERT embeddings')
parser.add_argument('-m', '--model', type=str)
parser.add_argument('-f')
parser.add_argument('-b', '--batch_size', type=int, default=32)
parser.add_argument('-d', '--device', type=str, default='cuda', choices=['cuda', 'cpu'])
parser.add_argument('-f', '--k_fold', type=str, default='1')
args = parser.parse_args()


def set_threshold(y_true, y):
    fpr, tpr, thresholds = roc_curve(y_true, y)

    scores = []
    for thresh in thresholds:
        scores.append(f1_score(y_true, [m <= thresh for m in y], average='weighted'))

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
model = CrossEncoder(model_name, device=device)

# wrapper
scores = defaultdict(list)
labels = defaultdict(list)
sentences = defaultdict(list)
mask_sentences = defaultdict(list)
distances = defaultdict(list)
mask_distances = defaultdict(list)

for data_set in ['train', 'test.iov', 'test.oov', 'dev']:
    lines = open(f'TRoTR/datasets/FOLD_{k_fold}/line-by-line/{data_set}.ranking.jsonl', mode='r', encoding='utf-8').readlines()
    for i, row in enumerate(open(f'TRoTR/datasets/FOLD_{k_fold}/line-by-line/{data_set}.binary.jsonl', mode='r', encoding='utf-8')):
        row = json.loads(row)
        start, end = [int(i) for i in row['indices_target_token'].split(':')]
        sentences[data_set].append(row['context'])
        mask_sentences[data_set].append(row['context'][:start] + ' - ' + row['context'][end:])
        if i % 2 == 0:
            labels[data_set].append(1-float(row['label'])) # distance -> similarity
            scores[data_set].append(float(json.loads(lines[i])['label']))

    distances[data_set] = model.predict([(sentences[data_set][i], sentences[data_set][i + 1])
                                         for i in range(0, len(sentences[data_set]) - 1, 2)], batch_size=32)
    mask_distances[data_set] = model.predict([(mask_sentences[data_set][i], mask_sentences[data_set][i + 1])
                                              for i in range(0, len(mask_sentences[data_set]) - 1, 2)], batch_size=32)

spearman_corr, spearman_pvalue = list(), list()
pearson_corr, pearson_pvalue = list(), list()
mask_spearman_corr, mask_spearman_pvalue = list(), list()
mask_pearson_corr, mask_pearson_pvalue = list(), list()

for data_set in ['train', 'test.iov', 'test.oov', 'dev']:
    corr, pvalue = spearmanr(scores[data_set], distances[data_set])
    spearman_corr.append(corr.round(3))
    spearman_pvalue.append(pvalue.round(3))
    corr, pvalue = pearsonr(scores[data_set], distances[data_set])
    pearson_corr.append(corr.round(3))
    pearson_pvalue.append(pvalue.round(3))

    corr, pvalue = spearmanr(scores[data_set], mask_distances[data_set])
    mask_spearman_corr.append(corr.round(3))
    mask_spearman_pvalue.append(pvalue.round(3))
    corr, pvalue = pearsonr(scores[data_set], mask_distances[data_set])
    mask_pearson_corr.append(corr.round(3))
    mask_pearson_pvalue.append(pvalue.round(3))

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

for data_set in ['train', 'test.iov', 'test.oov', 'dev']:
    f1 = f1_score(labels[data_set], [m <= thr for m in distances[data_set]], average='weighted')
    f1_scores.append(f1)
    f1 = f1_score(labels[data_set], [m <= mask_thr for m in mask_distances[data_set]], average='weighted')
    mask_f1_scores.append(f1)

    # f1 = f1_score(labels[data_set], [m <= thr for m in distances[data_set]], average='binary', pos_label=1)
    # f1_scores_label1.append(f1)
    # f1 = f1_score(labels[data_set], [m <= mask_thr for m in mask_distances[data_set]], average='binary', pos_label=1)
    # mask_f1_scores_label1.append(f1)

    # f1 = f1_score(labels[data_set], [m <= thr for m in distances[data_set]], average='binary', pos_label=0)
    # f1_scores_label0.append(f1)
    # f1 = f1_score(labels[data_set], [m <= mask_thr for m in mask_distances[data_set]], average='binary', pos_label=0)
    # mask_f1_scores_label0.append(f1)

    # rec = recall_score(labels[data_set], [m <= thr for m in distances[data_set]])
    # recall_scores.append(rec)
    # rec = recall_score(labels[data_set], [m <= mask_thr for m in mask_distances[data_set]])
    # mask_recall_scores.append(rec)

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
    # f1 = f1_score(labels[data_set], [m <= mask_thr for m in mask_distances[data_set]], average='binary', pos_label=1)
    # mask_f1_scores_label1.append(f1)

header = ['model', 'k_fold'] + [f'{data_set}-{column}' for data_set in ['train', 'test.iov', 'test.oov', 'dev'] for
                                column in
                                ['spearman_corr', 'spearman_pvalue', 'pearson_corr', 'pearson_pvalue', 'f1_score',
                                 'f1_scores_label1', 'f1_scores_label0', 'recall_label1', 'recall_label0',
                                 'precision_label1', 'precision_label0']] + ['thr']
header = "\t".join(header)

stats_file = "TRiC-stats.tsv"
if not Path(stats_file).is_file():
    lines = [header + '\n']
else:
    lines = open(stats_file, mode='r', encoding='utf-8').readlines()

lines.append(f'{model_name}\t{k_fold}\t' + "\t".join([
                                                         f'{spearman_corr[i]}\t{spearman_pvalue[i]}\t{pearson_corr[i]}\t{pearson_pvalue[i]}\t{f1_scores[i]}\t{f1_scores_label1[i]}\t{f1_scores_label0[i]}\t{recall_label1[i]}\t{recall_label0[i]}\t{precision_label1[i]}\t{precision_label0[i]}'
                                                         for i in range(4)]) + f'\t{thr}\n')
lines.append(f'{model_name}_mask\t{k_fold}\t' + "\t".join([
                                                              f'{mask_spearman_corr[i]}\t{mask_spearman_pvalue[i]}\t{mask_pearson_corr[i]}\t{mask_pearson_pvalue[i]}\t{mask_f1_scores[i]}\t{mask_f1_scores_label1[i]}\t{mask_f1_scores_label0[i]}\t{mask_recall_label1[i]}\t{mask_recall_label0[i]}\t{mask_precision_label1[i]}\t{mask_precision_label0[i]}'
                                                              for i in range(4)]) + f'\t{mask_thr}\n')

with open(stats_file, mode='w', encoding='utf-8') as f:
    f.writelines(lines)