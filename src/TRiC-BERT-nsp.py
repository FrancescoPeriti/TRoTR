import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import roc_curve
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import f1_score, roc_curve

from datasets import Dataset
from transformers import BertTokenizer, BertForNextSentencePrediction, logging as transformers_logging

# avoid boring logging
transformers_logging.set_verbosity_error()

import argparse

parser = argparse.ArgumentParser(prog='BERT', description='Extract BERT embeddings')
parser.add_argument('-m', '--model', type=str, default='bert-base-uncased')
parser.add_argument('-l', '--layer', type=int, default=7)
parser.add_argument('-f')
parser.add_argument('-b', '--batch_size', type=int, default=32)
parser.add_argument('-d', '--device', type=str, default='cuda', choices=['cuda', 'cpu'])
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
layer = args.layer
device = torch.device(args.device)

# BERT model
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForNextSentencePrediction.from_pretrained(model_name)
model.to(device)
model.eval()


def tokenize(examples):
    """Tokenization function"""
    return tokenizer([e for i, e in enumerate(examples["context"]) if i % 2 != 0],
                     [e for i, e in enumerate(examples["context"]) if i % 2 == 0],
                     return_tensors='pt',
                     padding="max_length",
                     max_length=tokenizer.model_max_length,
                     truncation=True).to(device)


def encode(sentences, batch_size):
    encoding = tokenizer([e for i, e in enumerate(sentences) if i % 2 != 0],
                         [e for i, e in enumerate(sentences) if i % 2 == 0], return_tensors="pt", padding="max_length",
                         max_length=tokenizer.model_max_length,
                         truncation=True).to(device)

    nsp_values = list()
    num_rows = encoding['input_ids'].shape[0]
    for i in range(0, num_rows, batch_size):
        start, end = i, min(i + batch_size, num_rows)
        num_row_batch = encoding['input_ids'][start:end].shape[0]

        model_input = dict()

        # to device
        model_input['input_ids'] = encoding['input_ids'][start:end].to(device)
        model_input['attention_mask'] = encoding['attention_mask'][start:end].to(device)

        # XLM-R doesn't use 'token_type_ids'
        if 'token_type_ids' in encoding:
            model_input['token_type_ids'] = encoding['token_type_ids'][start:end].to(device)

        # model prediction
        with torch.no_grad():
            model_output = model(**model_input, next_sentence_label=torch.ones((num_row_batch, 1),
                                                                               dtype=torch.long).to(device),
                                 target_batch_size=num_row_batch)

        # select the embeddings of a specific target word
        for j in range(num_row_batch):
            nsp = model_output.logits[j][0].item()
            nsp_values.append(nsp)

    return np.array(nsp_values)


# wrapper
scores = defaultdict(list)
labels = defaultdict(list)
sentences = defaultdict(list)
mask_sentences = defaultdict(list)
distances = defaultdict(list)
mask_distances = defaultdict(list)

for data_set in ['train', 'test', 'dev']:
    lines = open(f'datasets/line-by-line{data_set}-ranking.jsonl', mode='r', encoding='utf-8').readlines()
    for i, row in enumerate(open(f'datasets/line-by-line{data_set}-binary.jsonl', mode='r', encoding='utf-8')):
        row = json.loads(row)
        start, end = [int(i) for i in row['indices_target_token'].split(':')]
        sentences[data_set].append(row['context'])
        mask_sentences[data_set].append(row['context'][:start] + ' - ' + row['context'][end:])
        if i % 2 == 0:
            labels[data_set].append(float(row['label']))
            scores[data_set].append(float(json.loads(lines[i])['label']))

    distances[data_set] = encode(sentences[data_set], batch_size=batch_size)
    mask_distances[data_set] = encode(mask_sentences[data_set], batch_size=batch_size)

spearman_corr, spearman_pvalue = list(), list()
pearson_corr, pearson_pvalue = list(), list()
mask_spearman_corr, mask_spearman_pvalue = list(), list()
mask_pearson_corr, mask_pearson_pvalue = list(), list()

for data_set in ['train', 'test', 'dev']:
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
for data_set in ['train', 'test', 'dev']:
    f1 = f1_score(labels[data_set], [m <= thr for m in distances[data_set]], average='weighted')
    f1_scores.append(f1)
    f1 = f1_score(labels[data_set], [m <= mask_thr for m in mask_distances[data_set]], average='weighted')
    mask_f1_scores.append(f1)

header = ['model'] + [f'{data_set}-{column}' for data_set in ['train', 'test', 'dev'] for column in
                      ['spearman_corr', 'spearman_pvalue', 'pearson_corr', 'pearson_pvalue', 'f1_score']] + ['thr']
header = "\t".join(header)

stats_file = "TRiC-stats.tsv"
if not Path(stats_file).is_file():
    lines = [header + '\n']
else:
    lines = open(stats_file, mode='r', encoding='utf-8').readlines()

lines.append(f'{model_name}_NSP\t' + "\t".join(
    [f'{spearman_corr[i]}\t{spearman_pvalue[i]}\t{pearson_corr[i]}\t{pearson_pvalue[i]}\t{f1_scores[i]}' for i in
     range(3)]) + f'\t{thr}\n')
lines.append(f'{model_name}_NSP\t' + "\t".join([
                                                   f'{mask_spearman_corr[i]}\t{mask_spearman_pvalue[i]}\t{mask_pearson_corr[i]}\t{mask_pearson_pvalue[i]}\t{mask_f1_scores[i]}'
                                                   for i in range(3)]) + f'\t{mask_thr}\n')

with open(stats_file, mode='w', encoding='utf-8') as f:
    f.writelines(lines)