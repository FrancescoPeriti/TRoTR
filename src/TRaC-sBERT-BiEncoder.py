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
import csv


parser = argparse.ArgumentParser(prog='sBERT', description='Extract sBERT embeddings')
parser.add_argument('--add_tags', action='store_true')
parser.add_argument('-p', '--data_path', type=str, default='')
parser.add_argument('-m', '--model', type=str, default='all-MiniLM-L6-v2')
parser.add_argument('-b', '--batch_size', type=int, default=32)
parser.add_argument('-d', '--device', type=str, default='cuda', choices=['cuda', 'cpu'])
args = parser.parse_args()

# Parameters
model_name = args.model
batch_size = args.batch_size
device = args.device

# sBERT model
model = SentenceTransformer(model_name)

# wrapper
sentences = []
mask_sentences = []
embeddings = []
mask_embeddings = []
distances = []
mask_distances = []
lemma2index = {}


def load_gold_truth():
    gold_truth = {}
    with open(f'{args.data_path}TRoTR/datasets/trac_gold_scores.tsv', newline='') as csvfile:
        goldreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for j,row in enumerate(goldreader):
            if not j == 0:
                lemma, score = row
                gold_truth[lemma] = float(score)
    return gold_truth


def load_sentence(row, add_tags):
    if not add_tags:
        return row['context']
    else:
        start, end = [int(i) for i in row['indices_target_token'].split(':')]
        return row['context'][:start] + '<t>' + row['context'][start:end] + '</t>' + row['context'][end:]


for i, row in enumerate(open(f'{args.data_path}TRoTR/TRaC_datasets/line-by-line/TRaC.jsonl', mode='r', encoding='utf-8')):
    row = json.loads(row)
    start, end = [int(i) for i in row['indices_target_token'].split(':')]
    sentences.append(load_sentence(row, args.add_tags))
    mask_sentences.append(row['context'][:start] + ' - ' + row['context'][end:])
    if i % 2 == 0:
        if not row['lemma'] in lemma2index:
            lemma2index[row['lemma']] = []
        lemma2index[row['lemma']].append(int(i/2))



embeddings = model.encode(sentences, batch_size=batch_size, device=device)
mask_embeddings = model.encode(mask_sentences, batch_size=batch_size, device=device)

for i in range(0, embeddings.shape[0] - 1, 2):
    emb1 = embeddings[i]
    emb2 = embeddings[i + 1]
    mask_emb1 = mask_embeddings[i]
    mask_emb2 = mask_embeddings[i + 1]
    distances.append(cosine(emb1, emb2))
    mask_distances.append(cosine(mask_emb1, mask_emb2))

scores = []
predictions = []
mask_predictions = []
gold_truth = load_gold_truth()

for lemma in gold_truth:
    scores.append(gold_truth[lemma])
    predictions.append(np.mean([distances[i] for i in lemma2index[lemma]]))
    mask_predictions.append(np.mean([mask_distances[i] for i in lemma2index[lemma]]))


corr, pvalue = spearmanr(scores, np.array(predictions)) # distance -> similarity
pearson_corr, pearson_pvalue = pearsonr(scores, np.array(predictions)) # distance -> similarity

mask_corr, mask_pvalue = spearmanr(scores, np.array(mask_predictions)) # distance -> similarity
mask_pearson_corr, mask_pearson_pvalue = pearsonr(scores, np.array(mask_predictions)) # distance -> similarity



header = ['model'] + ['spearman_corr', 'spearman_pvalue', 'pearson_corr', 'pearson_pvalue']
header = "\t".join(header)

stats_file = "TRaC-stats.tsv"
if not Path(stats_file).is_file():
    lines = [header+'\n']
else:
    lines = open(stats_file, mode='r',encoding='utf-8').readlines()

lines.append(f'{model_name}\t' + f'{corr}\t{pvalue}\t{pearson_corr}\t{pearson_pvalue}\n')
lines.append(f'{model_name}_mask\t' + f'{mask_corr}\t{mask_pvalue}\t{mask_pearson_corr}\t{mask_pearson_pvalue}\n')

with open(stats_file, mode='w', encoding='utf-8') as f:
    f.writelines(lines)
