import argparse
import math
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from collections import defaultdict
from pathlib import Path


def load_uses(home, filename='data/uses.tsv', sep='\t'):
    df = list()
    with open(f'{home}/{filename}', mode='r', encoding='utf-8') as f:
        columns = f.readline().rstrip().split(sep)
        for line in f.readlines():
            df.append(dict(zip(columns, line.rstrip().split(sep))))

    return pd.DataFrame(df)


def load_instances(home, filename, dirname='rounds', sep='\t'):
    df = list()
    with open(f'{home}/{dirname}/{filename}', mode='r', encoding='utf-8') as f:
        columns = f.readline().rstrip().split(sep) + ['dataID1', 'dataID2']
        for line in f.readlines():
            record = dict(zip(columns, line[:-1].split('\t')))
            record['dataID1'], record['dataID2'] = record['dataIDs'].split(',')
            df.append(record)

    return pd.DataFrame(df)


def load_judgments(home, filename, dirname='judgments', sep='\t'):
    df = list()
    with open(f'{home}/{dirname}/{filename}', mode='r', encoding='utf-8') as f:
        columns = f.readline().rstrip().split(sep)
        for line in f.readlines():
            record = dict(zip(columns, line.rstrip().split(sep)))
            if record['label'] == '-':
                record['label'] = math.nan
            df.append(record)

    df = pd.DataFrame(df)
    df['label'] = df['label'].astype(float)

    return df


def merge_data(df_uses, df_instances, df_judgments):
    df = df_judgments.merge(df_instances).merge(df_uses, left_on='dataID1', right_on='dataID')
    del df['dataID']
    del df['lemma']
    df = df.rename(
        columns={column: f'{column}1' for column in ['context', 'indices_target_token', 'indices_target_sentence']})
    df = df.merge(df_uses, left_on='dataID2', right_on='dataID')
    del df['dataID']
    df = df.rename(
        columns={column: f'{column}2' for column in ['context', 'indices_target_token', 'indices_target_sentence']})

    column_order = ['instanceID', 'dataID1', 'dataID2', 'label', 'annotator', 'lemma', 'context1', 'context2',
                    'indices_target_token1', 'indices_target_sentence1', 'indices_target_sentence2',
                    'indices_target_token2', 'comment', 'label_set', 'non_label', 'dataIDs']
    return df[column_order]


def handle_cannot_decide(df):
    df_cannot_decide = df[~df['label'].isin([1, 2, 3, 4])].fillna('-')
    df_cannot_decide = df_cannot_decide.groupby(['instanceID', 'label']).count().reset_index()
    instances_to_remove = df_cannot_decide[df_cannot_decide['annotator'] > 1].instanceID.values

    # exclude pairs for which more than one annotator couldn't decide
    df = df[~df['instanceID'].isin(instances_to_remove)]

    # remove all nan judgments
    df = df[~df['label'].isna()]
    return df


def inter_annotator_agreement_per_target(df, targets=None, instances=None):
    targets = df.lemma.unique() if targets is None else targets
    instances = df.instanceID.unique() if instances is None else instances
    df = df[df['lemma'].isin(targets) & df['instanceID'].isin(instances)]

    pairwise_spearman = defaultdict(list)

    annotators = df.annotator.unique()
    targets = df.lemma.unique()
    instances_dict = defaultdict(int)
    for target in targets:
        for annotator1 in annotators:
            for annotator2 in annotators:
                if annotator1 == annotator2: continue

                df1 = df[(df['annotator'] == annotator1) & (df['lemma'] == target)]
                df2 = df[(df['annotator'] == annotator2) & (df['lemma'] == target)]

                if df2.shape[0] > df1.shape[0]:
                    df1, df2 = df2, df1

                instances = df2.instanceID.values
                instances_dict[target] = max(instances.shape[0], instances_dict[target])
                df1 = df1[df1['instanceID'].isin(instances)].sort_values('instanceID')
                df2 = df2[df2['instanceID'].isin(instances)].sort_values('instanceID')
                corr, pvalue = spearmanr(df1.label.values, df2.label.values, nan_policy='omit')
                if corr == corr:  # != math.nan
                    pairwise_spearman[target].append(corr)

    df_res = pd.DataFrame()
    df_res['lemma'] = targets
    df_res['avg_pairwise_spearman_agreement'] = [np.mean(pairwise_spearman[target]).round(3) for target in targets]
    df_res['n_instances'] = [instances_dict[target] for target in targets]

    return df_res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Random sampling', add_help=True)
    parser.add_argument('-a', '--annotators',
                        type=str,
                        default='Nisha AndreaMariaC iosakwe shur',
                        help='Annotators')
    parser.add_argument('-i', '--inter_ann_agreement',
                        type=float,
                        default=0.2,
                        help='Minimum inter-annotator agreement (avg. pairwise spearman)')
    parser.add_argument('-f', '--filename',
                        type=str,
                        default='TRoTR.tsv',
                        help='Convert file from tsv to jsonl format')
    parser.add_argument('-hm', '--home',
                        type=str,
                        default='TRoTR',
                        help='TRoTR home')
    args = parser.parse_args()

    annotators = args.annotators.split()

    round_ = args.filename
    home = args.home
    df_uses = load_uses(home)
    df_instances = load_instances(home, round_)
    df_judgments = load_judgments(home, round_)
    df = merge_data(df_uses, df_instances, df_judgments)
    df = df[df.annotator.isin(annotators)]  # excluded annotators

    # filtering
    aps_targets = inter_annotator_agreement_per_target(df)
    aps_targets = aps_targets[aps_targets['avg_pairwise_spearman_agreement'] > args.inter_ann_agreement]
    lemmas = aps_targets['lemma'].unique()

    # Generating OUT Folds
    df = df[df.lemma.isin(lemmas)]
    df = handle_cannot_decide(df)
    df = df[['lemma', 'label']].groupby('lemma').mean().reset_index()

    # processing
    df['label'] = 1 - df['label'] / df['label'].values.max()
    df = df.rename(columns={'label':'topic_variation'})

    Path('TRoTR/datasets/').mkdir(parents=True, exist_ok=True)
    df.sort_values('lemma').to_csv('TRoTR/datasets/trac_gold_scores.tsv', index=False)