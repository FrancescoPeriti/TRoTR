import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def load_uses(filename='TRoTR/data/uses.tsv', sep='\t'):
    tmp = list()
    with open(filename, mode='r', encoding='utf-8') as f:
        columns = f.readline().rstrip().split(sep)
        for line in f.readlines():
            tmp.append(dict(zip(columns, line.rstrip().split(sep))))

    return pd.DataFrame(tmp)


def load_instances(filename, dirname='rounds', sep='\t'):
    tmp = list()
    with open(f'TRoTR/{dirname}/{filename}', mode='r', encoding='utf-8') as f:
        columns = f.readline().rstrip().split(sep) + ['dataID1', 'dataID2']
        for line in f.readlines():
            tmp_record = dict(zip(columns, line[:-1].split('\t')))
            tmp_record['dataID1'], tmp_record['dataID2'] = tmp_record['dataIDs'].split(',')
            tmp.append(tmp_record)

    return pd.DataFrame(tmp)


def load_judgments(filename, dirname='judgments', sep='\t'):
    tmp = list()
    with open(f'TRoTR/{dirname}/{filename}', mode='r', encoding='utf-8') as f:
        columns = f.readline().rstrip().split(sep)
        for line in f.readlines():
            tmp_record = dict(zip(columns, line.rstrip().split(sep)))
            tmp.append(tmp_record)

    # -1: can not decide
    df = pd.DataFrame(tmp)
    df['label'] = df['label'].apply(lambda x: x.replace('-', '-1')).astype(int)

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

def split_rows(df):
    tmp = list()
    columns = ['instanceID', 'dataID1', 'dataID2', 'label', 'lemma', 'context{}',
               'context{}', 'indices_target_token{}', 'indices_target_sentence{}',
               'indices_target_sentence{}', 'indices_target_token{}', 'label_set',
               'non_label', 'dataIDs']
    for _, row in df.iterrows():
        for i in range(1, 3):
            record = dict()
            for c in columns:
                c = c.format(i)
                record[c.replace('1', '').replace('2', '')] = row[c]
            tmp.append(record)
    return pd.DataFrame(tmp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Random sampling', add_help=True)
    parser.add_argument('-a', '--annotators',
                        type=str,
                        default='Nisha shur AndreaMariaC', #iosakwe
                        help='Annotators')
    parser.add_argument('-v', '--vocab',
                        type=str,
                        default='out-of-vocabulary',  # iosakwe
                        choices=['in-of-vocabulary', 'out-of-vocabulary'])
    parser.add_argument('-f', '--filename',
                        type=str,
                        default='TRoTR.tsv',
                        help='Convert file from tsv to jsonl format')
    parser.add_argument('-s', '--subtask',
                        type=str,
                        default='binary',
                        help='Binary or ranking')
    parser.add_argument('-m', '--mode',
                        type=str,
                        default='line-by-line',
                        help='Specify the context format: "line-by-line" for one context per line, or "pair-by-line" for two contexts per line')
    args = parser.parse_args()

    annotators = args.annotators.split()
    vocab_mode = args.vocab

    round_ = args.filename
    df_uses = load_uses()
    df_instances = load_instances(round_)
    df_judgments = load_judgments(round_)
    df = merge_data(df_uses, df_instances, df_judgments)

    # remove cannot decide '-' and excluded annotators
    df = df[(df.label != -1) & (df.annotator.isin(annotators))]

    del df['comment']
    del df['annotator']
    df = df.groupby([c for c in df.columns.values if c != 'label']).mean().reset_index()

    if args.subtask == 'binary':
        # df = df[(df.label >= 3.5) | (df.label <= 1.5)].reset_index(drop=True)
        df['label'] = [int(label >= 3) for label in df.label.values]


    if vocab_mode == 'in-of-vocabulary':
        df = df.sample(frac=1, random_state=42)
        train, dev, test = np.split(df, [int(.6*len(df)), int(.8*len(df))])
    else:
        lemmas = df[['lemma']].drop_duplicates().sample(frac=1, random_state=42)
        train, dev, test = np.split(lemmas, [int(.6 * len(lemmas)), int(.8 * len(lemmas))])
        train = df[df['lemma'].isin(train.lemma.values)]
        dev = df[df['lemma'].isin(dev.lemma.values)]
        test = df[df['lemma'].isin(test.lemma.values)]

    if args.mode == 'line-by-line':
        train = split_rows(train)
        dev = split_rows(dev)
        test = split_rows(test)

    for k,v in {'train':train, 'dev':dev, 'test':test}.items():
        Path(f'TRoTR/datasets/{vocab_mode}/{args.mode}').mkdir(parents=True, exist_ok=True)
        v.to_json(f'TRoTR/datasets/{vocab_mode}/{args.mode}/{k}-{args.subtask}.jsonl', orient='records', lines=True)