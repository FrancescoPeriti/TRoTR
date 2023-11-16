import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def std_judgments(df, lemma=False):
    dfs = list()

    if not lemma:
        for ann in df.annotator.unique():
            tmp = df[df['annotator']==ann].copy()
            avg = np.mean(tmp.label[tmp.label >= 1].values)
            tmp.label = [i if i!=-1 else avg for i in tmp.label]
            tmp.label = (tmp.label - tmp.label.mean())/(tmp.label.std())
            dfs.append(tmp)
        return pd.concat(dfs)
    if lemma:
        for ann in df.annotator.unique():
            for lemma in df.lemma.unique():
                tmp = df[(df['annotator']==ann) & (df['lemma']==lemma)].copy()
                avg = np.mean(tmp.label[tmp.label >= 1].values)
                tmp.label = [i if i!=-1 else avg for i in tmp.label]
                tmp.label = (tmp.label - tmp.label.mean())/(tmp.label.std())
                dfs.append(tmp)
        return pd.concat(dfs)


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
                        default='Nisha AndreaMariaC iosakwe shur',
                        help='Annotators')
    parser.add_argument('-f', '--filename',
                        type=str,
                        default='TRoTR.tsv',
                        help='Convert file from tsv to jsonl format')
    parser.add_argument('-s', '--subtask',
                        type=str,
                        default='binary',
                        help='Binary or ranking')
    args = parser.parse_args()

    annotators = args.annotators.split()

    round_ = args.filename
    df_uses = load_uses()
    df_instances = load_instances(round_)
    df_judgments = load_judgments(round_)
    df = merge_data(df_uses, df_instances, df_judgments)

    # remove cannot decide '-' and excluded annotators
    df = df[(df.label != -1) & (df.annotator.isin(annotators))]

    df = std_judgments(df, lemma=False)

    del df['comment']
    del df['annotator']
    df = df.groupby([c for c in df.columns.values if c != 'label']).mean().reset_index()

    if args.subtask == 'binary':
        # df = df[(df.label >= 3.5) | (df.label <= 1.5)].reset_index(drop=True)
        df['label'] = [int(label > 0) for label in df.label.values]

    lemmas = df[['lemma']].drop_duplicates().sample(frac=1, random_state=42)
    # split per lemma
    train, dev_out, test_out = np.split(lemmas, [int(.8 * len(lemmas)), int(.9 * len(lemmas))])
    train = df[df['lemma'].isin(train.lemma.values)] # 80% of targets
    dev_out = df[df['lemma'].isin(dev_out.lemma.values)] # 10% of targets
    test_out = df[df['lemma'].isin(test_out.lemma.values)] # 10% of targets
    # train split to have shared train set per dev and test
    train = train.sample(frac=1, random_state=42)
    train, dev_in, test_in = np.split(train, [int(.7 * len(train)), int(.85 * len(train))])
    dev = pd.concat([dev_in, dev_out]) # 10% of out-of-vocabulary + 10% of train
    print('-- Train:', train.shape[0], '--')
    print('- 0:', train[train['label'] <= 0].shape[0])
    print('- 1:', train[train['label'] > 0].shape[0])
    print('-- Dev:', dev.shape[0], '--')
    print('- 0:', dev[dev['label'] <= 0].shape[0])
    print('- 1:', dev[dev['label'] > 0].shape[0])
    print('Dev (in/out):', dev_in.shape[0], '/', dev_out.shape[0])
    print('- 0 (in/out):', dev_in[dev_in['label'] <= 0].shape[0], '/', dev_out[dev_out['label'] <= 0].shape[0])
    print('- 1 (in/out):', dev_in[dev_in['label'] > 0].shape[0], '/', dev_out[dev_out['label'] > 0].shape[0])
    print('Test (in/out):', test_in.shape[0], '/', test_out.shape[0])
    print('- 0 (in/out):', test_in[test_in['label'] <= 0].shape[0], '/', test_out[test_out['label'] <= 0].shape[0])
    print('- 1 (in/out):', test_in[test_in['label'] > 0].shape[0], '/', test_out[test_out['label'] > 0].shape[0])

    train_lbl = split_rows(train)
    dev_lbl = split_rows(dev)
    dev_lbl_in = split_rows(dev_in)
    dev_lbl_out = split_rows(dev_out)
    test_lbl_in = split_rows(test_in)
    test_lbl_out = split_rows(test_out)

    for k, v in {'train': (train,train_lbl),
                 'dev':(dev, dev_lbl),
                 'test.iov': (test_in, test_lbl_in),
                 'test.oov': (test_out, test_lbl_out),
                 'dev.iov': (dev_in, dev_lbl_in),
                 'dev.oov': (dev_out, dev_lbl_out)}.items():
        Path(f'TRoTR/datasets/line-by-line').mkdir(parents=True, exist_ok=True)
        v[1].to_json(f'TRoTR/datasets/line-by-line/{k}.{args.subtask}.jsonl', orient='records', lines=True)
        Path(f'TRoTR/datasets/pair-by-line').mkdir(parents=True, exist_ok=True)
        v[0].to_json(f'TRoTR/datasets/pair-by-line/{k}.{args.subtask}.jsonl', orient='records', lines=True)