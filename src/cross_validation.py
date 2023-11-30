import argparse
import math
import pandas as pd
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


def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

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
    parser.add_argument('-hm', '--home',
                        type=str,
                        default='TRoTR',
                        help='TRoTR home')
    parser.add_argument('-s', '--subtask',
                        type=str,
                        default='binary',
                        help='Binary or ranking')
    parser.add_argument('-n', '--n_folds',
                        type=int,
                        default=10,
                        help='N. folds in Cross-validation')
    args = parser.parse_args()

    annotators = args.annotators.split()

    round_ = args.filename
    n_folds = args.n_folds
    home = args.home
    df_uses = load_uses(home)
    df_instances = load_instances(home, round_)
    df_judgments = load_judgments(home, round_)
    df = merge_data(df_uses, df_instances, df_judgments)
    df = df[df.annotator.isin(annotators)] # excluded annotators
    df = handle_cannot_decide(df)

    del df['comment']
    del df['annotator']
    #df = df.groupby([c for c in df.columns.values if c != 'label']).mean().reset_index()

    # processing: remove every pair with an average judgment between 2 and 3 and pair with max-min judgment > 1
    # remove instances with judgment difference greater than 1
    tmp = df[['instanceID', 'label']].groupby('instanceID').agg(['unique']).reset_index()
    tmp[('label', 'unique')] = [max(i) - min(i) < 2 for i in tmp[('label', 'unique')]]
    filtered_instances = tmp[tmp[('label', 'unique')]].instanceID.values
    
    # remove instances with avg judgment between 2 and 3
    tmp = df[['instanceID', 'label']].groupby('instanceID').mean().reset_index()
    tmp['label'] = [0 if i <= 2 else i for i in tmp['label']]
    tmp['label'] = [1 if i >= 3 else i for i in tmp['label']]
    tmp = tmp[(tmp['label'].isin([0,1])) & (tmp['instanceID'].isin(filtered_instances))]
    filtered_instances = tmp.instanceID.values
    
    df = df[df.instanceID.isin(filtered_instances)]

    df = df[['instanceID', 'dataID1', 'dataID2', 'label', 'lemma', 'context1', 'context2', 'indices_target_token1',
             'indices_target_sentence1', 'indices_target_sentence2', 'indices_target_token2', 'dataIDs', 'label_set', 'non_label']].groupby(['instanceID', 
                                                                                                                                             'dataID1', 'dataID2', 
                                                                                                                                            'lemma',
                                                                                                                                             'context1', 'context2', 
                                                                                                                                             'indices_target_token1',
                                                                                                                                             'indices_target_sentence1', 
                                                                                                                                             'indices_target_sentence2',
                                                                                                                                             'indices_target_token2', 'dataIDs', 'label_set', 'non_label']).mean().reset_index()
    
    if args.subtask == 'binary':
        df['label'] = [0 if i <= 2 else i for i in df['label']]
        df['label'] = [1 if i >= 3 else i for i in df['label']]
        
    lemmas = df['lemma'].unique()

    #Generating OUT Folds
    lemmas_out = split(lemmas, n_folds)

    for j, fold in enumerate(lemmas_out):
        # split per lemma
        train = df[~df['lemma'].isin(fold)]  # 90% of targets  -> # Everything except the current fold
        dev_out_lemmas, test_out_lemmas = split(fold, 2) # 10% of targets -> Split in two 0.5% subfolds
        dev_out = df[df['lemma'].isin(dev_out_lemmas)] # 0.5 % of targets
        test_out = df[df['lemma'].isin(test_out_lemmas)] # 0.5 % of targets

        # train split to have shared train set per dev and test
        train = train.sample(frac=1, random_state=42).reset_index(drop=True)
        n_example_fold = pd.concat([dev_out, test_out]).shape[0]
        dev_in = train.loc[0:n_example_fold//2] # 0.5 % of data from in-vocabulary
        test_in = train.loc[n_example_fold//2: n_example_fold]  # 0.5 % of data from in-vocabulary
        train = train.loc[n_example_fold:] # from 0.9% of data to 0.8% of data
        dev = pd.concat([dev_in, dev_out]) # from 0.5% of data to 10% of data (0.5% out-of-vocabulary, 0.5% in-of-vocabulary)
        test = pd.concat([test_in, test_out])

        print(f'FOLD {fold}')
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
        print('-----------------------------------------------------------')

        train_lbl = split_rows(train)
        dev_lbl = split_rows(dev)
        dev_lbl_in = split_rows(dev_in)
        dev_lbl_out = split_rows(dev_out)
        test_lbl = split_rows(test)
        test_lbl_in = split_rows(test_in)
        test_lbl_out = split_rows(test_out)

        for k, v in {'train': (train,train_lbl),
                     'dev':(dev, dev_lbl),
                     'test': (test, test_lbl),
                     'test.iov': (test_in, test_lbl_in),
                     'test.oov': (test_out, test_lbl_out),
                     'dev.iov': (dev_in, dev_lbl_in),
                     'dev.oov': (dev_out, dev_lbl_out)}.items():
            Path(f'TRoTR/datasets/FOLD_{j+1}').mkdir(parents=True, exist_ok=True)
            Path(f'TRoTR/datasets/FOLD_{j+1}/line-by-line').mkdir(parents=True, exist_ok=True)
            v[1].to_json(f'TRoTR/datasets/FOLD_{j+1}/line-by-line/{k}.{args.subtask}.jsonl', orient='records', lines=True)
            Path(f'TRoTR/datasets/FOLD_{j+1}/pair-by-line').mkdir(parents=True, exist_ok=True)
            v[0].to_json(f'TRoTR/datasets/FOLD_{j+1}/pair-by-line/{k}.{args.subtask}.jsonl', orient='records', lines=True)
