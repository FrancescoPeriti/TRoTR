import json
import random
import argparse
from pathlib import Path
from itertools import combinations
from collections import defaultdict

# The Answer to the Great Question of Life, the Universe and Everything is Forty-two
random.seed(42)

def sample_instances(usesIDs:list, k: int) -> list:
    '''Sample instances to be annotated

    Args:
        k(int): number of instances to be annotated
    Returns:
        list of instances'''
    all_pairs = set()
    for pair in combinations(usesIDs, 2):
        id1, id2 = pair
        if id1 < id2:
            all_pairs.add(",".join([id1, id2]))
        elif id1 > id2: # exclude pair of records with the same id
            all_pairs.add(",".join([id2, id1]))

    return random.sample(list(all_pairs), min(k, len(all_pairs)))

def store(data: dict, filename: str) -> None:
    '''Store data in tsv files'''
    for u in data:
        Path(f'pic/{u.replace(":", " ")}/').mkdir(parents=True, exist_ok=True)
        with open(f'pic/{u.replace(":", " ")}/{filename}', mode='w', encoding='utf-8') as f:
            data[u] = data[u][:-1] + [data[u][-1][:-1]] # remove last '\n'
            f.writelines(data[u])

def aggregate(filename: str) -> None:
    full_data = list()
    for i, f in enumerate(Path(f'pic/').glob(f'*/{filename}')):
        with open(f, mode='r', encoding='utf-8') as f:
            data = f.readlines()
            if i != 0:
                data = data[1:]
            data = data[:-1] + [data[-1]+'\n']
            full_data.extend(data)

    with open(f'pic/{filename}', mode='w', encoding='utf-8') as f:
        f.writelines(full_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Random sampling', add_help=True)
    parser.add_argument('-k', '--sample_size',
                        type=int,
                        default=3,
                        help='Number of pairs to be annotated per quotation')
    args = parser.parse_args()

    # tweet collection
    filename = 'quotations.jsonl'

    # create uses.tsv file for each quotation
    uses = defaultdict(list)
    uses_header = "dataID\tcontext\tindices_target_token\tindices_target_sentence\tlemma\n"
    for line in open(filename, mode='r', encoding='utf-8').readlines():
        line = line.strip()
        dict_ = json.loads(line)
        if not dict_['quote_id'] in uses:
            uses[dict_['quote_id']].append(uses_header)
        uses[dict_['quote_id']].append(f"{dict_['id']}\t{dict_['sentence']}\t{dict_['start']}:{dict_['end']}\t0:{len(dict_['sentence'])}\t{dict_['quote']}\n")

    store(uses, "uses.tsv")
    aggregate("uses.tsv")

    # create instances.tsv file for each quotation
    instances_header = "instanceID\tdataIDs\tlabel_set\tnon_label\n"
    instances = defaultdict(list)
    for u in uses:
        usesIDs = [line.split('\t')[0] for line in uses[u][1:]]
        dataIDs = sample_instances(usesIDs, args.sample_size)
        instanceID = [f'pair_{i}_{u}' for i in range(len(dataIDs))]
        label_set = ["1,2,3,4"]*len(dataIDs)
        non_label = ["-"]*len(dataIDs)
        instances[u].append(instances_header)
        for i in range(len(dataIDs)):
            instances[u].append(f'{instanceID[i]}\t{dataIDs[i]}\t{label_set[i]}\t{non_label[i]}\n')

    store(instances, "instances.tsv")
    aggregate("instances.tsv")
