from torch.utils.data import DataLoader, TensorDataset
import torch
import os
import json
from collections import Counter
import pandas as pd
from glob import glob
import numpy as np
from collections import namedtuple

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class DataProcessor:

    def get_examples(self, fname):
        Example = namedtuple('Example', ['docId', 'text_1', 'text_2', 'start_1', 'end_1', 'start_2', 'end_2', 'label'])
        examples = []
        with open(fname) as f:
            for line in f:
                ex = json.loads(line)
                if 'label' in ex:
                    label = ex['label']
                else:
                    label = -1
                start1, end1 = ex['indices_target_token1'].split(':')
                start1, end1 = int(start1), int(end1)
                start2, end2 = ex['indices_target_token1'].split(':')
                start2, end2 = int(start2), int(end2)
                examples.append(Example(ex['instanceID'], ex['context1'], ex['context2'], start1, end1, start2, end2, label))
        return examples


def get_dataloader_and_tensors(features: list, batch_size: int):
    input_ids = torch.tensor(
        [f.input_ids for f in features],
        dtype=torch.long
    )
    input_mask = torch.tensor(
        [f.input_mask for f in features],
        dtype=torch.long
    )
    token_type_ids = torch.tensor(
        [f.token_type_ids for f in features],
        dtype=torch.long
    )
    syn_labels = torch.tensor(
        [f.syn_label for f in features]
    )
    positions = torch.tensor(
        [f.positions for f in features],
        dtype=torch.long
    )
    eval_data = TensorDataset(
        input_ids, input_mask, token_type_ids,
        syn_labels, positions
    )

    dataloader = DataLoader(eval_data, batch_size=batch_size)

    return dataloader

