import json
import math
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertPreTrainedModel, AutoTokenizer, AutoModel
from transformers.configuration_utils import PretrainedConfig
from sentence_transformers import models
import torch
from torch import Tensor
from torch import nn
from typing import Union, Tuple, List, Iterable, Dict
import os
import json
from sentence_transformers import models


class SubstringPooling(models.Pooling):
    def __init__(self, model_size: int):
        super(SubstringPooling, self).__init__(word_embedding_dimension=model_size,
                                               pooling_mode_mean_tokens=True)

    def substring_mean(tensor, substring_mask):
        # Set embeddings of non-target words to zero
        tensor_masked = torch.tensor(tensor * substring_mask.unsqueeze(-1)).sum(axis=0)

        # Count the number of target words
        n_targets = torch.sum(substring_mask, dim=1)

        # Calculate the mean of target embeddings, avoiding division by zero
        return tensor_masked / (n_targets + 1e-8)

    def forward(self, features: Dict[str, Tensor]):
        token_embeddings = features['token_embeddings']
        substring_mask = features['substring_mask']

        ## Pooling strategy
        output_vectors = [self.substring_mean(token_embeddings, substring_mask)]
        output_vector = torch.cat(output_vectors, 1)
        features.update({'sentence_embedding': output_vector})
        return features

class Feature:
    def __init__(self,
                 input_ids: list,
                 input_mask: list,
                 substring_mask: list,
                 label: float,
                 token_type_ids: list,
                 example: int):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.substring_mask = substring_mask
        self.example = example

class BERTModel(models.Transformer):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self._model = AutoModel.from_config(config)
        self._tokenizer = AutoTokenizer.from_pretrained(config._name_or_path)
        self._input_size = config.hidden_size
        self._max_seq_len = config.max_position_embeddings
        self.init_weights()

    def convert_dataset_to_features(self, examples: list):
        features = []
        num_too_long_exs = 0

        for ex_index, ex in enumerate(examples):
            sent1 = ex['context1']
            sent2 = ex['context2']
            start1, end1 = [int(i) for i in ex['indices_target_token1'].split(':')]
            start2, end2 = [int(i) for i in ex['indices_target_token2'].split(':')]

            tokens = [self._tokenizer.cls_token]

            substring_mask = []
            left1, target1, right1 = sent1[:start1], sent1[start1:end1], sent1[end1:]
            left2, target2, right2 = sent2[:start2], sent2[start2:end2], sent2[end2:]

            if left1:
                tokens += self._tokenizer.tokenize(left1)
                substring_mask += [0] * len(tokens)

            target_tokens = self._tokenizer.tokenize(target1)
            substring_mask += [1] * len(target_tokens)
            tokens += target_tokens

            if right1:
                right_tokens = self._tokenizer.tokenize(right1) + [self._tokenizer.sep_token]
                substring_mask += [0] * len(right_tokens)
                tokens += right_tokens

            if left2:
                left_tokens = self._tokenizer.tokenize(left2)
                substring_mask += [0] * len(left_tokens)
                tokens += left_tokens

            target_tokens = self._tokenizer.tokenize(target2)
            substring_mask += [1] * len(target_tokens)
            tokens += target_tokens

            if right2:
                right_tokens = self._tokenizer.tokenize(right2) + [self._tokenizer.sep_token]
                substring_mask += [0] * len(right_tokens)
                tokens += right_tokens

            input_ids = self._tokenizer.convert_tokens_to_ids(tokens)
            if len(input_ids) > self._max_seq_len:
                input_ids = input_ids[:self._max_seq_len]
                num_too_long_exs += 1

            input_mask = [1] * len(input_ids)
            padding = [self._tokenizer.convert_tokens_to_ids([self._tokenizer.pad_token])[0]] * (self._max_seq_len - len(input_ids))
            input_ids += padding
            substring_mask += [0] * len(padding)
            input_mask += [0] * len(padding)
            token_type_ids = [0] * self._max_seq_len

            features.append(
                Feature(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    token_type_ids=token_type_ids,
                    label=ex.label,
                    substring_mask=substring_mask,
                    example=ex
                )
            )
        return features

    def load_dataset(self, fname: str):
        examples = get_examples(fname)
        features = self.convert_dataset_to_features(examples)
        random.shuffle(features)
        dataloader = get_dataloader(features, batch_size)
        batches = [batch for batch in dataloader]
        return batches

def get_examples(file_name: str):
    with open(file_name, mode='r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def get_dataloader(features: list, batch_size: int):
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
    labels = torch.tensor(
        [f.label for f in features]
    )
    substring_mask = torch.tensor(
        [f.substring_mask for f in features],
        dtype=torch.long
    )

    eval_data = TensorDataset(
        input_ids, input_mask, token_type_ids,
        labels, substring_mask
    )

    dataloader = DataLoader(eval_data, batch_size=batch_size)

    return dataloader






pretrained_model='xlm-roberta-base'
train_path='filename'
batch_size=16
device='cuda'
num_epochs=2
warmup_step=0.1 #10% of train data for warm-up
model_save_path='output'
evaluation_steps=100

model = BERTModel.from_pretrained(pretrained_model)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = SubstringPooling(model_size=model._max_seq_len,)

model = SentenceTransformer(modules=[model, pooling_model], device=device)
train_loss = losses.ContrastiveLoss(model)

train_dataloader = model.load_dataset(train_path, model)

warmup_steps = math.ceil(len(train_dataloader) * num_epochs * warmup_step)

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
          epochs=num_epochs,
          evaluation_steps=evaluation_steps,
          warmup_steps=warmup_steps,
          output_path=model_save_path)