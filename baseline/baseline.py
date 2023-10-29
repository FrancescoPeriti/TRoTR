from utils import DataProcessor, get_dataloader_and_tensors, set_seed
from xlmr import XLMRModel
import argparse
import logging
import os
import random
import time
import json
from datetime import datetime
import tempfile
import shutil
import numpy as np
import pandas as pd
import torch
from transformers.optimization import (AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup)
from transformers.file_utils import (PYTORCH_PRETRAINED_BERT_CACHE,WEIGHTS_NAME, CONFIG_NAME)
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import (precision_recall_fscore_support, classification_report, accuracy_score)
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from scipy.stats import spearmanr


class Baseline:

    def __init__(self,args):
        self.lr = args.lr
        self.n_epochs = args.n_epochs
        self.batch_size = args.batch_size
        self.device = args.device
        self.weight_decay = args.weight_decay
        self.max_grad_norm = args.max_grad_norm
        self.best_model_path = args.best_model_path
        self.do_validation = args.do_validation
        self.train_path = args.train_path
        self.dev_path = args.dev_path
        self.found_model = args.found_model

    def load_dataset(self, fname, model):
        data_processor = DataProcessor()
        examples = data_processor.get_examples(fname)
        features = model.convert_dataset_to_features(examples)
        random.shuffle(features)
        dataloader = get_dataloader_and_tensors(features, self.batch_size)
        batches = [batch for batch in dataloader]
        return  batches

    def train(self):
        model = XLMRModel.from_pretrained(self.found_model)
        train_batches = self.load_dataset(self.train_path, model)
        dev_batches = self.load_dataset(self.dev_path, model)

        param_optimizer = list(model.named_parameters())
        optimizer_parameters = [{'params': [param for name, param in param_optimizer], 'weight_decay': float(self.weight_decay)}]
        optimizer = AdamW(
            optimizer_parameters,
            lr=float(self.lr),
        )
        model.to(self.device)

        best_dev_result = 0.
        
        for epoch in range(1, 1 + self.n_epochs):
            tr_loss = 0
            nb_tr_examples = 0
            nb_tr_steps = 0
            cur_train_loss = defaultdict(float)

            model.train()
            random.shuffle(train_batches)

            train_bar = tqdm(train_batches, total=len(train_batches), desc='training ... ')

            for step, batch in enumerate(train_bar):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, token_type_ids, syn_labels, positions = batch
                train_loss, _ = model(input_ids=input_ids,
                                      token_type_ids=token_type_ids,
                                      attention_mask=input_mask,
                                      input_labels={'syn_labels': syn_labels, 'positions': positions}
                )
                loss = train_loss['total'].mean().item()
                for key in train_loss:
                    cur_train_loss[key] += train_loss[key].mean().item()

                loss_to_optimize = train_loss['total']

                loss_to_optimize.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),self.max_grad_norm)

                tr_loss += loss_to_optimize.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                optimizer.step()
                optimizer.zero_grad()

            model.eval()

            if self.do_validation:

                dev_bar = tqdm(dev_batches, total=len(dev_batches), desc='evaluation DEV ... ')

                truth = []
                predictions = []

                for step, batch in enumerate(dev_bar):
                    batch = tuple(t.to(self.device) for t in batch)
                    input_ids, input_mask, token_type_ids, syn_labels, positions = batch
                    _, preds = model(input_ids=input_ids,
                                          token_type_ids=token_type_ids,
                                          attention_mask=input_mask,
                                          input_labels={'syn_labels': syn_labels, 'positions': positions}
                                          )
                    predictions = predictions + [v[0] for v in preds.detach().cpu().tolist()]
                    truth = truth + syn_labels.detach().cpu().tolist()


                dev_spearman,_ = spearmanr(truth,predictions)

                print('Correlation DEV SET',dev_spearman)

            if dev_spearman > best_dev_result:
                #self.save_model(model)
                best_dev_result = dev_spearman

            model.train()

    def save_model(self, model):
        if os.path.exists(self.best_model_path):
            shutil.rmtree(self.best_model_path)
        os.mkdir(self.best_model_path)
        model_to_save = model.module if hasattr(model, 'module') else model
        output_config_file = os.path.join(self.best_model_path, 'model.conf')
        torch.save(model_to_save.state_dict(), os.path.join(self.best_model_path,'model.pt'))
        model_to_save.config.to_json_file(os.path.join(self.best_model_path,'model.json'))

        for fname, binary_name, ranking_name in [('ranking/test.jsonl', 'binary.jsonl', 'ranking.jsonl'), ('ranking/test_eng.jsonl', 'binary_eng.jsonl', 'ranking_eng.jsonl')]:
            examples = b.load_dataset(fname)
            features = model.convert_dataset_to_features(examples)
            test_dataloader = get_dataloader_and_tensors(features, self.batch_size)
            test_batches = [batch for batch in test_dataloader]

            test_bar = tqdm(test_batches, total=len(test_batches), desc='evaluation TEST ... ')

            predictions = []

            for step, batch in enumerate(test_bar):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, token_type_ids, syn_labels, positions = batch
                _, preds = model(input_ids=input_ids,
                                 token_type_ids=token_type_ids,
                                 attention_mask=input_mask,
                                 input_labels={'syn_labels': syn_labels, 'positions': positions}
                                 )
                predictions = predictions +  [v[0] for v in preds.detach().cpu().tolist()]

            ex2pred = {}
            for j,e in enumerate(examples):
                ex2pred[e.docId] = predictions[j]


            with open(os.path.join(self.best_model_path,binary_name),'w+') as f:
                with open(os.path.join(self.best_model_path,ranking_name),'w+') as g:
                    for ex in ex2pred:
                        rankingline = {'id':ex, 'label':ex2pred[ex]}
                        binaryline = {'id':ex}
                        if ex2pred[ex] < 2:
                            binaryline['label'] = 0
                        else:
                            binaryline['label'] = 1
                        f.write(f'{json.dumps(binaryline)}\n')
                        g.write(f'{json.dumps(rankingline)}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Baseline model',
        description="Training of the baseline model")

    parser.add_argument('--lr', default=1e-5)
    parser.add_argument('--n_epochs', default=20)
    parser.add_argument('--batch_size', default=16)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--weight_decay', default=0.)
    parser.add_argument('--max_grad_norm', default=1.)
    parser.add_argument('--train_path', default="")
    parser.add_argument('--loss', default="contrastive", choices=['ce', 'mse', 'contrastive'])
    parser.add_argument('--strategy', default='target', choices=['context', 'target'])
    parser.add_argument('--dev_path', default="")
    parser.add_argument('--test_path', default="")
    parser.add_argument('--do_validation', default=True)
    parser.add_argument('--best_model_path', default='best_model')
    parser.add_argument('--found_model', default='roberta-large')

    args = parser.parse_args()

    set_seed()

    b = Baseline(args)
    b.train()
