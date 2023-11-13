import argparse
import logging
import os
import random
import time
import json
import tempfile
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (precision_recall_fscore_support, classification_report, accuracy_score)
from collections import namedtuple
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, models, losses, InputExample, CrossEncoder
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
import torch
import random


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Baseline:

    def __init__(self,args):
        self.lr = args.lr
        self.n_epochs = args.n_epochs
        self.batch_size = args.batch_size
        self.device = args.device
        self.weight_decay = args.weight_decay
        self.output_path = args.output_path
        self.do_validation = args.do_validation
        self.train_path = args.train_path
        self.dev_path = args.dev_path
        self.pretrained_model = args.pretrained_model
        self.max_seq_length = args.max_seq_length
        self.warmup_percentage = args.warmup_percentage
        self.add_tags = args.add_tags
        self.loss = args.loss
        self.evaluation = args.evaluation
        self.finetune_sbert = args.finetune_sbert
        self.sbert_pretrained_model = args.sbert_pretrained_model
        self.pretrained_model = args.pretrained_model
        self.model_type = args.model_type


    def load_data(self, path, return_sentences=False, return_inputs=False):
        examples = []
        sentences = [[],[]]
        labels = []
        with (open(path) as f):
            for line in f:
                line = json.loads(line)
                sentence1 = line['context1']
                sentence2 = line['context2']

                if self.add_tags:
                    start1, end1 = line['indices_target_token1'].split(':')
                    start2, end2 = line['indices_target_token1'].split(':')
                    start1, end1 = int(start1), int(end1)
                    start2, end2 = int(start2), int(end2) 

                    new_sentence1 = sentence1[:start1] + '<t>' + sentence1[start1:end1] + '</t>' + sentence1[end1:]
                    new_sentence2 = sentence2[:start2] + '<t>' + sentence2[start2:end2] + '</t>' + sentence2[end2:]

                examples.append(InputExample(texts=[sentence1, sentence2], label=line['label']))
                sentences[0].append(sentence1)
                sentences[1].append(sentence2)
                labels.append(line['label'])

        dataloader = DataLoader(examples, shuffle=True, batch_size=self.batch_size)

        if return_sentences:
            return sentences, labels
        elif return_inputs:
            return [[sentences[0][j],sentences[1][j]] for j in range(len(sentences[0]))], labels
        else:
            return dataloader



    def init_model(self):
        if self.finetune_sbert:
            if self.model_type == 'crossencoder':
                self.model = CrossEncoder(self.sbert_pretrained_model,num_labels=1)
                if self.add_tags:
                    word_embedding_model = self.model.model
                    tokens = ["<t>", "</t>"]
                    self.model.tokenizer.add_tokens(tokens, special_tokens=True)
                    word_embedding_model.resize_token_embeddings(len(self.model.tokenizer))
            else:
                self.model = SentenceTransformer(self.sbert_pretrained_model)
                if self.add_tags:
                    word_embedding_model = self.model._first_module()
                    tokens = ["<t>", "</t>"]
                    word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
                    word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))
        else:
            word_embedding_model = models.Transformer(self.pretrained_model, max_seq_length=self.max_seq_length)
            if self.add_tags:
                tokens = ["<t>", "</t>"]
                word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
                word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
            self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=self.device)

        if self.loss == 'contrastive':
            self.train_loss = losses.ContrastiveLoss(self.model)

        if self.loss == 'mse':
            self.train_loss = losses.MSELoss(self.model)


    def train(self):
        class Callback(object):
            def __init__(self, warmup_steps, steps_per_epoch, model_name, output_path):
                self.best_score = 0
                self.patience = 10
                self.warmup_steps = warmup_steps
                self.steps_per_epoch = steps_per_epoch
                self.model_name = model_name
                self.output_path = output_path

            def __call__(self, score, epoch, steps):
                if score > self.best_score + 0.02:
                    self.best_score = score
                else:
                    if max(0, (self.steps_per_epoch * (epoch - 1))) + steps > warmup_steps:
                        if self.patience == 0:
                            print('Early stop training.')
                            with open(os.path.join(f"{self.output_path}",f"results.tsv"), 'a+') as f:
                                f.write(f'{self.model_name}\t{self.best_score}\n')
                            exit()
                        self.patience = self.patience - 1

        self.init_model()
        train_dataloader = self.load_data(self.train_path)
        warmup_steps = self.warmup_percentage * (len(train_dataloader) * self.n_epochs)
        evaluation_steps = int(0.25 * (len(train_dataloader)))

        if self.evaluation == 'binary' and self.loss == 'contrastive':
            dev_sentences, dev_labels = self.load_data(self.dev_path, return_sentences=True)
            evaluator = BinaryClassificationEvaluator(dev_sentences[0], dev_sentences[1], dev_labels)

        if self.evaluation == 'correlation':
            dev_sentences, dev_labels = self.load_data(self.dev_path, return_inputs=True)
            evaluator = CECorrelationEvaluator(dev_sentences, dev_labels)

        if self.model_type == 'siamese':
            self.model.fit(
                train_objectives=[(train_dataloader, self.train_loss)],
                epochs=self.n_epochs,
                optimizer_params={'lr': self.lr},
                warmup_steps=warmup_steps,
                evaluator=evaluator,
                callback=Callback(warmup_steps, len(train_dataloader),model_name=f"model_{self.lr}_{self.weight_decay}", output_path=self.output_path),
                evaluation_steps=evaluation_steps,
                output_path=os.path.join(f"{self.output_path}",f"model_{self.lr}_{self.weight_decay}"),
                weight_decay=self.weight_decay,
                show_progress_bar=False
            )
        else:
            self.model.fit(
                train_dataloader=train_dataloader,
                epochs=self.n_epochs,
                optimizer_params={'lr': self.lr},
                warmup_steps=warmup_steps,
                evaluator=evaluator,
                callback=Callback(warmup_steps, len(train_dataloader),model_name=f"model_{self.lr}_{self.weight_decay}", output_path=self.output_path),
                evaluation_steps=evaluation_steps,
                output_path=os.path.join(f"{self.output_path}",f"model_{self.lr}_{self.weight_decay}"),
                weight_decay=self.weight_decay,
                show_progress_bar=False
            )


    def predict(self):
        model = SentenceTransformer(self.pretrained_model)

        examples = self.load_dataset(args.test_path, model)
        features = model.convert_dataset_to_features(examples)
        test_dataloader = get_dataloader_and_tensors(features, self.batch_size)
        test_batches = [batch for batch in test_dataloader]

        test_bar = tqdm(test_batches, total=len(test_batches), desc='Evaluation - TEST set ...', leave=True, position=0)

        predictions = []
        gold_labels = []
        gold_scores = []
        for step, batch in enumerate(test_bar):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, token_type_ids, labels, positions = batch
            _, preds = model(input_ids=input_ids,
                             token_type_ids=token_type_ids,
                             attention_mask=input_mask,
                             input_labels={'labels': labels,
                                           'positions': positions}
                             )
            gold_scores.extend(labels)
            gold_labels.extend([int(l>=2.5) for l in labels])
            predictions.extend([v[0] for v in preds.detach().cpu().tolist()])

        ex2pred = {e.docId: predictions[j] for j, e in enumerate(examples)}

        labels = []
        scores = []
        Path(self.stats_path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(self.stats_path, 'ranking.jsonl'), 'w+') as f:
            with open(os.path.join(self.stats_path, 'binary.jsonl'), 'w+') as g:
                for ex in ex2pred:
                    rankingline = {'id': ex, 'label': ex2pred[ex]}
                    binaryline = {'id': ex, 'label': int(ex2pred[ex] >= 2.5)}
                    scores.append(rankingline['label'])
                    labels.append(binaryline['label'])

                    f.write(f'{json.dumps(binaryline)}\n')
                    g.write(f'{json.dumps(rankingline)}\n')

        print('Spearman\tMacro-F1')
        test_spearman, _ = spearmanr(scores, gold_scores)
        test_f1, _ = f1_score(labels, gold_labels, average='weighted')
        print(f'{round(test_spearman, 3)}\t{round(test_f1, 3)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Baseline model',
        description="Training of the baseline model")

    parser.add_argument('--lr', default=2e-5)
    parser.add_argument('--n_epochs', default=20)
    parser.add_argument('--batch_size', default=16)
    parser.add_argument('--weight_decay', default=0.0)
    parser.add_argument('--warmup_percentage', default=0.01)
    parser.add_argument('--loss', default="contrastive", choices=['ce', 'mse', 'contrastive'])
    parser.add_argument('--add_tags', default=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--train_path', default="")
    parser.add_argument('--evaluation', default="binary")
    parser.add_argument('--strategy', default='target', choices=['context', 'target'])
    parser.add_argument('--dev_path', default="")
    parser.add_argument('--test_path', default="")
    parser.add_argument('--model_type', default="siamese")
    parser.add_argument('--do_validation', default=True)
    parser.add_argument('--output_path', default='models')
    parser.add_argument('--finetune_sbert', default=False)
    parser.add_argument('--pretrained_model', default='roberta-large')
    parser.add_argument('--sbert_pretrained_model', default='paraphrase-multilingual-mpnet-base-v2')
    parser.add_argument('--max_seq_length', default=512)


    args = parser.parse_args()

    set_seed()

    b = Baseline(args)
    b.train()
