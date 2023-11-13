import os
import json
import torch
import random
import shutil
import argparse
from tqdm import tqdm
from pathlib import Path
from reg_head import RegModel
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
from collections import defaultdict
from transformers.optimization import AdamW #torch.optim.AdamW
from utils import DataProcessor, get_dataloader_and_tensors, set_seed
from transformers import AutoTokenizer, AutoModel, logging as transformers_logging

# avoid boring logging
transformers_logging.set_verbosity_error()

class TRiCModel:

    def __init__(self, args):
        self.lr = args.learning_rate
        self.n_epochs = args.n_epochs
        self.batch_size = args.batch_size
        self.device = args.device
        self.weight_decay = args.weight_decay
        self.max_grad_norm = args.max_grad_norm
        self.best_model_path = args.best_model_path
        self.do_validation = args.do_validation
        self.train_path = args.train_path
        self.dev_path = args.dev_path
        self.test_path = args.test_path
        self.stats_path = args.stats_path
        self.pretrained_model = args.pretrained_model
        self.accum_iter = args.accum_iter

    def load_dataset(self, fname:str, model:RegModel) -> list:
        data_processor = DataProcessor()
        examples = data_processor.get_examples(fname)
        features = model.convert_dataset_to_features(examples)
        random.shuffle(features)
        dataloader = get_dataloader_and_tensors(features, self.batch_size)
        batches = [batch for batch in dataloader]
        return batches

    def train(self):
        model = RegModel.from_pretrained(self.pretrained_model)
        train_batches = self.load_dataset(self.train_path, model)
        dev_batches = self.load_dataset(self.dev_path, model)

        param_optimizer = list(model.named_parameters())
        optimizer_parameters = [{'params':
                                     [param for name, param in param_optimizer],
                                 'weight_decay': float(self.weight_decay)}]
        optimizer = torch.optim.AdamW( #AdamW(
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

            train_bar = tqdm(train_batches, total=len(train_batches), desc=f'Training ... (epoch: {epoch})', leave=True,)

            for step, batch in enumerate(train_bar):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, token_type_ids, labels, positions, sentence_position = batch
                train_loss, _ = model(input_ids=input_ids,
                                      token_type_ids=token_type_ids,
                                      attention_mask=input_mask,
                                      input_labels={'labels': labels, 'positions': positions, 'sentence_position':sentence_position}
                )

                for key in train_loss:
                    cur_train_loss[key] += train_loss[key].mean().item()

                # normalize loss to account for batch accumulation
                loss_to_optimize = train_loss['total'] / self.accum_iter

                loss_to_optimize.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)

                tr_loss += loss_to_optimize.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                # weights update
                if ((step + 1) % self.accum_iter == 0) or (step + 1 == len(train_batches)):
                    optimizer.step()
                    optimizer.zero_grad()

            model.eval()

            if self.do_validation:

                dev_bar = tqdm(dev_batches, total=len(dev_batches), desc='evaluation DEV ... ', leave=True, position=0)

                truth = []
                predictions = []

                for step, batch in enumerate(dev_bar):
                    batch = tuple(t.to(self.device) for t in batch)
                    input_ids, input_mask, token_type_ids, labels, positions, sentence_position = batch
                    _, preds = model(input_ids=input_ids,
                                          token_type_ids=token_type_ids,
                                          attention_mask=input_mask,
                                          input_labels={'labels': labels, 'positions': positions, 'sentence_position':sentence_position}
                                          )
                    predictions = predictions + [v[0] for v in preds.detach().cpu().tolist()]
                    truth = truth + labels.detach().cpu().tolist()

                dev_spearman, _ = spearmanr(truth, predictions)
                print('Spearman Correlation - DEV set:', dev_spearman)

            if dev_spearman > best_dev_result:
                self.save_model(model)
                best_dev_result = dev_spearman

            model.train()

    def save_model(self, model: RegModel):
        if os.path.exists(self.best_model_path):
            shutil.rmtree(self.best_model_path)

        os.mkdir(self.best_model_path)
        model_to_save = model.module if hasattr(model, 'module') else model
        output_config_file = os.path.join(self.best_model_path, 'model.conf') ##
        torch.save(model_to_save.state_dict(), os.path.join(self.best_model_path, 'model.pt'))
        model_to_save.config.to_json_file(os.path.join(self.best_model_path, 'config.json'))

    def predict(self):
        model = RegModel.from_pretrained(self.pretrained_model)
        model.load_state_dict(torch.load(os.path.join(self.best_model_path, 'model.pt')))
        model.to(self.device)
        model.eval()
        #model._reg.eval()

        test_batches = self.load_dataset(self.test_path, model)
        test_bar = tqdm(test_batches, total=len(test_batches), desc='Evaluation - TEST set ...', leave=True, position=0)

        predictions = []
        gold_labels = []
        gold_scores = []
        for step, batch in enumerate(test_bar):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, token_type_ids, labels, positions, sentence_position = batch
            _, preds = model(input_ids=input_ids,
                             token_type_ids=token_type_ids,
                             attention_mask=input_mask,
                             input_labels={'labels': labels,
                                           'positions': positions,
                                           'sentence_position':sentence_position}
                             )

            gold_scores.extend(labels.detach().cpu().tolist())
            gold_labels.extend([int(l>=2.5) for l in labels.detach().cpu().tolist()])
            predictions.extend([v[0] for v in preds.detach().cpu().tolist()])

        scores = predictions
        labels = [int(p>= 2.5) for p in predictions]
        #labels = []
        #scores = []
        #Path(self.stats_path).mkdir(parents=True, exist_ok=True)
        #with open(os.path.join(self.stats_path, 'ranking.jsonl'), 'w+') as f:
        #    with open(os.path.join(self.stats_path, 'binary.jsonl'), 'w+') as g:
        #        for ex in ex2pred:
        #            rankingline = {'id': ex, 'label': ex2pred[ex]}
        #            binaryline = {'id': ex, 'label': int(ex2pred[ex] >= 2.5)}
        #            scores.append(rankingline['label'])
        #            labels.append(binaryline['label'])

        #            f.write(f'{json.dumps(binaryline)}\n')
        #            g.write(f'{json.dumps(rankingline)}\n')

        print('Spearman\tMacro-F1')
        test_spearman, _ = spearmanr(scores, gold_scores)
        test_f1 = f1_score(labels, gold_labels, average='weighted')
        print(f'{round(test_spearman, 3)}\t{round(test_f1, 3)}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='TRiCModel',
        description="Training of the TRiC model")

    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-6)
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--max_grad_norm', type=float, default=1.)
    parser.add_argument('--train_path', type=str, default="")
    parser.add_argument('--dev_path', type=str, default="")
    parser.add_argument('--test_path', type=str, default="")
    parser.add_argument('--stats_path', type=str, default="")
    parser.add_argument('--loss', type=str, default="contrastive", choices=['ce', 'mse', 'contrastive'])
    parser.add_argument('--strategy', type=str, default='target', choices=['context', 'target'])
    parser.add_argument('--do_validation', action='store_true')
    parser.add_argument('--do_training', action='store_true')
    parser.add_argument('--do_prediction', action='store_true')
    parser.add_argument('--best_model_path', type=str, default='TRoBERTa')
    parser.add_argument('--pretrained_model', type=str, default='roberta-large')
    parser.add_argument('--accum_iter', type=int, default=4, help='batch accumulation parameter (gradient-accumulation)')

    args = parser.parse_args()

    # 'The ultimate answer to the great question of life, the universe and everything'
    set_seed(42)

    model = TRiCModel(args)

    if args.do_training:
        model.train()

    if args.do_prediction:
        model.predict()
