import torch
from torch import nn
from typing import Optional
from torch.nn import MSELoss
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

from transformers import BertPreTrainedModel

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, num_classes, input_size):
        super().__init__()
        self.out_proj = nn.Linear(input_size, num_classes)

    def forward(self, features, **kwargs):
        x = features
        x = self.out_proj(x)
        return x


class Feature:
    def __init__(self, input_ids: list, input_mask: list, token_type_ids: list, label: float, positions: list, example: int):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.positions = positions
        self.example = example


class CLSModel(BertPreTrainedModel): #PreTrainedModel):

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self._model = AutoModel.from_config(config)
        self._tokenizer = AutoTokenizer.from_pretrained(config._name_or_path)
        self._input_size = 512 #config.hidden_size
        self._max_seq_len = config.max_position_embeddings
        self._clf = ClassificationHead(1, self._input_size * 2) # two embeddings will be concatenated
        self.init_weights()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                input_labels: Optional[torch.Tensor] = None
                ):

        loss = defaultdict(float)
        outputs = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequences_output = outputs[0]  # bs x seq x hidden

        labels = input_labels['labels']  # bs
        positions = input_labels['positions'] # bs x 4 - (i.e., 4: start1, end1, start2, end2)

        features = self.extract_features(sequences_output, positions) # bs x hidden

        # clf.forward is called
        logits = self._clf(features)  # bs x 2 or bs

        if input_labels is not None:
            loss['total'] = MSELoss()(logits, labels.unsqueeze(-1).float())

        return (loss, logits)


    def extract_features(self,
                         hidden_states: torch.tensor,
                         positions: torch.tensor):

        bs, seq, hs = hidden_states.size()
        features = []
        for ex_id in range(bs):
            start1, end1, start2, end2 = positions[ex_id, 0].item(), positions[ex_id, 1].item(), positions[ex_id, 2].item(), positions[ex_id, 3].item()
            emb1 = hidden_states[ex_id, start1:end1].mean(axis=0)
            emb2 = hidden_states[ex_id, start2:end2].mean(axis=0)
            merged_feature = torch.cat((emb1, emb2))
            features.append(merged_feature.unsqueeze(0))

        output = torch.cat(features, dim=0)
        return output


    def convert_dataset_to_features(self, examples: object):
        features = []
        num_too_long_exs = 0

        for ex_index, ex in enumerate(examples):
            sent1 = ex.text_1
            sent2 = ex.text_2
            start1, end1, start2, end2 = int(ex.start_1), int(ex.end_1), int(ex.start_2), int(ex.end_2)
            tokens = [self._tokenizer.cls_token]

            positions = [0,0,0,0]
            left1, target1, right1 = sent1[:start1], sent1[start1:end1], sent1[end1:]
            left2, target2, right2 = sent2[:start2], sent2[start2:end2], sent2[end2:]

            if left1:
                tokens += self._tokenizer.tokenize(left1)

            positions[0] = len(tokens)
            target_subtokens = self._tokenizer.tokenize(target1)
            tokens += target_subtokens
            positions[1] = len(tokens)

            if right1:
                tokens += self._tokenizer.tokenize(right1) + [self._tokenizer.sep_token]
            if left2:
                tokens += self._tokenizer.tokenize(left2)

            positions[2] = len(tokens)
            target_subtokens = self._tokenizer.tokenize(target2)
            tokens += target_subtokens
            positions[3] = len(tokens)

            if right2:
                tokens += self._tokenizer.tokenize(right2) + [self._tokenizer.sep_token]

            input_ids = self._tokenizer.convert_tokens_to_ids(tokens)
            if len(input_ids) > self._max_seq_len:
                input_ids = input_ids[:self._max_seq_len]
                num_too_long_exs += 1

                # target is beyond the max_sequence_len
                if max(positions) > self._max_seq_len - 1:
                    continue

            input_mask = [1] * len(input_ids)
            token_type_ids = [0] * self._max_seq_len
            padding = [self._tokenizer.convert_tokens_to_ids([self._tokenizer.pad_token])[0]] * (self._max_seq_len - len(input_ids))
            input_ids += padding
            input_mask += [0] * len(padding)

            features.append(
                Feature(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    token_type_ids=token_type_ids,
                    label=ex.label,
                    positions=positions,
                    example=ex
                    )
                )
        return features
