from transformers import RobertaModel, XLMRobertaConfig
from transformers import BertPreTrainedModel
from transformers import AutoTokenizer, AutoModel
from torch import nn
from torch.nn import MSELoss
import torch
from collections import defaultdict
from torch import Tensor
import torch.nn.functional as F


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, num_classes, input_size):
        super().__init__()
        self.out_proj = nn.Linear(input_size, num_classes)

    def forward(self, features, **kwargs):
        x = features
        x = self.out_proj(x)
        return x


class WiCFeature2:
    def __init__(self, input_ids, input_mask, token_type_ids, syn_label, positions, example):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.token_type_ids = token_type_ids
        self.syn_label = syn_label
        self.positions = positions
        self.example = example



class XLMRModel(BertPreTrainedModel):

    def __init__(self, config : XLMRobertaConfig):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-large')
        self.input_size = self.roberta.config.hidden_size
        self.max_seq_len = 512

        self.input_size *= 2
        print('Classification head input size:', self.input_size)
        self.clf = RobertaClassificationHead(1, self.input_size)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            input_labels=None,
    ):
        loss = defaultdict(float)
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequences_output = outputs[0]  # bs x seq x hidden

        syn_labels = input_labels['syn_labels']  # bs
        positions = input_labels['positions'] # bs x 4

        syn_features = self.extract_features(sequences_output, positions) # bs x hidden
        syn_logits = self.clf(syn_features)  # bs x 2 or bs

        if input_labels is not None:
            loss['total'] = MSELoss()(syn_logits, syn_labels.unsqueeze(-1).float())

        return (loss, syn_logits)


    def extract_features(self, hidden_states, positions):
        bs, seq, hs = hidden_states.size()
        features = []
        for ex_id in range(bs):
            start1, end1, start2, end2 = positions[ex_id, 0].item(), positions[ex_id, 1].item(), positions[ex_id, 2].item(), positions[ex_id, 3].item()
            emb1 = hidden_states[ex_id, start1]
            emb2 = hidden_states[ex_id, start2]
            merged_feature = torch.cat((emb1, emb2))
            features.append(merged_feature.unsqueeze(0))
        output = torch.cat(features, dim=0)
        return output


    def convert_dataset_to_features(self, examples):
        features = []
        max_seq_len = self.max_seq_len
        num_too_long_exs = 0

        for ex_index, ex in enumerate(examples):
            sent1 = ex.text_1
            sent2 = ex.text_2
            st1, end1, st2, end2 = int(ex.start_1), int(ex.end_1), int(ex.start_2), int(ex.end_2)
            tokens = [self.tokenizer.cls_token]

            positions = [0,0,0,0]
            left1, target1, right1 = sent1[:st1], sent1[st1:end1], sent1[end1:]
            left2, target2, right2 = sent2[:st2], sent2[st2:end2], sent2[end2:]

            if left1:
                tokens += self.tokenizer.tokenize(left1)
            positions[0] = len(tokens)
            target_subtokens = self.tokenizer.tokenize(target1)
            tokens += target_subtokens
            positions[1] = len(tokens)

            if right1:
                tokens += self.tokenizer.tokenize(right1) + [self.tokenizer.sep_token]
            if left2:
                tokens += self.tokenizer.tokenize(left2)

            positions[2] = len(tokens)
            target_subtokens = self.tokenizer.tokenize(target2)
            tokens += target_subtokens
            positions[3] = len(tokens)
            if right2:
                tokens += self.tokenizer.tokenize(right2) + [self.tokenizer.sep_token]

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            if len(input_ids) > max_seq_len:
                input_ids = input_ids[:max_seq_len]
                num_too_long_exs += 1
                if max(positions) > max_seq_len - 1:
                    continue

            input_mask = [1] * len(input_ids)
            token_type_ids = [0] * max_seq_len
            padding = [self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]] * (max_seq_len - len(input_ids))
            input_ids += padding
            input_mask += [0] * len(padding)

            features.append(
                WiCFeature2(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    token_type_ids=token_type_ids,
                    syn_label=ex.label,
                    positions=positions,
                    example=ex
                    )
                )
        return features
