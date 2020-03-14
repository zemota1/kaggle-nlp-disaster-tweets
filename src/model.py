import json

import torch.nn as nn
import transformers

with open('../bert/config.json', 'r') as cfg:
    data = cfg.read()

config = json.loads(data)


class BERTModel(nn.Module):

    def __init__(self):
        super(BERTModel, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(
            pretrained_model_name_or_path='../bert/'
        )
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(
            in_features=config['hidden_size'],
            out_features=1
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, out2 = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        x = self.dropout(out2)
        x = self.linear(x)
        return x
