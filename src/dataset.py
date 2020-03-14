import numpy as np
import torch
import transformers
from nltk.tokenize import RegexpTokenizer


class TweetsDataset:

    def __init__(self, keyword, location, text, target):
        self.__keyword = keyword.values
        self.__location = location.values
        self.__text = text.values
        self.__target = target.values
        self.__tokenizer = transformers.BertTokenizer.from_pretrained(
            "../bert/",
            do_lower_case=True
        )
        self.__max_len = 300

    def get_tokenizer(self):
        return self.__tokenizer

    def get_max_len(self):
        return self.__max_len

    def get_keyword(self):
        return self.__keyword

    def get_location(self):
        return self.__location

    def get_text(self):
        return self.__text

    def get_target(self):
        return self.__target

    def __len__(self):
        return len(self.get_text())

    def __getitem__(self, item):
        keyword = str(self.get_keyword()[item])
        location = str(self.get_location()[item])
        text = str(self.get_text()[item])

        tokenizer = RegexpTokenizer(r'\w+')
        text_conc = '{} 0 {} 0 {}'.format(keyword, location, text)
        text_conc = tokenizer.tokenize(text_conc)

        encoded_sequence = self.get_tokenizer().encode_plus(
            text=text_conc,
            add_special_tokens=True,
            max_length=self.get_max_len()
        )

        inputs_ids_array = np.zeros(self.get_max_len())
        attention_mask_array = np.zeros(self.get_max_len())
        token_type_ids_array = np.zeros(self.get_max_len())
        offset = len(encoded_sequence['input_ids'])

        inputs_ids_array[:offset] = encoded_sequence['input_ids']
        attention_mask_array[:offset] = encoded_sequence['attention_mask']
        token_type_ids_array[:offset] = encoded_sequence['token_type_ids']

        return [
            torch.tensor(data=inputs_ids_array, dtype=torch.int64),
            torch.tensor(data=attention_mask_array, dtype=torch.int64),
            torch.tensor(data=token_type_ids_array, dtype=torch.int64),
            torch.tensor(data=self.get_target()[item], dtype=torch.float64)
        ]
