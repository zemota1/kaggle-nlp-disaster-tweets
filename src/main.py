import torch
import pandas as pd
from sklearn.model_selection import train_test_split

from dataset import TweetsDataset
from model import BERTModel
from engine import train_func, eval_func

PATH_TRAIN = '../data/train.csv'
RANDOM_STATE = 42
NUMBER_EPOCHS = 1


def pre_process(path_train):
    df_train = pd.read_csv(path_train)
    df_train = df_train.fillna('')
    df_train = df_train.iloc[:100, :]

    train, test = train_test_split(
        df_train,
        test_size=0.1,
        random_state=RANDOM_STATE,
        stratify=df_train.target.values
    )

    return train, test


def get_loaders(train_set, test_set):

    train_dataset = TweetsDataset(
        keyword=train_set.keyword,
        location=train_set.location,
        text=train_set.text,
        target=train_set.target
    )

    test_dataset = TweetsDataset(
        keyword=test_set.keyword,
        location=test_set.location,
        text=test_set.text,
        target=test_set.target
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=8,
        shuffle=False
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=4,
        shuffle=False
    )

    return train_loader, test_loader


def main():

    train_set, test_set = pre_process(PATH_TRAIN)
    train_data_loader, test_data_loader = get_loaders(train_set, test_set)

    model = BERTModel()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    for epoch in range(NUMBER_EPOCHS):
        train_func(data=train_data_loader, model=model, opt_param=optimizer_grouped_parameters, lr=1e-5)
        eval_func(data=test_data_loader, model=model)


if __name__ == '__main__':
    main()


