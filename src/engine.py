import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import AdamW


def train_func(data, model, opt_param, lr):
    opt = get_optimizer(opt_param, lr)
    model.train()
    for _, value in tqdm(enumerate(data), total=len(data)):
        opt.zero_grad()

        ids = value[0]
        attention_mask = value[1]
        token_type_ids = value[2]
        targets = value[3].view(-1, 1)

        outputs = model(
            input_ids=ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        loss = loss_func(pred=outputs, true=targets)
        loss.backward()

        opt.step()


def eval_func(data, model):
    model.eval()
    true = []
    pred = []
    with torch.no_grad():
        for _, value in tqdm(enumerate(data), total=len(data)):
            ids = value[0]
            attention_mask = value[1]
            token_type_ids = value[2]
            targets = value[3].view(-1, 1)

            outputs = model(
                input_ids=ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            outputs = torch.gt(outputs, 0.5).double()

            true.extend(targets.squeeze().tolist())
            pred.extend(outputs.squeeze().tolist())
            
        acc = accuracy_score(y_true=true, y_pred=pred)
        print(f"Accuracy : {acc}")


def loss_func(pred, true):
    return nn.BCEWithLogitsLoss()(pred, true)


def get_optimizer(opt_param, lr):
    return AdamW(params=opt_param, lr=lr)
