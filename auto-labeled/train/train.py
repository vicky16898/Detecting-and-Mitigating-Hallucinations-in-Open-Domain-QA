import os
import sys
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import datetime
from dataset import TrainDataset
import json
import numpy as np
from sklearn.metrics import accuracy_score
import random

# Add parent directories to path so we can import utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.multi_layer import get_model_config, get_input_size, get_feature_keys


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(0)


def binary_eval(predy, testy):
    acc = accuracy_score(testy, predy)
    return acc


def get_data(root_path, data_model, type_="train", strategy="original"):
    """
    Load hidden state features for training/validation/test.

    Args:
        root_path: Path to the data directory
        data_model: Model name (e.g., "llama3base8b")
        type_: Data split ("train", "valid", "test")
        strategy: Feature strategy ("original" or "multi_layer")

    Returns:
        List of {"hd": feature_vector, "label": 0 or 1} dicts
    """
    feature_keys = get_feature_keys(strategy)

    # Load all feature files
    feature_data = {}
    for key in feature_keys:
        path = os.path.join(root_path, data_model, f"{key}_{type_}.json")
        if not os.path.exists(path):
            print(f"ERROR: Feature file not found: {path}")
            print(f"Available files: {os.listdir(os.path.join(root_path, data_model))}")
            sys.exit(1)
        feature_data[key] = json.load(open(path, encoding='utf-8'))

    # Build combined feature vectors
    num_samples = len(feature_data[feature_keys[0]])
    halu_features = {key: [] for key in feature_keys}
    right_features = {key: [] for key in feature_keys}

    for key in feature_keys:
        for sample in feature_data[key]:
            halu_features[key] += sample["hallu"]
            right_features[key].append(sample["right"])

    num_hallu = len(halu_features[feature_keys[0]])

    # Trim right to match hallu count
    for key in feature_keys:
        right_features[key] = right_features[key][:num_hallu]

    # Concatenate all feature keys into single vectors
    enddata = []
    for i in range(num_hallu):
        right_vec = []
        hallu_vec = []
        for key in feature_keys:
            right_vec += right_features[key][i]
            hallu_vec += halu_features[key][i]

        enddata.append({"hd": right_vec, "label": 0})
        enddata.append({"hd": hallu_vec, "label": 1})

    return enddata


class Model():
    def __init__(self, args, path=None):
        self.args = args

        # Compute input size based on model config and strategy
        config = get_model_config(args.model_name)
        input_size = get_input_size(
            config["hidden_dim"], config["num_layers"], args.strategy
        )
        print(f"Model: {args.model_name} | Strategy: {args.strategy} | Input size: {input_size}")

        self.model = nn.Sequential()
        self.model.add_module("dropout", nn.Dropout(args.dropout))
        self.model.add_module("linear1", nn.Linear(input_size, 256))
        self.model.add_module("relu1", nn.ReLU())
        self.model.add_module("linear2", nn.Linear(256, 128))
        self.model.add_module("relu2", nn.ReLU())
        self.model.add_module("linear3", nn.Linear(128, 64))
        self.model.add_module("relu3", nn.ReLU())
        self.model.add_module("linear4", nn.Linear(64, 2))

        if path is not None:
            self.model.load_state_dict(
                torch.load(path, map_location="cpu")["model_state_dict"]
            )
        self.model.to(args.device)

    def save(self, acc, ei, prefix, name):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "valid_acc": acc,
                "epoch": ei,
                "strategy": self.args.strategy,
                "model_name": self.args.model_name,
            },
            prefix + f"{name}_model.pt",
        )

    def run(self, optim):
        prefix = f"{self.args.output_path}/{self.args.model_name}/train_log/"
        if not os.path.exists(prefix):
            os.makedirs(prefix, exist_ok=True)
        epoch, epoch_start = self.args.train_epoch, 1

        train_data = get_data(
            self.args.data_path, self.args.model_name,
            "train", self.args.strategy
        )
        valid_data = get_data(
            self.args.data_path, self.args.model_name,
            "valid", self.args.strategy
        )
        test_data = get_data(
            self.args.data_path, self.args.model_name,
            "test", self.args.strategy
        )

        # Use train+valid for training, test for validation
        # (following the original paper's setup)
        rtrain_data = train_data + valid_data
        train_dataset = TrainDataset(rtrain_data, self.args)
        valid_dataset = TrainDataset(test_data, self.args, typ="valid")
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4,
        )
        valid_dataloader = DataLoader(
            dataset=valid_dataset,
            batch_size=self.args.batch_size // 2,
            shuffle=False,
            num_workers=4,
        )
        nSamples = [
            len(train_dataset) - train_dataset.halu_num,
            train_dataset.halu_num,
        ]
        normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
        normedWeights = torch.FloatTensor(normedWeights).to(self.args.device)
        loss_func = nn.CrossEntropyLoss(weight=normedWeights).to(self.args.device)

        best_acc = -1
        best_epoch = [0]
        for ei in range(epoch_start, epoch + 1):
            cnt = 0
            self.model.train()
            train_loss = 0
            predy, trainy, hallu_sm_score = [], [], []
            for step, batch in enumerate(train_dataloader):
                input_ = batch["input"].to(self.args.device)
                label_ids = torch.LongTensor(
                    [k[0] for k in batch["y"].tolist()]
                ).to(self.args.device)
                score = self.model(input_)
                hallu_sm = F.softmax(score, dim=1)[:, 1]
                _, pred = torch.max(score, dim=1)

                trainy.extend(label_ids.tolist())
                predy.extend(pred.tolist())
                hallu_sm_score.extend(hallu_sm.tolist())
                loss = loss_func(score, label_ids)
                train_loss += loss.item()
                optim.zero_grad()
                loss.backward()
                optim.step()
                cnt += 1
                if cnt % 10 == 0:
                    print(
                        "Training Epoch {} - {:.2f}% - Loss : {}".format(
                            ei, 100.0 * cnt / len(train_dataloader), train_loss / cnt
                        )
                    )
            print("Training Epoch {} ...".format(ei))
            acc = binary_eval(predy, trainy)
            print(
                "Train Epoch {} end ! Loss : {}; Train Acc: {}".format(
                    ei, train_loss, acc
                )
            )

            self.model.eval()
            predy, validy, hallu_sm_score = [], [], []
            valid_loss = 0
            for step, batch in enumerate(valid_dataloader):
                input_ = batch["input"].to(self.args.device)
                label_ids = torch.LongTensor(
                    [k[0] for k in batch["y"].tolist()]
                ).to(self.args.device)
                score = self.model(input_)
                hallu_sm = F.softmax(score, dim=1)[:, 1]
                _, pred = torch.max(score, dim=1)
                validy.extend(label_ids.tolist())
                predy.extend(pred.tolist())
                hallu_sm_score.extend(hallu_sm.tolist())
                loss = loss_func(score, label_ids)
                valid_loss += loss.item()
            print("Valid Epoch {} ...".format(ei))

            acc = binary_eval(predy, validy)

            if acc > best_acc:
                best_acc = acc
                best_epoch[0] = ei
                self.save(acc, ei, prefix, "best_acc")

        self.save(acc, ei, prefix, "last")
        print(f"Best acc : {best_acc} from epoch {best_epoch[0]}th;")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llama3base8b")
    parser.add_argument("--output_path", default="../auto-labeled/output", type=str)
    parser.add_argument("--data_path", default="../auto-labeled/output", type=str)
    parser.add_argument("--strategy", type=str, default="multi_layer",
                        choices=["original", "multi_layer"])

    parser.add_argument("--train_epoch", default=20, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--wd", default=1e-5, type=float)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--device", default="cuda:0", type=str)

    args = parser.parse_args()

    model = Model(args)

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optim_func = torch.optim.Adam
    named_params = list(model.model.named_parameters())
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in named_params if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.wd,
            "lr": args.lr,
        },
        {
            "params": [
                p for n, p in named_params if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": args.lr,
        },
    ]
    optimizer = optim_func(optimizer_grouped_parameters)

    model.run(optimizer)
    print(f"Model: {args.model_name} | Strategy: {args.strategy}")