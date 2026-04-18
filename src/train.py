import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from dataset import TrainDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.multi_layer import get_feature_keys, get_input_size, get_model_config


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _feature_dir(strategy: str) -> str:
    """Ablation strategies share the multi_layer feature files."""
    if strategy in ("multi_layer_last_token", "multi_layer_mean", "multi_layer_deltas"):
        return "multi_layer"
    return strategy


def _load_feature_files(root_path: str, data_model: str, split: str, feature_keys, strategy: str):
    """Load one JSON file per feature key for the given split."""
    features = {}
    feat_dir = _feature_dir(strategy)
    for key in feature_keys:
        path = os.path.join(root_path, data_model, feat_dir, f"{key}_{split}.json")
        if not os.path.exists(path):
            available = os.listdir(os.path.join(root_path, data_model, feat_dir))
            raise FileNotFoundError(
                f"Feature file not found: {path}\nAvailable files: {available}"
            )
        with open(path, encoding="utf-8") as f:
            features[key] = json.load(f)
    return features


def _collect_samples(feature_files, feature_keys):
    """
    Flatten per-key feature files into parallel lists of hallu/right vectors.

    Each sample in the source file has a "hallu" list (multiple hallucinations)
    and a single "right" vector. We flatten hallu across samples, then trim
    right to the same count so we can pair them 1:1.
    """
    hallu = {key: [] for key in feature_keys}
    right = {key: [] for key in feature_keys}
    for key in feature_keys:
        for sample in feature_files[key]:
            hallu[key].extend(sample["hallu"])
            right[key].append(sample["right"])

    num_pairs = len(hallu[feature_keys[0]])
    for key in feature_keys:
        right[key] = right[key][:num_pairs]
    return hallu, right, num_pairs


def get_data(root_path: str, data_model: str, split: str, strategy: str):
    """
    Load hidden-state features for one split and return a list of
    {"hd": concatenated_feature_vector, "label": 0|1} dicts.

    Label 0 = correct (right), label 1 = hallucinated.
    """
    feature_keys = get_feature_keys(strategy)
    feature_files = _load_feature_files(root_path, data_model, split, feature_keys, strategy)
    hallu, right, num_pairs = _collect_samples(feature_files, feature_keys)

    samples = []
    for i in range(num_pairs):
        right_vec = [v for key in feature_keys for v in right[key][i]]
        hallu_vec = [v for key in feature_keys for v in hallu[key][i]]
        samples.append({"hd": right_vec, "label": 0})
        samples.append({"hd": hallu_vec, "label": 1})
    return samples


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class HalluClassifier(nn.Module):
    """4-layer MLP that classifies a hidden-state vector as correct vs hallucinated."""

    def __init__(self, input_size: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, 256), nn.ReLU(),
            nn.Linear(256, 128),        nn.ReLU(),
            nn.Linear(128, 64),         nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, model: nn.Module, optimizer, loss_fn, device, log_dir: str, meta: dict):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.log_dir = log_dir
        self.meta = meta  # stored alongside checkpoints (strategy, model_name, ...)

    def save(self, name: str, valid_acc: float, epoch: int) -> None:
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "valid_acc": valid_acc,
            "epoch": epoch,
            **self.meta,
        }
        torch.save(ckpt, os.path.join(self.log_dir, f"{name}_model.pt"))

    def _run_epoch(self, loader: DataLoader, train: bool):
        self.model.train(train)
        total_loss = 0.0
        preds, labels = [], []

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for step, batch in enumerate(loader):
                inputs = batch["input"].to(self.device)
                # batch["y"] comes in as list-of-lists from TrainDataset
                y = torch.LongTensor([k[0] for k in batch["y"].tolist()]).to(self.device)

                logits = self.model(inputs)
                loss = self.loss_fn(logits, y)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()
                preds.extend(torch.argmax(logits, dim=1).tolist())
                labels.extend(y.tolist())

                if train and (step + 1) % 10 == 0:
                    pct = 100.0 * (step + 1) / len(loader)
                    print(f"  [{pct:5.1f}%] avg loss: {total_loss / (step + 1):.4f}")

        avg_loss = total_loss / max(len(loader), 1)
        return avg_loss, accuracy_score(labels, preds)

    def fit(self, train_loader: DataLoader, valid_loader: DataLoader, epochs: int):
        best_acc, best_epoch = -1.0, 0
        for epoch in range(1, epochs + 1):
            print(f"\n=== Epoch {epoch}/{epochs} ===")
            train_loss, train_acc = self._run_epoch(train_loader, train=True)
            print(f"Train | loss: {train_loss:.4f} | acc: {train_acc:.4f}")

            valid_loss, valid_acc = self._run_epoch(valid_loader, train=False)
            print(f"Valid | loss: {valid_loss:.4f} | acc: {valid_acc:.4f}")

            if valid_acc > best_acc:
                best_acc, best_epoch = valid_acc, epoch
                self.save("best_acc", valid_acc, epoch)

        self.save("last", valid_acc, epoch)
        print(f"\nBest valid acc: {best_acc:.4f} (epoch {best_epoch})")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def build_dataloaders(args):
    """
    NOTE: following the original paper's setup, we train on train+valid and
    use the *test* split for per-epoch validation/model selection.
    """
    train_samples = get_data(args.data_path, args.model_name, "train", args.strategy)
    valid_samples = get_data(args.data_path, args.model_name, "valid", args.strategy)
    test_samples  = get_data(args.data_path, args.model_name, "test",  args.strategy)

    train_ds = TrainDataset(train_samples + valid_samples, args)
    eval_ds  = TrainDataset(test_samples, args, typ="valid")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4)
    eval_loader  = DataLoader(eval_ds,  batch_size=args.batch_size // 2,
                              shuffle=False, num_workers=4)
    return train_loader, eval_loader, train_ds


def build_loss_fn(train_ds, device):
    """Class-balanced cross-entropy, weighted by inverse class frequency."""
    n_hallu  = train_ds.halu_num
    n_right  = len(train_ds) - n_hallu
    total    = n_right + n_hallu
    weights  = torch.tensor([1 - n_right / total, 1 - n_hallu / total],
                            dtype=torch.float, device=device)
    return nn.CrossEntropyLoss(weight=weights)


def build_optimizer(model: nn.Module, lr: float, wd: float):
    no_decay = ("bias", "LayerNorm.bias", "LayerNorm.weight")
    decay_params, nodecay_params = [], []
    for name, p in model.named_parameters():
        (nodecay_params if any(nd in name for nd in no_decay) else decay_params).append(p)
    groups = [
        {"params": decay_params,   "weight_decay": wd,  "lr": lr},
        {"params": nodecay_params, "weight_decay": 0.0, "lr": lr},
    ]
    return torch.optim.Adam(groups)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",  type=str, default="llama3base8b")
    parser.add_argument("--output_path", type=str, default="./data/auto-labeled/output")
    parser.add_argument("--data_path",   type=str, default="./data/auto-labeled/output")
    parser.add_argument("--strategy",    type=str, default="multi_layer",
                        choices=["original", "multi_layer", "multi_layer_last_token", "multi_layer_mean", "multi_layer_deltas"])
    parser.add_argument("--train_epoch", type=int,   default=20)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=5e-4)
    parser.add_argument("--wd",          type=float, default=1e-5)
    parser.add_argument("--dropout",     type=float, default=0.2)
    parser.add_argument("--device",      type=str,   default="cpu")
    return parser.parse_args()


def main():
    setup_seed(0)
    args = parse_args()

    config = get_model_config(args.model_name)
    input_size = get_input_size(config["hidden_dim"], config["num_layers"], args.strategy)
    print(f"Model: {args.model_name} | Strategy: {args.strategy} | Input size: {input_size}")

    model = HalluClassifier(input_size, dropout=args.dropout).to(args.device)

    train_loader, eval_loader, train_ds = build_dataloaders(args)
    loss_fn   = build_loss_fn(train_ds, args.device)
    optimizer = build_optimizer(model, args.lr, args.wd)

    log_dir = os.path.join(args.output_path, args.model_name, args.strategy, "train_log")
    os.makedirs(log_dir, exist_ok=True)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=args.device,
        log_dir=log_dir,
        meta={"strategy": args.strategy, "model_name": args.model_name},
    )
    trainer.fit(train_loader, eval_loader, epochs=args.train_epoch)
    print(f"Model: {args.model_name} | Strategy: {args.strategy}")


if __name__ == "__main__":
    main()