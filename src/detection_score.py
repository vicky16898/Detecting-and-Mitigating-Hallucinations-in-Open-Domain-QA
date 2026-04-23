import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default="cpu")
parser.add_argument("--task_name", type=str, default="helm")
parser.add_argument("--strategy", type=str, default="multi_layer",
                    choices=["original", "multi_layer", "multi_layer_last_token", "multi_layer_mean", "multi_layer_deltas"])
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
from tqdm import tqdm
import torch.nn.functional as F
import json
from sklearn.metrics import precision_recall_curve, auc
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils.multi_layer import get_model_config, get_input_size, get_feature_keys
from classifier import HalluClassifier

task_name = args.task_name
strategy = args.strategy


def get_AUC(preds, human_labels, pos_label=1, oneminus_pred=False):
    preds = [v for v in preds]
    assert len(preds) == len(human_labels)
    P, R, thre = precision_recall_curve(human_labels, preds, pos_label=pos_label)
    return auc(R, P) * 100


def load_classifier(input_size, path):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = HalluClassifier(input_size)
    state_dict = torch.load(path, map_location="cpu")["model_state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device


def eval_score(model, device, hd):
    input_ = torch.tensor([hd]).to(device)
    logits = model(input_)
    return F.softmax(logits, dim=1)[:, 1][0].item()


# ──────────────────────────────────────────────
# Main evaluation
# ──────────────────────────────────────────────

root_path = f"./data/{task_name}"
model_dirs = os.listdir(root_path + "/hd")
model_dirs = sorted(model_dirs)

feature_keys = get_feature_keys(strategy)

result_sent_halu = {"Our_score": {}}
result_psg_corr = {"Our_score": {}}
result_psg_halu = {"Our_score": {}}
result_sent_corr = {"Our_score": {}}

for mo in tqdm(model_dirs):
    ckpt_path = f"./data/auto-labeled/output/{mo}/{args.strategy}/train_log/best_acc_model.pt"
    if not os.path.exists(ckpt_path):
        print(f"Skipping {mo}: no checkpoint found at {ckpt_path}")
        continue

    # Compute input size from model config
    config = get_model_config(mo)
    input_size = get_input_size(
        config["hidden_dim"], config["num_layers"], strategy
    )
    print(f"\nModel: {mo} | Strategy: {strategy} | Input size: {input_size}")

    mlp, device = load_classifier(input_size, ckpt_path)

    if task_name == "helm":
        # Ablation strategies share the multi_layer HD file
        hd_strategy = "multi_layer" if strategy in ("multi_layer_last_token", "multi_layer_mean", "multi_layer_deltas") else strategy
        hd_result_path = f"{root_path}/hd/{mo}/hd_{hd_strategy}.json"

        # Fallback: try old-style hd.json for backward compatibility
        if not os.path.exists(hd_result_path):
            hd_result_path = f"{root_path}/hd/{mo}/hd.json"
            if not os.path.exists(hd_result_path):
                print(f"Skipping {mo}: no HD file found")
                continue

        labeled_path = f"{root_path}/data/{mo}/data.json"
        if not os.path.exists(labeled_path):
            print(f"Skipping {mo}: no label file at {labeled_path}")
            continue

        with open(hd_result_path) as f:
            hd = json.load(f)
        with open(labeled_path) as f:
            labeled = json.load(f)

        labels = []
        pre = []
        psglabels = []
        psgpre = []
        psglabelsbysent = []

        for k in labeled:
            dts = labeled[k]["sentences"]
            hds = hd[k]["sentences"]
            psg_bi = 0
            psg_not_bi = 0

            for dt, d in zip(dts, hds):
                # Concatenate all feature keys for the classifier input
                feature_vec = []
                for key in feature_keys:
                    if key in d:
                        feature_vec += d[key]

                score = eval_score(mlp, device, feature_vec)
                labels.append(dt["label"])
                pre.append(score)
                if dt["label"] == 1:
                    psg_bi = 1
                    psg_not_bi += 1

            psg_not_bi /= len(dts)

            # Passage-level features
            passage_vec = []
            for key in feature_keys:
                if key in hd[k]["passage"]:
                    passage_vec += hd[k]["passage"][key]

            psgscore = eval_score(mlp, device, passage_vec)
            psglabels.append(psg_bi)
            psgpre.append(psgscore)
            psglabelsbysent.append(psg_not_bi)

        roc_auc_hallu_s = get_AUC(pre, labels)
        roc_auc_fact_s = get_AUC([1 - x for x in pre], [1 - x for x in labels])
        roc_auc_hallu_p = get_AUC(psgpre, psglabels)
        roc_auc_fact_p = get_AUC(
            [1 - x for x in psgpre], [1 - x for x in psglabels]
        )
        corr = np.corrcoef(psgpre, psglabelsbysent)

        model_key = mo.split(".json")[0]
        result_sent_halu["Our_score"][model_key] = roc_auc_hallu_s
        result_psg_corr["Our_score"][model_key] = corr[0][1]
        result_psg_halu["Our_score"][model_key] = roc_auc_hallu_p
        result_sent_corr["Our_score"][model_key] = np.corrcoef(pre, labels)[0][1]

        print(f"  Sent AUC: {roc_auc_hallu_s:.2f} | Psg AUC: {roc_auc_hallu_p:.2f}")

# Save results
import pandas as pd

out_dir = os.path.join(root_path, "results", strategy)
os.makedirs(out_dir, exist_ok=True)

pd.DataFrame(result_sent_halu).to_csv(os.path.join(out_dir, "sent_halu.csv"))
pd.DataFrame(result_psg_halu).to_csv(os.path.join(out_dir, "psg_halu.csv"))
pd.DataFrame(result_sent_corr).to_csv(os.path.join(out_dir, "sent_corr.csv"))
pd.DataFrame(result_psg_corr).to_csv(os.path.join(out_dir, "psg_corr.csv"))

print(f"\nResults saved to {out_dir}/")
print("Sentence-level hallucination AUC:")
for k, v in result_sent_halu["Our_score"].items():
    print(f"  {k}: {v:.2f}")
