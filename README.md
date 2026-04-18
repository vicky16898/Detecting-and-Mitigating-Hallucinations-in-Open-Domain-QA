# Detecting and Mitigating Hallucinations in Open-Domain QA

This project extends the [MIND paper (ACL 2024)](https://arxiv.org/abs/2407.12943) with multi-layer hidden-state features and a HELM evaluation pipeline.

## Environment

```bash
conda create -n odtformer python=3.9
conda activate odtformer
pip install torch==2.0.1
pip install -r requirements.txt
```

On the cluster all scripts are run via `sbatch scripts/<name>.sh`. Logs go to `slurm_logs/`.

---

## Training a classifier for a model / strategy

Run these steps in order. All commands are run from the repo root.

### 1. Generate auto-labeled data

Builds hallucinated/correct sentence pairs from Wikipedia.

```bash
python src/generate_data.py --model_family <family> --model_type <size> --gpu 0
```

Output: `data/auto-labeled/output/<model>/data_{train,valid,test}.json`

### 2. Extract training features

```bash
python src/generate_hd.py --model_family <family> --model_type <size> --strategy <strategy> --gpu 0
```

- `--strategy original` — 2-channel features (last-token avg + last-layer mean), dim = `2 × hidden_dim`
- `--strategy multi_layer` — 10-channel features (5 last-token layers + 3 mean-pool layers + 2 deltas), dim = `10 × hidden_dim`
- `--strategy multi_layer_last_token` — last-token concat only (subset of `multi_layer`), dim = `5 × hidden_dim`
- `--strategy multi_layer_mean` — mean-pool only (subset of `multi_layer`), dim = `3 × hidden_dim`
- `--strategy multi_layer_deltas` — layer deltas only (subset of `multi_layer`), dim = `2 × hidden_dim`

The three `multi_layer_*` ablation strategies reuse the feature files written by `--strategy multi_layer`, so no separate `generate_hd.py` run is needed for them.

Output: `data/auto-labeled/output/<model>/<strategy>/{feature_key}_{split}.json` for each key returned by `get_feature_keys(strategy)`.

### 3. Train the classifier

```bash
python src/train.py --model_name <model> --strategy <strategy> --device cuda:0
```

The MLP input size is derived automatically from the model config — never hardcode it.

Output: `data/auto-labeled/output/<model>/<strategy>/train_log/best_acc_model.pt`

---

## Evaluation on HELM

HELM evaluation requires labeled continuation data and hidden-state features for the HELM prompts. These steps assume the classifier from the training section above already exists.

### 1. Generate model continuations (if not already present)

```bash
python src/generate_helm_data.py --model_family <family> --model_type <size> --gpu 0
```

Output: `data/helm/data/<model>/data.json`

### 2. Label continuations for hallucination

Uses the OpenAI Batch API (requires `OPENAI_API_KEY`). Run three times in sequence:

```bash
python src/label_helm_data.py --model_name <model> --mode submit
python src/label_helm_data.py --model_name <model> --mode status   # repeat until done
python src/label_helm_data.py --model_name <model> --mode fetch
```

Labels are written back into `data/helm/data/<model>/data.json`.

### 3. Extract HELM hidden-state features

```bash
python src/generate_hd_for_helm.py --model_family <family> --model_type <size> --strategy <strategy> --gpu 0
```

Output: `data/helm/hd/<model>/hd_<strategy>.json`

### 4. Run detection scoring

```bash
python src/detection_score.py --strategy <strategy>
```

Requires both `data/helm/hd/<model>/hd_<strategy>.json` and `data/auto-labeled/output/<model>/<strategy>/train_log/best_acc_model.pt` to exist for a model to be scored.

Output CSVs written to `data/helm/results/<strategy>/`:
- `sent_halu.csv` — sentence-level hallucination AUC
- `psg_halu.csv` — passage-level hallucination AUC
- `sent_corr.csv` — sentence-level correlation
- `psg_corr.csv` — passage-level correlation

To view results as a table:

```bash
python src/show_results.py                    # all strategies
python src/show_results.py --strategy original
```

---

## Adding a new model

1. Add an entry to `MODEL_CONFIGS` in `src/utils/multi_layer.py` with the correct `hidden_dim` and `num_layers`.
2. Run the full training pipeline (steps 1–3 above).
3. Run the full HELM evaluation pipeline (steps 1–4 above).

## Adding a new strategy

1. Update `get_feature_keys`, `extract_features`, and `get_input_size` in `src/utils/multi_layer.py` — all three must agree.
2. Retrain from scratch; old checkpoints are incompatible with a changed input dim.
3. Add the new strategy to the `choices` list in any script that accepts `--strategy`.
