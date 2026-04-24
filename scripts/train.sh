#!/bin/bash
#SBATCH --job-name=train_hallu_cls
#SBATCH --output=./slurm_logs/train_%A.out
#SBATCH --error=./slurm_logs/train_%A.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1

set -eo pipefail

# train.py argument defaults
MODEL_NAME="gptj7b"
OUTPUT_PATH="./data/auto-labeled/output"
DATA_PATH="./data/auto-labeled/output"
STRATEGY="multi_layer" # ["original", "multi_layer", "multi_layer_last_token", "multi_layer_mean", "multi_layer_deltas"]
TRAIN_EPOCH=20
BATCH_SIZE=32
LR=5e-4
WD=1e-5
DROPOUT=0.2
DEVICE="cuda:0"

python src/train.py \
  --model_name "${MODEL_NAME}" \
  --output_path "${OUTPUT_PATH}" \
  --data_path "${DATA_PATH}" \
  --strategy "${STRATEGY}" \
  --train_epoch "${TRAIN_EPOCH}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --wd "${WD}" \
  --dropout "${DROPOUT}" \
  --device "${DEVICE}" \
  "$@"

echo "Done!"