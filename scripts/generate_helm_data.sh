#!/bin/bash
#SBATCH --job-name=helm_data_gen
#SBATCH --output=./slurm_logs/helm_data_gen_%A.out
#SBATCH --error=./slurm_logs/helm_data_gen_%A.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=18G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu               
#SBATCH --gres=gpu:h200:1

python src/generate_helm_data.py --model_family gptj --model_type 7 --gpu 0

echo "Done!"
