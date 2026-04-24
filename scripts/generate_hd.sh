#!/bin/bash
#SBATCH --job-name=data_gen
#SBATCH --output=./slurm_logs/data_gen_%A.out
#SBATCH --error=./slurm_logs/data_gen_%A.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=18G
#SBATCH --time=00:30:00
#SBATCH --partition=gpu               
#SBATCH --gres=gpu:h200:1

python src/generate_hd.py --model_family gptj --model_type 7 --strategy multi_layer --gpu 0

echo "Done!"