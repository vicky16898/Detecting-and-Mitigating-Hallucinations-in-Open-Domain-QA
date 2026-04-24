#!/bin/bash
#SBATCH --job-name=data_gen
#SBATCH --output=./slurm_logs/data_gen_%A.out
#SBATCH --error=./slurm_logs/data_gen_%A.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=18G
#SBATCH --time=08:00:00
#SBATCH --partition=gpu               
#SBATCH --gres=gpu:v100-sxm2:1

python src/generate_data.py --model_family gptj --model_type 8 --gpu 0

echo "Done!"