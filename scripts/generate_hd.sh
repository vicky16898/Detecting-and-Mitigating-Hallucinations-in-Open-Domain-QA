#!/bin/bash
#SBATCH --job-name=data_gen
#SBATCH --output=./slurm_logs/data_gen_%A.out
#SBATCH --error=./slurm_logs/data_gen_%A.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=18G
#SBATCH --time=06:00:00
#SBATCH --partition=jiang               
#SBATCH --gres=gpu:a5000:1

source /home/${USER}/.bashrc
source activate odtformer

cd /projects/vig/ajay/persistent_memory/Detecting-and-Mitigating-Hallucinations-in-Open-Domain-QA
python src/generate_hd.py --model_family llama3base --model_type 8 --strategy multi_layer --gpu 0

echo "Done!"