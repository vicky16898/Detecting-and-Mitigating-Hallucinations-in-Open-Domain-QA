#!/bin/bash
#SBATCH --job-name=helm_data_gen
#SBATCH --output=./slurm_logs/helm_data_gen_%A.out
#SBATCH --error=./slurm_logs/helm_data_gen_%A.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=18G
#SBATCH --time=08:00:00
#SBATCH --partition=gpu               
#SBATCH --gres=gpu:v100-sxm2:1

source /home/${USER}/.bashrc
source activate odtformer

cd /projects/vig/ajay/persistent_memory/Detecting-and-Mitigating-Hallucinations-in-Open-Domain-QA
python generate_helm_data.py --model_family llama3base --model_type 8 --gpu 0

echo "Done!"
