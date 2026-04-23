#!/bin/bash
#SBATCH --job-name=helm_hd_gen
#SBATCH --output=./slurm_logs/helm_hd_gen_%A.out
#SBATCH --error=./slurm_logs/helm_hd_gen_%A.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=18G
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1

source /home/${USER}/.bashrc
source activate odtformer

cd /projects/vig/ajay/persistent_memory/Detecting-and-Mitigating-Hallucinations-in-Open-Domain-QA
python src/generate_hd_for_helm.py --model_family gptj --model_type 7 --strategy multi_layer --gpu 0

echo "Done!"
