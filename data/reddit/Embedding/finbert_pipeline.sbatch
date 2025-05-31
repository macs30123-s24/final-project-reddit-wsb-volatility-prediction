#!/bin/bash
#SBATCH --job-name=finbert
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --account=macs30123
#SBATCH --output=pipeline_%j.out
#SBATCH --error=pipeline_%j.err
#SBATCH --chdir=/scratch/midway3/zhengzhiyu6689/macs30123/project/reddit/Embedding

module load cuda/11.8 python/anaconda-2022.05 spark/3.3.2
export PYSPARK_PYTHON=$(which python)
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/scratch/midway3/zhengzhiyu6689/macs30123/project/reddit/hf

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L                           # check GPU type

# Directory where the script is located
ARGS=""
[[ "$SKIP_TOKENIZE" == "1" ]] && ARGS="--skip-tokenize"

python finbert_pipeline.py $ARGS
