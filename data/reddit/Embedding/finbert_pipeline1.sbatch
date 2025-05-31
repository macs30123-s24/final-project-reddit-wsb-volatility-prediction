#!/bin/bash
#SBATCH --job-name=finbert
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4              
#SBATCH --cpus-per-task=32        
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --account=macs30123
#SBATCH --output=pipeline1.out
#SBATCH --error=pipeline1.err
#SBATCH --chdir=/scratch/midway3/zhengzhiyu6689/macs30123/project/reddit/Embedding

module load cuda/11.8 python/anaconda-2022.05 spark/3.3.2
export PYSPARK_PYTHON=$(which python)
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/scratch/midway3/zhengzhiyu6689/macs30123/project/reddit/hf

# if skip tokenizer： sbatch --export=SKIP_TOKENIZE=1 finbert_pipeline.sbatch
ARGS=""
[[ "$SKIP_TOKENIZE" == "1" ]] && ARGS="--skip-tokenize"

python finbert_pipeline1.py $ARGS
