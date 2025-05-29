#!/bin/bash
#SBATCH --job-name=vol_cal
#SBATCH --output=/scratch/midway3/zhengzhiyu6689/macs30123/project/log/har.out
#SBATCH --error=/scratch/midway3/zhengzhiyu6689/macs30123/project/log/har.err
#SBATCH --account=macs30123
#SBATCH --partition=amd
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/zhengzhiyu6689/nlp5  
module load java/21.0

export CORES=${SLURM_CPUS_PER_TASK}   
export DRIVER_MEM_GB=60              

SCRIPT=/scratch/midway3/zhengzhiyu6689/macs30123/project/prediction/benchmark.py

echo "Running spark-submit on $SCRIPT  |  cores=$CORES  mem=${DRIVER_MEM_GB}g"

spark-submit \
    --master "local[$CORES]" \
    --driver-memory "${DRIVER_MEM_GB}g" \
    --conf spark.sql.parquet.compression.codec=snappy \
    "$SCRIPT"