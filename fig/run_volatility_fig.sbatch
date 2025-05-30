#!/bin/bash
#SBATCH --job-name=vol_fig
#SBATCH --output=/scratch/midway3/zhengzhiyu6689/macs30123/project/log/vol_fig_amd.out
#SBATCH --error=/scratch/midway3/zhengzhiyu6689/macs30123/project/log/vol_fig_amd.err
#SBATCH --account=macs30123
#SBATCH --partition=amd
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12   
#SBATCH --mem=128G

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /project/dachxiu/zhengzhiyu/nlp5

module load java/21.0

SCRIPT=/scratch/midway3/zhengzhiyu6689/macs30123/project/crsp_raw/volatility_fig.py
FIG_DIR=/scratch/midway3/zhengzhiyu6689/macs30123/project/crsp_raw/fig
mkdir -p "$FIG_DIR"              

export CORES=${SLURM_CPUS_PER_TASK}  
export DRIVER_MEM_GB=120           

echo "Running spark-submit on $SCRIPT  |  cores=$CORES  mem=${DRIVER_MEM_GB}g"

spark-submit \
    --master "local[$CORES]" \
    --driver-memory "${DRIVER_MEM_GB}g" \
    --conf spark.sql.parquet.compression.codec=snappy \
    "$SCRIPT" --sample 0