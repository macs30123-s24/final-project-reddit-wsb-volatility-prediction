#!/bin/bash
#SBATCH --job-name=prediction
#SBATCH --output=/scratch/midway3/zhengzhiyu6689/macs30123/project/log/prediction_full_amd.out
#SBATCH --error=/scratch/midway3/zhengzhiyu6689/macs30123/project/log/prediction_full_amd.err
#SBATCH --account=macs30123
#SBATCH --partition=amd
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G


source $(conda info --base)/etc/profile.d/conda.sh
conda activate /project/dachxiu/zhengzhiyu/nlp5

module load java/21.0


SCRIPT="/scratch/midway3/zhengzhiyu6689/macs30123/project/prediction/reddit/embadding_model.py"


export CORES=${SLURM_CPUS_PER_TASK} 
export DRIVER_MEM_GB=120            

echo "Running spark-submit on $SCRIPT  |  cores=$CORES  mem=${DRIVER_MEM_GB}g"

spark-submit \
    --master "local[$CORES]" \
    --driver-memory "${DRIVER_MEM_GB}g" \
    --conf spark.sql.parquet.compression.codec=snappy \
    "$SCRIPT" 