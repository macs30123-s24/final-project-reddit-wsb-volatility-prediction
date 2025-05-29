#!/bin/bash
#SBATCH --job-name=vol_cal
#SBATCH --output=/scratch/midway3/zhengzhiyu6689/macs30123/project/log/vol_cal.out
#SBATCH --error=/scratch/midway3/zhengzhiyu6689/macs30123/project/log/vol_cal.err
#SBATCH --account=macs30123
#SBATCH --partition=amd
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32    
#SBATCH --mem=64G

# 0) Software environment
conda activate /home/zhengzhiyu6689/nlp5  
module load java/21.0

# 1) Derive Spark resource settings from Slurm variables

export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-16}
export DRIVER_MEM_GB=60

# 2) Submit the PySpark driver
SCRIPT_PATH=/scratch/midway3/zhengzhiyu6689/macs30123/project/crsp_raw/crsp_vol_calculation.py

spark-submit \
    --master "local[${SLURM_CPUS_PER_TASK}]" \   
    --driver-memory "${DRIVER_MEM_GB}g" \       
    --conf spark.sql.parquet.compression.codec=snappy \  
    "$SCRIPT_PATH"





