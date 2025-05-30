#!/bin/bash
#SBATCH --job-name=redcrsp
#SBATCH --output=/scratch/midway3/zhengzhiyu6689/macs30123/project/log/reddit_crsp.out
#SBATCH --error=/scratch/midway3/zhengzhiyu6689/macs30123/project/log/reddit_crsp.err
#SBATCH --account=macs30123
#SBATCH --partition=caslake
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=140G

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /project/dachxiu/zhengzhiyu/nlp5 

module load java/21.0

export CORES=${SLURM_CPUS_PER_TASK} 
export DRIVER_MEM_GB=130     

if [ "$1" == "test" ]; then
  echo ">> Running TEST mode (5 rows only)"
  spark-submit \
      --master local[$SLURM_CPUS_PER_TASK] \
      --driver-memory "${DRIVER_MEM_GB}g" \
      /scratch/midway3/zhengzhiyu6689/macs30123/project/prediction/data/reddit_crsp_linkage.py --test
else
  echo ">> Running FULL mode"
  spark-submit \
      --master local[$SLURM_CPUS_PER_TASK] \
      --driver-memory "${DRIVER_MEM_GB}g" \
      /scratch/midway3/zhengzhiyu6689/macs30123/project/prediction/data/reddit_crsp_linkage.py
fi
