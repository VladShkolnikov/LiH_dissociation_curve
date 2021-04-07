#!/usr/bin/env bash

#SBATCH -J LiH-dissoc-curve
#SBATCH -p normal_q
#SBATCH -A jvandyke_alloc
#SBATCH -n 1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32gb
#SBATCH --array=0-20
#SBATCH -t 12:00:00

source activate chem

python LiH.py 1.0 3.0 $SLURM_ARRAY_TASK_COUNT $SLURM_ARRAY_TASK_ID

wait
exit 0
