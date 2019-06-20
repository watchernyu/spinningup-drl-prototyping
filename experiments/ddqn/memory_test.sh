#!/bin/sh
#SBATCH --verbose
#SBATCH -p aquila
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --mem=128GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cw1681@nyu.edu

#SBATCH --output=sbl.out
#SBATCH --error=sbl.err

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

module load anaconda3 cuda/9.0 glfw/3.3 gcc/7.3
source activate rl

python memory_test.py
