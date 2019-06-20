#!/bin/sh
#SBATCH --verbose
#SBATCH -p aquila
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xc1305@nyu.edu

#SBATCH --array=0-19
##SBATCH --output=sbl_%A_%a.out #if you need each subjob to generate an output file.
#SBATCH --output=sbl_%A_%a.out # %A is SLURM_ARRAY_JOB_ID, %a is SLURM_ARRAY_TASK_ID, which is 0-9
#SBATCH --error=sbl_%A_%a.err

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

module load anaconda3 cuda/9.0 glfw/3.3 gcc/7.3
source activate rl

echo ${SLURM_ARRAY_TASK_ID}
python ddqn_job_array.py --setting ${SLURM_ARRAY_TASK_ID}
