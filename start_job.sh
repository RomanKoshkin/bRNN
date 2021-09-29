#!/bin/bash

#SBATCH --job-name=JOBNAME1
#SBATCH --mail-user=some.email@somehost.jp
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2g
#SBATCH --output=./slurm_big/%j.out


## submit 4 jobs as an array, give them individual id from 1 to 4
#SBATCH --array=1-1
#SBATCH --time=2-6:45:0

## $1 means that we pass the first argument (passed into this bash scipt) into the python script
echo ${SLURM_ARRAY_JOB_ID}
echo ${SLURM_ARRAY_TASK_ID}
python grid_cell.py $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID $1
