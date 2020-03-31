#!/bin/bash -x
#SBATCH -J parameter_sweep # A single job name for the array
#SBATCH -n 1                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -p shared
#SBATCH -t 04:00:00         # Runtime in D-HH:MM:SS, minimum of 10 minutes
#SBATCH --mem=48000          # Memory pool for all cores (see also --mem-per-cpu) MBs
#SBATCH -o joblogs/parameter_sweep_%A_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e joblogs/parameter_sweep_%A_%a.err  # File to which STDERR will be written, %j inserts jobid

set -x

date
echo index ${SLURM_ARRAY_TASK_ID}
python3 parameter_sweep.py $1 $2 --index ${SLURM_ARRAY_TASK_ID} --sim_name $3 --seed_offset $4
