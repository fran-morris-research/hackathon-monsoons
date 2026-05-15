#!/bin/bash 
#SBATCH --partition=standard
#SBATCH --account=firstrains
#SBATCH --qos=high
#SBATCH --ntasks=10
#SBATCH --mem=80000
#SBATCH -t 3:59:00
#SBATCH --array=1,2,12
#SBATCH -o meanstate_%A_%a.out
#SBATCH -e meanstate_%A_%a.err
#SBATCH --job-name=meanstate

# executable 
echo mean state, sim=$1, var=$2, month=${SLURM_ARRAY_TASK_ID}

module load jaspy

export PYTHONUNBUFFERED=TRUE
export PYTHONPATH=$(pwd)

#python get_mean_state.py -s $1 -v $2 -f $3 -m ${SLURM_ARRAY_TASK_ID} -c True
python get_mean_state.py -s n2560_RAL3p3_tuned -v precip -f b -o /work/scratch-nopw2/franmorr/hk/ -m ${SLURM_ARRAY_TASK_ID} -c True
