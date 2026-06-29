#!/bin/bash
#SBATCH --partition=standard
#SBATCH --account=firstrains
#SBATCH --qos=high
#SBATCH --time=12:00:00
#SBATCH --mem=1000GB
#SBATCH --array=6,7,8
#SBATCH --job-name=dwtp_%a
#SBATCH -o dwtp_highzoom_%A_%a.out
#SBATCH -e dwtp_highzoom_%A_%a.err

module load jaspy
source /home/users/franmorr/miniforge3/bin/activate
conda activate hk26_env

cd /home/users/franmorr/hk26/hackathon-monsoons/
export PYTHONUNBUFFERED=TRUE
export PYTHONPATH=$(pwd)

python onset_identification/onset.py ${SLURM_ARRAY_TASK_ID}