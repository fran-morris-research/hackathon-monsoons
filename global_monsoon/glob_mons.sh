#!/bin/bash
#SBATCH --partition=standard
#SBATCH --account=firstrains
#SBATCH --qos=short
#SBATCH --time=4:00:00
#SBATCH --mem=100GB
#SBATCH --array=1
#SBATCH --job-name=global_monsoon
#SBATCH -o glob_mons_%A.out
#SBATCH -e glob_mons_%A.err

module load jaspy
source /home/users/franmorr/miniforge3/bin/activate
conda activate /home/users/franmorr/miniforge3/envs/hackathon

cd /home/users/franmorr/hk26/hackathon-monsoons/
export PYTHONUNBUFFERED=TRUE
export PYTHONPATH=$(pwd)

python global_monsoon/compute_global_monsoon.py