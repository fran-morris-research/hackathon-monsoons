#!/bin/bash
#SBATCH --partition=standard
#SBATCH --account=firstrains
#SBATCH --qos=short
#SBATCH --time=1:00:00
#SBATCH --mem=100GB
#SBATCH --array=1
#SBATCH --job-name=dwtp_%a
#SBATCH -o dwtp_%A_%a.out
#SBATCH -e dwtp_%A_%a.err

module load jaspy
source /home/users/franmorr/miniforge3/bin/activate
conda activate hk26_env

cd /home/users/franmorr/hk26/hackathon-monsoons/
export PYTHONUNBUFFERED=TRUE
export PYTHONPATH=$(pwd)

python onset_identification/onset.py