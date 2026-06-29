#!/bin/bash
#SBATCH --partition=standard
#SBATCH --account=firstrains
#SBATCH --qos=short
#SBATCH --time=4:00:00
### SBATCH --mem=40GB
#SBATCH --array=1
#SBATCH --job-name=dwtp_%a
#SBATCH -o dwtp_lowzoom_%A_%a.out
#SBATCH -e dwtp_lowzoom_%A_%a.err

module load jaspy
source /home/users/franmorr/miniforge3/bin/activate
conda activate hk26_env

cd /home/users/franmorr/hk26/hackathon-monsoons/
export PYTHONUNBUFFERED=TRUE
export PYTHONPATH=$(pwd)

python onset_identification/onset.py None