#!/bin/bash
#SBATCH --partition=standard
#SBATCH --account=firstrains
#SBATCH --qos=high
#SBATCH --time=12:00:00
#SBATCH --mem=400GB
#SBATCH --array=1
#SBATCH --job-name=precip
#SBATCH -o monthly_%A.out
#SBATCH -e monthly_%A.err

module load jaspy
source /home/users/franmorr/miniforge3/bin/activate
conda activate /home/users/franmorr/miniforge3/envs/hackathon
cd /home/users/franmorr/hk26/hackathon-monsoons/
export PYTHONUNBUFFERED=TRUE
export PYTHONPATH=$(pwd)
python precip/monthlies.py