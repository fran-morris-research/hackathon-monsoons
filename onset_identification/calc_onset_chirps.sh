#!/bin/bash
#SBATCH --partition=standard
#SBATCH --account=firstrains
#SBATCH --qos=short
#SBATCH --time=4:00:00
#SBATCH --mem=100GB
#SBATCH --job-name=dwtp_%a
#SBATCH -o dwtp_chirps_%A_%a.out
#SBATCH -e dwtp_chirps_%A_%a.err

module load jaspy
source /home/users/franmorr/miniforge3/bin/activate
conda activate /home/users/franmorr/miniforge3/envs/hackathon
cd /home/users/franmorr/hk26/hackathon-monsoons/

export PYTHONUNBUFFERED=TRUE
export PYTHONPATH=$(pwd)

python onset_identification/onset-chirps.py 