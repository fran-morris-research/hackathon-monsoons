#!/bin/bash
#SBATCH --job-name=imerg
#SBATCH --output=imerg_.out
#SBATCH --error=imerg_.err
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=1000G
#SBATCH --partition=standard
#SBATCH --account=firstrains
#SBATCH --qos=dask

module load jaspy
source ~/.bashrc
conda activate /home/users/rwjones/.conda/envs/hk26_env

python rainfall_bias_plots.py