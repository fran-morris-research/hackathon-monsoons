#!/bin/bash
#SBATCH --job-name=eddycalc
#SBATCH --output=eddycalc_.out
#SBATCH --error=eddycalc_.err
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=1000G
#SBATCH --partition=standard
#SBATCH --account=firstrains
#SBATCH --qos=dask

module load jaspy

python uvw_eachtimestep.py