#!/bin/bash

#SBATCH --job-name=validate_factorized
#SBATCH -p genx
#SBATCH -C genoa
#SBATCH -N1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=4:00:00
#SBATCH -o %x.out


module load openmpi
module load triqs

export PYTHONPATH="/mnt/home/szhang2/miniforge3/envs/ExD/lib/python3.11/site-packages:$PYTHONPATH"

export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=1
ulimit -l unlimited

# Compare method="factorized" against method="direct" (the slow reference).
python validate_factorized.py dimer 5
python validate_factorized.py plaquette 2
