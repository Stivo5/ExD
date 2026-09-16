#!/bin/bash

#SBATCH --job-name=check
#SBATCH -p genx
#SBATCH -N1
#SBATCH --ntasks-per-node=90
#SBATCH --time=30
#SBATCH -o %x.out


module load openmpi
module load triqs

export PYTHONPATH="/mnt/home/szhang2/miniforge3/envs/ExD/lib/python3.11/site-packages:$PYTHONPATH"



ulimit -l unlimited

# mpirun python 2_by_2_plaquette.py
mpirun python check.py
