#!/bin/bash

#SBATCH --job-name=compare
#SBATCH --constraint=rome
#SBATCH -N1
#SBATCH --ntasks-per-node=128
#SBATCH -o %x.out
#SBATCH -e %x.err


module load triqs-tprf

mpirun python 2_by_2_plaquette.py
