#!/bin/bash

##SBATCH --job-name=chi3_pomerol
#SBATCH --job-name=G2_pomerol
#SBATCH -p genx
#SBATCH -C genoa
#SBATCH -N1
#SBATCH --ntasks-per-node=90
#SBATCH -o %x.out

module use /mnt/home/wentzell/opt/modules
module load devenv9/gcc-py3-mkl
module load triqs/4.0.x_gcc

ulimit -l unlimited

# mpirun python chi3_pomerol.py
mpirun python G2_pomerol.py
