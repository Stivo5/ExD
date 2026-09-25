#!/bin/bash

##SBATCH --job-name=check_chi3
#SBATCH --job-name=check_G2
#SBATCH -p genx
#SBATCH -C genoa
#SBATCH -N1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH -o %x.out


module load openmpi
module load triqs

export PYTHONPATH="/mnt/home/szhang2/miniforge3/envs/ExD/lib/python3.11/site-packages:$PYTHONPATH"



ulimit -l unlimited
# export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1

# mpirun python 2_by_2_plaquette.py
# mpirun python check_chi3.py
mpirun python check_G2.py
