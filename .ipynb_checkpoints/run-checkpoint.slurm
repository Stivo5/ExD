#!/bin/bash

#SBATCH --job-name=dimer
#SBATCH -p ccq
#SBATCH -N5
#SBATCH --ntasks-per-node=90
#SBATCH -o %x.out


# source ~/.gw_edmft.sh

module load openmpi
module load triqs

export PYTHONPATH="/mnt/home/szhang2/miniforge3/envs/ExD/lib/python3.11/site-packages:$PYTHONPATH"


# export PYTHONPATH="/mnt/sw/nix/store/7x5jhsqlyg8cgr410f39ic85q37sx177-py-mpi4py-4.0.1-view/lib/python3.11/site-packages:/mnt/sw/nix/store/bkvk8541rjwil1z6w4whb4c4r3fpd7wf-triqs-3.3.1/lib/python3.11/site-packages:/mnt/sw/nix/store/bgrh0ch4v8v2vg6q35qw838ai3fnnm5l-triqs-tprf-3.3.1/lib/python3.11/site-packages:/mnt/home/szhang2/miniforge3/envs/ExD/lib/python3.11/site-packages:/mnt/sw/nix/store/7kfpfhj104s5hynnhr5jzsrm25p4vwla-python-3.11.11-view/lib/python3.11/site-packages"

ulimit -l unlimited

# mpirun python 2_by_2_plaquette.py
mpirun python dimer.py
