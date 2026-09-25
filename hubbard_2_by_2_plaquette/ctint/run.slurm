#!/bin/bash

#SBATCH --job-name=G2_ctint
#SBATCH -p genx
#SBATCH -C genoa
#SBATCH -N1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH -o %x.out

# CTINT was built against cyeh's llvm19 TRIQS (unstable, python 3.11).
# Its bundled FINUFFT (used by the M4 measurement) needs AVX-512, so keep -C genoa.
module use /mnt/home/cyeh/opt/env
module load llvm19-mkl-mod2.4
module load triqs/unstable

ulimit -l unlimited
export OMP_NUM_THREADS=1

# Optional argument: number of MC cycles per MPI rank (default 100000)
mpirun python G2_ctint.py 50000
