import numpy as np

from triqs.atom_diag import AtomDiagComplex
from triqs.operators import n, c, c_dag
from itertools import product
from triqs.plot.mpl_interface import oplot

import sys
sys.path.append("../")
from sort import sort_states_full
from three_pt_corr import ThreePtCorr

import triqs.utility.mpi as mpi

from h5 import HDFArchive
import time



# Hubbard dimer

####################
# Input parameters #
####################

beta = 2.0              # Inverse temperature
eps = [-1.9, -2.1]      # Energy levels of the atoms
t = 0.5                 # Hopping matrix element
U = 4.0                 # Coulomb repulsion
h_field = 0.05          # Magnetic field

spin_names = ("up", "dn")
atoms = (0, 1)

# Number of bosonic Matsubara frequencies for susceptibility calculation
n_iw = 10
# Number of fermionic Matsubara frequencies for susceptibility calculation
n_inu = 10

# Block structure of \chi^3
gf_struct = [['up', 2], ['dn', 2]]

H = sum(e * n(s, a) for (a, e), s in product(zip(atoms, eps), spin_names))
H += sum(-h_field * (n('up', a) - n('dn', a)) for a in atoms)
H += U * sum(n('up', a) * n('dn', a) for a in atoms)
H += t * sum((c_dag(sp, 0) * c(sp, 1) + c_dag(sp, 1) * c(sp, 0)) for sp in spin_names)

fops = list(product(spin_names, atoms))


# Diagonalize Hamiltonian
ad = AtomDiagComplex(H, fops)
# Sort eigenstates
eigensys = sort_states_full(spin_names, atoms, ad)


# Compute correlators
ops = [(c_dag('up', 1), 'Fermion'),
       (c('up',0), 'Fermion'),
       (c('up', 0)*c_dag('up',1), 'Boson')]

corr = ThreePtCorr(ops = ops, hamiltonian = (ad, eigensys))

mesh_pt = np.arange(500,1000,50)
t = []
G = []
for n in mesh_pt:
    start_time = time.perf_counter()
    G_iw = corr.G_div_diff_iw(beta = beta, n_iw = [n,n])
    elapsed_time = time.perf_counter() - start_time
    print(f"Finished n={n} in time {elapsed_time:.6f}")

    t.append(elapsed_time)
    G.append(G_iw)


if mpi.is_master_node():
    with HDFArchive("n_vs_time.h5", 'w') as A:
        A["mesh_pt"] = mesh_pt
        A["time"] = t
        # A["G"] = G

