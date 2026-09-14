from h5 import HDFArchive
from triqs.atom_diag import AtomDiagComplex
from triqs.operators import n, c, c_dag
from itertools import product
# from triqs.plot.mpl_interface import oplot

from sort import sort_states_full
from three_pt_corr import *
from four_pt_corr_v2 import FourPtCorr

import triqs.utility.mpi as mpi



# Number of bosonic Matsubara frequencies for susceptibility calculation
bn_iw = 10
# Number of fermionic Matsubara frequencies for susceptibility calculation
fn_iw = 10

beta = 2.0              # Inverse temperature

biw_mesh = MeshImFreq(beta = beta, statistic = 'Boson', n_iw = bn_iw)
fiw_mesh = MeshImFreq(beta = beta, statistic = 'Fermion', n_iw = fn_iw)




# 2x2 Hubbard plaquette

####################
# Input parameters #
####################
eps = [-1.9, -2.1, -1.9, -2.1]  # On-site energies
t = 0.5                 # Nearest-neighbor hopping
U = 4.0                 # Coulomb repulsion
h_field = 0.05          # Magnetic field

spin_names = ("up", "dn")
atoms = (0, 1, 2, 3)


####################
# Hubbard Hamiltonian
####################

# On-site energies
H = sum(
    e * n(s, a)
    for (a, e), s in product(zip(atoms, eps), spin_names)
)

# Magnetic field
H += sum(
    -h_field * (n('up', a) - n('dn', a))
    for a in atoms
)

# On-site Hubbard interaction
H += U * sum(
    n('up', a) * n('dn', a)
    for a in atoms
)

# Nearest-neighbor hopping
#
#       0 ---- 1
#       |      |
#       |      |
#       2 ---- 3
#
H += t * sum(
    c_dag(sp, i) * c(sp, j)
    + c_dag(sp, j) * c(sp, i)
    for sp in spin_names
    for i, j in [
        (0, 1),  # top
        (1, 3),  # right
        (3, 2),  # bottom
        (2, 0),  # left
    ]
)

fops = list(product(spin_names, atoms))

# Diagonalize Hamiltonian
ad = AtomDiagComplex(H, fops)

eigensys = sort_states_full(spin_names, atoms, ad)

# Compute correlator
ops = [(c_dag('up', 0), 'Fermion'),
       (c('up',0), 'Fermion'),
       (c('up', 0), 'Fermion'),
       (c_dag('up', 0), 'Fermion')]

corr = FourPtCorr(ops = ops, hamiltonian = (ad, eigensys))

G = corr.G_div_diff_iw(beta = beta, n_iw = [10,10,10])


print("Ends computation.")

if mpi.is_master_node():
    with HDFArchive("plaq_4pt.h5",'w') as A:
        A["G"] = G

