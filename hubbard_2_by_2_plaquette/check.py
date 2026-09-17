from h5 import HDFArchive
from triqs.atom_diag import AtomDiagComplex
from triqs.operators import n, c, c_dag
from itertools import product
# from triqs.plot.mpl_interface import oplot


import sys
sys.path.append('../')


from sort import sort_states_full
from three_pt_corr_v2 import *

import triqs.utility.mpi as mpi




# Number of bosonic Matsubara frequencies for susceptibility calculation
bn_iw = 50
# Number of fermionic Matsubara frequencies for susceptibility calculation
fn_iw = 50

beta = 2.0             # Inverse temperature

biw_mesh = MeshImFreq(beta = beta, statistic = 'Boson', n_iw = bn_iw)
fiw_mesh = MeshImFreq(beta = beta, statistic = 'Fermion', n_iw = fn_iw)



# Hubbard 2 x 2 plaquette

####################
# Input parameters #
####################
eps = [-1.9, -2.1, -1.9, -2.1]  # On-site energies
t = 0.5                 # Nearest-neighbor hopping
U = 4.0                 # Coulomb repulsion
h_field = 0.05          # Magnetic field

spin_names = ("up", "dn")
atoms = (0, 1, 2, 3)
fops = list(product(spin_names, atoms))

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



######################
# Divided difference #
######################

ad = AtomDiagComplex(H, fops)
eigensys = sort_states_full(spin_names, atoms, ad)

chi3 = {}

for s1, s2 in product(spin_names, repeat=2):
    
    G_iw = dict()
    for x, y, z, w in product(range(2),repeat=4):
        ops = [(c_dag(s1, x), 'Fermion'),
               (c(s1,y), 'Fermion'),
               (c_dag(s2, z)*c(s2,w), 'Boson')]
        
        corr = ThreePtCorr(ops = ops, hamiltonian = (ad, eigensys))
        
        G_iw[f'%i%i%i%i'%(x, y, z, w)] =  corr.G_div_diff_iw(beta = beta, n_iw = [fn_iw,fn_iw])

    chi3[s1+s2] = G_iw





################
# Load Pomerol #
################

ref = HDFArchive("pomerol_ref.h5", 'r')
chi3_ref = ref['chi3_ph_AABB']





###########
# Compare #
###########
binary_strings = [f"{i:04b}" for i in range(16)]

f = [p.index for p in fiw_mesh]

for s1, s2 in product(spin_names, repeat=2):
    spin = []
    error = []
    overlap = []
    
    for s in binary_strings:
        g_ref = chi3_ref[s1,s2][int(s[0]),int(s[1]),int(s[2]),int(s[3])]
        g = chi3[s1+s2][s]
        worst = 0.0  # Greatest difference
        n = 0  # Number of points in overlap
        for iOmega in g_ref.mesh[0]:
            for inu in g_ref.mesh[1]:
                a, b = -inu.index - 1, iOmega.index + inu.index
                if a in f and b in f:
                    d = abs(g[Idx(a), Idx(b)] + g_ref[Idx(iOmega.index), Idx(inu.index)])
                    worst = max(worst, d)
                    n += 1
        spin.append(s)
        error.append(worst)
        overlap.append(n)
    
        
    if mpi.is_master_node():
        print(f"Spin configuration is ({s1},{s2})")
        for i in range(len(spin)):
            print(f'Orbital configuration {spin[i]}, max difference is {error[i]}, number of points checked {overlap[i]}')