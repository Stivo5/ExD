from h5 import HDFArchive
from triqs.atom_diag import AtomDiagComplex
from triqs.operators import n, c, c_dag
from triqs.gf import Idx, MeshImFreq
from itertools import product
# from triqs.plot.mpl_interface import oplot


import os
import sys
import time
sys.path.append('../')


from sort import sort_states_full
from four_pt_corr_v2 import *

import triqs.utility.mpi as mpi




# Number of bosonic Matsubara frequencies for susceptibility calculation
bn_iw = 5
# Number of fermionic Matsubara frequencies for susceptibility calculation
fn_iw = 5

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

# All (spin, orbital) components, distributed round-robin over MPI ranks
tasks = [(s1, s2, x, y, z, w)
         for s1, s2 in product(spin_names, repeat=2)
         for x, y, z, w in product(range(len(atoms)), repeat=4)]

t_start = time.time()

my_data = {}
for i, (s1, s2, x, y, z, w) in enumerate(tasks):
    if i % mpi.size != mpi.rank:
        continue

    ops = [(c_dag(s1, x), 'Fermion'),
           (c(s1,y), 'Fermion'),
           (c_dag(s2, z), 'Fermion'),
           (c(s2,w), 'Fermion')]

    corr = FourPtCorr(ops = ops, hamiltonian = (ad, eigensys))

    my_data[i] = corr.G_div_diff_iw(beta = beta, n_iw = [fn_iw,fn_iw,fn_iw], method="factorized").data

# Collect every rank's results on all ranks
all_data = {}
for d in (mpi.world.allgather(my_data) if mpi.size > 1 else [my_data]):
    all_data.update(d)

if mpi.is_master_node():
    print(f"Computed {len(tasks)} components on {mpi.size} rank(s) in {time.time() - t_start:.1f} s", flush=True)

G2_mesh = MeshProduct(*[MeshImFreq(beta = beta, statistic = 'Fermion', n_iw = fn_iw)] * 3)

G2 = {s1+s2: dict() for s1, s2 in product(spin_names, repeat=2)}
for i, (s1, s2, x, y, z, w) in enumerate(tasks):
    G2[s1+s2][f'%i%i%i%i'%(x, y, z, w)] = Gf(mesh = G2_mesh, data = all_data[i])






# ################
# # Load Pomerol #
# ################

# ref_file = "pomerol_G2_ref.h5"

# if mpi.is_master_node() and not os.path.exists(ref_file):
#     print(f"{ref_file} not found, skipping comparison with Pomerol", flush=True)

# elif mpi.is_master_node():

#     ref = HDFArchive(ref_file, 'r')
#     G2_ref = ref['G2_ph_AABB']

#     ###########
#     # Compare #
#     ###########
#     # Pomerol PH convention: G2_ref(w, nu, nu') ~ c^dag(-nu) c(nu+w) c^dag(-nu'-w) c(nu'),
#     # so G2_ref(w, nu, nu') = -g(a, b, cc) with
#     #   a  = -nu        -> index -nu.index - 1
#     #   b  = nu + w     -> index w.index + nu.index
#     #   cc = -(nu' + w) -> index -nu'.index - w.index - 1
#     orbital_strings = [f'%i%i%i%i'%idx for idx in product(range(len(atoms)), repeat=4)]

#     f = [p.index for p in fiw_mesh]

#     for s1, s2 in product(spin_names, repeat=2):
#         spin = []
#         error = []
#         overlap = []

#         for s in orbital_strings:
#             g_ref = G2_ref[s1,s2][int(s[0]),int(s[1]),int(s[2]),int(s[3])]
#             g = G2[s1+s2][s]
#             worst = 0.0  # Greatest difference
#             n_pts = 0  # Number of points in overlap
#             for iOmega in g_ref.mesh[0]:
#                 for inu in g_ref.mesh[1]:
#                     for inup in g_ref.mesh[2]:
#                         a = -inu.index - 1
#                         b = iOmega.index + inu.index
#                         cc = -inup.index - iOmega.index - 1
#                         if a in f and b in f and cc in f:
#                             d = abs(g[Idx(a), Idx(b), Idx(cc)] + g_ref[Idx(iOmega.index), Idx(inu.index), Idx(inup.index)])
#                             worst = max(worst, d)
#                             n_pts += 1
#             spin.append(s)
#             error.append(worst)
#             overlap.append(n_pts)

#         print(f"Spin configuration is ({s1},{s2})")
#         for i in range(len(spin)):
#             print(f'Orbital configuration {spin[i]}, max difference is {error[i]}, number of points checked {overlap[i]}')


if mpi.is_master_node():
    with HDFArchive('my_G2.h5', 'w') as ar:
        ar['G2'] = G2
    print('job finished')