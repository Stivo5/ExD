from h5 import HDFArchive
from triqs.atom_diag import AtomDiagComplex
from triqs.operators import n, c, c_dag
from triqs.gf import Idx, MeshImFreq
from itertools import product
# from triqs.plot.mpl_interface import oplot

import time

import sys
sys.path.append('../')


from sort import sort_states_full
from four_pt_corr_v2 import *

import triqs.utility.mpi as mpi




# Number of bosonic Matsubara frequencies for susceptibility calculation
bn_iw = 10
# Number of fermionic Matsubara frequencies for susceptibility calculation
fn_iw = 10

beta = 2.0             # Inverse temperature

biw_mesh = MeshImFreq(beta = beta, statistic = 'Boson', n_iw = bn_iw)
fiw_mesh = MeshImFreq(beta = beta, statistic = 'Fermion', n_iw = fn_iw)



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
fops = list(product(spin_names, atoms))


H = sum(e * n(s, a) for (a, e), s in product(zip(atoms, eps), spin_names))
H += sum(-h_field * (n('up', a) - n('dn', a)) for a in atoms)
H += U * sum(n('up', a) * n('dn', a) for a in atoms)
H += t * sum((c_dag(sp, 0) * c(sp, 1) + c_dag(sp, 1) * c(sp, 0)) for sp in spin_names)




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





import os

##############
# Load CTINT #
##############
# ctint/G2_ctint.py stores S.G2ph_iw, which has the same definition and meshes as
# pomerol's G2_ph_AABB, so both are compared with exactly the same index mapping below.
G2_ctint = HDFArchive("ctint_G2_ref.h5", 'r')['G2_ph_AABB']

#############################
# Load Pomerol (if present) #
#############################
G2_pomerol = HDFArchive("pomerol_G2_ref.h5", 'r')['G2_ph_AABB'] if os.path.exists("pomerol_G2_ref.h5") else None





###########
# Compare #
###########
# Pomerol PH convention: G2_ref(w, nu, nu') ~ c^dag(-nu) c(nu+w) c^dag(-nu'-w) c(nu'),
# so G2_ref(w, nu, nu') = -g(a, b, cc) with
#   a  = -nu        -> index -nu.index - 1
#   b  = nu + w     -> index w.index + nu.index
#   cc = -(nu' + w) -> index -nu'.index - w.index - 1
binary_strings = [f"{i:04b}" for i in range(16)]

f = [p.index for p in fiw_mesh]

def compare(G2_ref, label):
    if mpi.is_master_node():
        print(f"\n===== Divided difference vs {label} =====")
    for s1, s2 in product(spin_names, repeat=2):
        spin = []
        error = []
        overlap = []
        largest = []

        for s in binary_strings:
            g_ref = G2_ref[s1,s2][int(s[0]),int(s[1]),int(s[2]),int(s[3])]
            g = G2[s1+s2][s]
            worst = 0.0  # Greatest difference
            big = 0.0    # Greatest |G2_ref| on the overlap, sets the scale of the error
            n = 0  # Number of points in overlap
            for iOmega in g_ref.mesh[0]:
                for inu in g_ref.mesh[1]:
                    for inup in g_ref.mesh[2]:
                        a = -inu.index - 1
                        b = iOmega.index + inu.index
                        cc = -inup.index - iOmega.index - 1
                        if a in f and b in f and cc in f:
                            r = g_ref[Idx(iOmega.index), Idx(inu.index), Idx(inup.index)]
                            d = abs(g[Idx(a), Idx(b), Idx(cc)] + r)
                            worst = max(worst, d)
                            big = max(big, abs(r))
                            n += 1
            spin.append(s)
            error.append(worst)
            overlap.append(n)
            largest.append(big)


        if mpi.is_master_node():
            print(f"Spin configuration is ({s1},{s2})")
            for i in range(len(spin)):
                print(f'Orbital configuration {spin[i]}, max difference is {error[i]}, max |G2_ref| is {largest[i]}, number of points checked {overlap[i]}')

# CTINT is stochastic: expect differences at the Monte Carlo noise level, not 1e-16
compare(G2_ctint, "CTINT")
if G2_pomerol is not None:
    compare(G2_pomerol, "Pomerol")