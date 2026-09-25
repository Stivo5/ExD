"""
CTINT equivalent of pomerol2triqs' ed.G2_iw_inu_inup(channel='PH', block_order='AABB')
for the Hubbard plaquette (see ../pomerol_ref/G2_pomerol.py).

Convention check (pomerol2triqs c++/g2.cpp vs. triqs_ctint c++/triqs_ctint/post_process.cpp):

  pomerol PH/AABB : G2[A,B](w, nu, nu')_{abcd}
                    = + < T c^+_{Aa}(nu) c_{Ab}(nu+w) c^+_{Bc}(nu'+w) c_{Bd}(nu') >
  CTINT G2ph_iw   : G2ph[A,B](w, nu, nu')_{ijkl}, whose disconnected part is
                    beta d_{w,0} G_A(nu)_{ji} G_B(nu')_{lk}
                  - beta d_{AB} d_{nu,nu'} G_A(nu)_{li} G_B(nu+w)_{jk}
                    i.e. the same operator string and Fourier convention.

So S.G2ph_iw can be compared element by element with pomerol's G2_ph_AABB,
provided the meshes are the same. TRIQS bosonic meshes built with n_iw run over
-(n_iw-1)..(n_iw-1), and fermionic ones over -n_inu..n_inu-1. We pick
n_iW_M4 = n_iw and n_iw_M4 = n_inu, which reproduces pomerol's meshes exactly.

In CTINT only the interaction U n_up n_dn goes into h_int. All quadratic terms
(eps, h_field, t) go into the Weiss field G0_iw.
"""
import sys
import numpy as np
from itertools import product

# Import CTINT solver
sys.path.append("/mnt/home/szhang2/Softwares/ctint/ctint.build/install/lib/python3.11/site-packages/")
from triqs_ctint import Solver

from h5 import HDFArchive
from triqs.operators import n
from triqs.gf import iOmega_n, inverse
import triqs.utility.mpi as mpi


# Hubbard plaquette

####################
# Input parameters #
####################

beta = 2.0              # Inverse temperature
eps = [-1.9, -2.1, -1.9, -2.1]  # On-site energies
t = 0.5                 # Nearest-neighbor hopping
U = 2.0                 # Coulomb repulsion
h_field = 0.05          # Magnetic field

spin_names = ("up", "dn")
atoms = (0, 1, 2, 3)

# Number of bosonic Matsubara frequencies for susceptibility calculation
bn_iw = 10
# Number of fermionic Matsubara frequencies for susceptibility calculation
fn_iw = 10

# Block structure of G and G^2
gf_struct = [('up', len(atoms)), ('dn', len(atoms))]


# Interacting part of H. The full H is
#   sum eps_a n_{s,a} - h sum_a (n_{up,a} - n_{dn,a}) + t sum_s sum_<ij> (c^+_{si} c_{sj} + h.c.) + U sum_a n_{up,a} n_{dn,a}
h_int = U * sum(n('up', a) * n('dn', a) for a in atoms)

# Nearest-neighbor bonds
#
#       0 ---- 1
#       |      |
#       |      |
#       2 ---- 3
#
bonds = [
    (0, 1),  # top
    (1, 3),  # right
    (3, 2),  # bottom
    (2, 0),  # left
]

# Quadratic part of H, one matrix per spin block
sigma = {'up': +1.0, 'dn': -1.0}
h0 = {}
for s in spin_names:
    h0[s] = np.diag([e - sigma[s] * h_field for e in eps])
    for i, j in bonds:
        h0[s][i, j] = h0[s][j, i] = t


################
# CTINT solver #
################

solver_const = {
    "beta" : beta,
    "gf_struct" : gf_struct,
    "n_iw" : 500,        # Frequencies for G0_iw / G_iw (must cover nu + w of the G2 mesh)
    "n_tau" : 5001,
}

S = Solver(**solver_const)

# Weiss field G0 = (iw - h0)^{-1}
for s, g0 in S.G0_iw:
    g0 << inverse(iOmega_n - h0[s])

solve_params = {
    "h_int" : h_int,
    "n_cycles" : int(sys.argv[1]) if len(sys.argv) > 1 else 100000,  # MC cycles per MPI rank
    "length_cycle" : 15,
    "n_warmup_cycles" : 3000000,
    "random_seed" : 34788 + 928374 * mpi.rank,
    "delta" : 0.1,               # alpha is built automatically from the Hartree-Fock densities
    "measure_M_tau" : True,      # Needed for M_iw / G_iw in the post-processing
    "measure_M4ph_iw" : True,    # Two-particle Green function in the PH channel
    "n_iW_M4" : bn_iw,            # Bosonic mesh:   -(n_iw-1) .. n_iw-1  (same as pomerol)
    "n_iw_M4" : fn_iw,           # Fermionic mesh: -n_inu .. n_inu-1    (same as pomerol)
    "post_process" : True,
}

S.solve(**solve_params)

# S.G2ph_iw[A,B](w, nu, nu')(a,b,c,d) is directly comparable with pomerol's G2_ph_AABB
G2_ph_AABB = S.G2ph_iw

if mpi.is_master_node():
    with HDFArchive('../ctint_G2_ref.h5', 'w') as ar:
        ar["G2_ph_AABB"] = G2_ph_AABB
        # ar["G2_ph_conn_AABB"] = S.G2ph_conn_iw
        # ar["G_iw"] = S.G_iw
        # ar["G0_iw"] = S.G0_iw
        # ar["average_sign"] = S.average_sign
        # ar["average_k"] = S.average_k
        # ar["solve_params"] = {k: v for k, v in solve_params.items() if k != "h_int"}
