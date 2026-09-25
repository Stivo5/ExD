from h5 import HDFArchive
from triqs.operators import c, c_dag, n
from triqs.utility import mpi
from triqs.utility.comparison_tests import *
from pomerol2triqs import PomerolED
from itertools import product



# 2x2 Hubbard plaquette

####################
# Input parameters #
####################
beta = 2.0              # Inverse temperature

eps = [-1.9, -2.1, -1.9, -2.1]  # On-site energies
t = 0.5                 # Nearest-neighbor hopping
U = 4.0                 # Coulomb repulsion
h_field = 0.05          # Magnetic field

spin_names = ("up", "dn")
atoms = (0, 1, 2, 3)


# Number of bosonic Matsubara frequencies for susceptibility calculation
n_iw = 5
# Number of fermionic Matsubara frequencies for susceptibility calculation
n_inu = 5

# Block structure of \chi^3
gf_struct = [['up', 4], ['dn', 4]]

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




# Conversion from TRIQS to Pomerol notation for operator indices
index_converter = {}
index_converter.update({(sn, 0) : ("A", 0, "down" if sn == "dn" else "up") for sn in spin_names})
index_converter.update({(sn, 1) : ("B", 0, "down" if sn == "dn" else "up") for sn in spin_names})
index_converter.update({(sn, 2) : ("C", 0, "down" if sn == "dn" else "up") for sn in spin_names})
index_converter.update({(sn, 3) : ("D", 0, "down" if sn == "dn" else "up") for sn in spin_names})

# Make PomerolED solver object
ed = PomerolED(index_converter, verbose = True)





# Diagonalize H
ed.diagonalize(H)

#####################################
# Compute G^2(i\omega, i\nu, i\nu') #
#####################################


# chi2 = dict()
# for x, y, z, w in product(range(2),repeat=4):
#     chi2[f'%i%i%i%i'%(x,y,z,w)] = ed.chi_iw(("up", x), ("up", y), ("up", z), ("up", w),beta,n_iw, connected=False)

params = {'gf_struct': gf_struct, 'beta': beta, 'n_iw': n_iw, 'n_inu': n_inu}
G2_ph_AABB = ed.G2_iw_inu_inup(**params, channel='PH', block_order='AABB')


if mpi.is_master_node():
    with HDFArchive('../pomerol_G2_ref.h5', 'w') as ar:
        # for x, y, z, w in product(range(2),repeat=4):
        #     ar[f'%i%i%i%i'%(x,y,z,w)] = chi2[f'%i%i%i%i'%(x,y,z,w)]
        ar["G2_ph_AABB"] = G2_ph_AABB
        ar['energies'] = ed.energies
