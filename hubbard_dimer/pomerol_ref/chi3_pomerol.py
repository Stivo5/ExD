from h5 import HDFArchive
from triqs.operators import c, c_dag, n
from triqs.utility import mpi
from triqs.utility.comparison_tests import *
from pomerol2triqs import PomerolED
from pomerol2triqs import pom_chi3
from itertools import product


# # Hubbard dimer

# ####################
# # Input parameters #
# ####################

# beta = 2.0              # Inverse temperature
# eps = [-1.9, -2.1]      # Energy levels of the atoms
# t = 0.5                 # Hopping matrix element
# U = 4.0                 # Coulomb repulsion
# h_field = 0.05          # Magnetic field

# spin_names = ("up", "dn")
# atoms = (0, 1)

# # Number of bosonic Matsubara frequencies for susceptibility calculation
# n_iw = 10
# # Number of fermionic Matsubara frequencies for susceptibility calculation
# n_inu = 10

# # Block structure of \chi^3
# gf_struct = [['up', 2], ['dn', 2]]

# # Conversion from TRIQS to Pomerol notation for operator indices
# index_converter = {}
# index_converter.update({(sn, 0) : ("A", 0, "down" if sn == "dn" else "up") for sn in spin_names})
# index_converter.update({(sn, 1) : ("B", 0, "down" if sn == "dn" else "up") for sn in spin_names})

# # Make PomerolED solver object
# ed = PomerolED(index_converter, verbose = True)

# # Hamiltonian
# H = sum(e * n(s, a) for (a, e), s in product(zip(atoms, eps), spin_names))
# H += sum(-h_field * (n('up', a) - n('dn', a)) for a in atoms)
# H += U * sum(n('up', a) * n('dn', a) for a in atoms)
# H += t * sum((c_dag(sp, 0) * c(sp, 1) + c_dag(sp, 1) * c(sp, 0)) for sp in spin_names)

# # Diagonalize H
# ed.diagonalize(H)

# ##################
# # Compute \chi^3 #
# ##################


# chi2 = dict()
# for x, y, z, w in product(range(2),repeat=4):
#     chi2[f'%i%i%i%i'%(x,y,z,w)] = ed.chi_iw(("up", x), ("up", y), ("up", z), ("up", w),beta,n_iw, connected=False)

# params = {'gf_struct': gf_struct, 'beta': beta, 'n_iw': 10, 'n_inu': 10}
# chi3_ph_AABB = ed.chi3_iw_inu(**params, channel='PH', block_order='AABB')


# if mpi.is_master_node():
#     with HDFArchive('../pomerol_ref.h5', 'w') as ar:
#         for x, y, z, w in product(range(2),repeat=4):
#             ar[f'%i%i%i%i'%(x,y,z,w)] = chi2[f'%i%i%i%i'%(x,y,z,w)]
#         ar["chi3_ph_AABB"] = chi3_ph_AABB
#         ar['energies'] = ed.energies
