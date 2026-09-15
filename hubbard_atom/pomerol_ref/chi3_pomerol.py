from h5 import HDFArchive
from triqs.operators import c, c_dag, n
from triqs.utility import mpi
from triqs.utility.comparison_tests import *
from pomerol2triqs import PomerolED
from itertools import product

# Hubbard atom

####################
# Input parameters #
####################

beta = 2.0

# eps = [-1.9, -2.1]      # Energy levels of the atoms
# h_field = 0.05          # Magnetic field
U = 4.0                 # Coulomb repulsion
mu = U/2              # Chemical potential    

spin_names = ("up", "dn")
atoms = (0,)

# Number of bosonic Matsubara frequencies for susceptibility calculation
n_iw = 10
# Number of fermionic Matsubara frequencies for susceptibility calculation
n_inu = 10

# Block structure of \chi^3
gf_struct = [['up', 1], ['dn', 1]]

# Conversion from TRIQS to Pomerol notation for operator indices
index_converter = {}
index_converter.update({(sn, 0) : ("A", 0, "down" if sn == "dn" else "up") for sn in spin_names})
# index_converter.update({(sn, 1) : ("B", 0, "down" if sn == "dn" else "up") for sn in spin_names})

# Make PomerolED solver object
ed = PomerolED(index_converter, verbose = True)

# Hamiltonian
# H = sum(e * n(s, a) for (a, e), s in product(zip(atoms, eps), spin_names))
# H += sum(-h_field * (n('up', a) - n('dn', a)) for a in atoms)
H = U * n('up', 0) * n('dn', 0)
H += -mu * sum(n(s,0) for s in spin_names)


# Diagonalize H
ed.diagonalize(H)

##################
# Compute \chi^3 #
##################

chi2 = ed.chi_iw(("up", 0), ("up", 0), ("up", 0), ("up", 0),beta,n_iw, connected=False)

params = {'gf_struct': gf_struct, 'beta': beta, 'n_iw': 20, 'n_inu': 20}
chi3_ph_AABB = ed.chi3_iw_inu(**params, channel='PH', block_order='AABB')


if mpi.is_master_node():
    with HDFArchive('../pomerol_ref.h5', 'w') as ar:
        ar['chi2'] = chi2
        ar["chi3_ph_AABB"] = chi3_ph_AABB
