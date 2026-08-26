from h5 import HDFArchive
from triqs.atom_diag import AtomDiagComplex
from triqs.operators import n, c, c_dag
from itertools import product
from triqs.plot.mpl_interface import oplot

from sort import sort_states_full
from three_pt_corr import *





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
atoms = (0, 1, 2, 3)

# Number of bosonic Matsubara frequencies for susceptibility calculation
n_iw = 10
# Number of fermionic Matsubara frequencies for susceptibility calculation
n_inu = 10

# Block structure of \chi^3
gf_struct = [['up', 4], ['dn', 4]]

H = sum(e * n(s, a) for (a, e), s in product(zip(atoms, eps), spin_names))
H += sum(-h_field * (n('up', a) - n('dn', a)) for a in atoms)
H += U * sum(n('up', a) * n('dn', a) for a in atoms)
H += t * sum((c_dag(sp, 0) * c(sp, 1) + c_dag(sp, 1) * c(sp, 0)) for sp in spin_names)
H += t * sum((c_dag(sp, 2) * c(sp, 3) + c_dag(sp, 3) * c(sp, 2)) for sp in spin_names)
H += t * sum((c_dag(sp, 0) * c(sp, 3) + c_dag(sp, 3) * c(sp, 0)) for sp in spin_names)

fops = list(product(spin_names, atoms))






ad = AtomDiagComplex(H, fops)




eigensys = sort_states_full(spin_names, atoms, ad)





ops = [(c_dag('up', 1), 'Fermion'),
       (c('up',0), 'Fermion'),
       (c('up', 0)*c_dag('up',1), 'Boson')]

corr = ThreePtCorr(ops = ops, hamiltonian = (ad, eigensys))

G_iw = corr.G_div_diff_iw(beta = beta, n_iw = [10,10])




# G = Gf(mesh = iw_mesh2, target_shape = [])
# G.zero()

# for iw1 in iw_mesh2:
#     for iw2 in iw_mesh2:
#         G[iw1] += G_iw[iw1,iw2]



G = Gf(mesh = iw_mesh2, target_shape = [])
G.zero()

for iw2 in iw_mesh2:
    for iw1 in iw_mesh2:
        G[iw2] += G_iw[iw1,iw2]


