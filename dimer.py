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

beta = 50.0              # Inverse temperature

biw_mesh = MeshImFreq(beta = beta, statistic = 'Boson', n_iw = bn_iw)
fiw_mesh = MeshImFreq(beta = beta, statistic = 'Fermion', n_iw = fn_iw)



# Hubbard dimer

####################
# Input parameters #
####################

eps = [-1.9, -2.1]      # Energy levels of the atoms
t = 0.5                 # Hopping matrix element
U = 4.0                 # Coulomb repulsion
h_field = 0.05          # Magnetic field

spin_names = ("up", "dn")
atoms = (0, 1)

H = sum(e * n(s, a) for (a, e), s in product(zip(atoms, eps), spin_names))
H += sum(-h_field * (n('up', a) - n('dn', a)) for a in atoms)
H += U * sum(n('up', a) * n('dn', a) for a in atoms)
H += t * sum((c_dag(sp, 0) * c(sp, 1) + c_dag(sp, 1) * c(sp, 0)) for sp in spin_names)



# fops = list(product(spin_names, atoms))

# Diagonalize Hamiltonian
ad = AtomDiagComplex(H, fops)

eigensys = sort_states_full(spin_names, atoms, ad)



G_iw = dict()
for x, y, z, w in product(range(2),repeat=4):
    ops = [(c_dag('up', x)*c('up', y), 'Boson'),
           (c_dag('up',z), 'Fermion'),
           (c('up', w), 'Fermion')]
    
    corr = ThreePtCorr(ops = ops, hamiltonian = (ad, eigensys))
    
    # G = corr.G_div_diff_iw(beta = beta, n_iw = [10,10])
    G_iw[f'%i%i%i%i'%(x, y, z, w)] = corr.G_div_diff_iw(beta = beta, n_iw = [bn_iw,fn_iw])

# Integrate over the second frequency
G_int1 = dict()
for x, y, z, w in product(range(2),repeat=4):
    
    G = Gf(mesh = biw_mesh, target_shape = [])
    G.zero()
    
    for iw1 in biw_mesh:
        for iw2 in fiw_mesh:
            G[iw1] += G_iw[f'%i%i%i%i'%(x, y, z, w)][iw1,iw2]

    G_int1[f'%i%i%i%i'%(x, y, z, w)] = G / beta



if mpi.is_master_node():
    with HDFArchive('ref.h5', 'w') as ar:
        for x, y, z, w in product(range(2),repeat=4):
            ar[f'%i%i%i%i'%(x,y,z,w)] = G_int1[f'%i%i%i%i'%(x,y,z,w)]
