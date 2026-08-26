from triqs.operators.util.hamiltonians import *
from triqs.operators.util import *
from triqs.operators import *
from triqs.gf import *
from triqs.atom_diag import *

import sys


import numpy as np
import sympy as sp



#*************************************************************************************
# Print out information about the states
def sort_states_full(spin_names,orb_names,ad,verbose=True):
    '''
    Sort eigenstates and write them in fock basis
    
    Inputs:
    orb_names: List of orbitals
    spin_names: List of spins
    ad: Solution to the atomic problem

    Outputs:
    eigensys: List in the form [states, energies, eigenstate #]
    
    '''

    n_orb=len(orb_names)
    n_spin=len(spin_names)

    
    n_eigenvec=0.0
    eigensys=[]

    # Loop through subspaces
    for sub in range(ad.n_subspaces):
        
        # convert states in computational basis
        subspace_fock_state=[]
        print('states in subspace ', sub, ': ')
        for fs in ad.fock_states[sub]:
            state =int(bin(fs)[2:])
                
            state_len=n_orb*n_spin
            # to fill in the leading 0's in computational basis
            state_comp = f'{state:0{state_len}d}'
            subspace_fock_state.append(state_comp)
            print('|', state_comp, '>')
       

        u_mat=ad.unitary_matrices[sub].conj().T
        st_mat=subspace_fock_state
            
        # store state and energy    
        for ind in range(ad.get_subspace_dim(sub)):

            eng=ad.energies[sub][ind]
            eigensys.append([[u_mat[ind,:],st_mat],eng,int(n_eigenvec)])

            # Keep track of the absolute number of eigenstate
            n_eigenvec += 1

    # Sort by energy from smallest to largest
    eigensys.sort(key=take_second)

    if verbose:
        print('Finished sorting states')

    return eigensys
#*************************************************************************************










#*************************************************************************************
# For sorting
def take_second(elem):
    return elem[1]
#*************************************************************************************
















