from triqs.operators.util.hamiltonians import *
from triqs.operators.util import *
from triqs.operators import *
from triqs.gf import *
from triqs.atom_diag import *

import sys
sys.path.append("/Users/ShuyuZhang/Documents/Projects/Cyrus/Codes/d_orbital_atom_Cyrus/defect_atom_diag/")
from def_atom_diag import *


import numpy as np
import sympy as sp



class TwoPtCorr:
    '''
    Compute various two-point functions.
    
    Parameters
    __________
    ops = (A, B, statistic)
            A and B are operators whose correlation is to be computed.
            statistic is 'Boson' or 'Fermion', giving the statistics between A and B
    hamiltonian = (ad, eigensys)
                    ad is the AtomDiag object
                    eigensys is the assorted spectrum
    max_states: int
                upper cut-off on the number of states
    '''

    def __init__(self, ops, hamiltonian, max_states = 10000):
        '''
        Initialize the matrix elements.

        Notes
        -----
        Matrix element convention: 
            A_mat[m][n] = <m|A|n>
        '''

        self.ops = ops
        self.hamiltonian = hamiltonian
        self.max_states = max_states

        A = ops[0]
        B = ops[1]

        ad = hamiltonian[0]
        hilb_dim = ad.full_hilbert_space_dim

        eigensys = hamiltonian[1]

        A_mat = []
        B_mat = []
        for m in range(min(max_states, hilb_dim)):

            A_mat_row = []
            B_mat_row = []

            for n in range(min(max_states, hilb_dim)):

                m_state = np.zeros(hilb_dim)
                m_state[eigensys[m][2]] = 1

                n_state = np.zeros(hilb_dim)
                n_state[eigensys[n][2]] = 1

                A_mat_ele = np.dot(m_state, act(A, n_state, ad))
                B_mat_ele = np.dot(m_state, act(B, n_state, ad))

                A_mat_row.append(A_mat_ele)
                B_mat_row.append(B_mat_ele)

            A_mat.append(A_mat_row)
            B_mat.append(B_mat_row)
            
        A_mat = np.array(A_mat)
        B_mat = np.array(B_mat)

        mat = [A_mat, B_mat]  
        self.mat = mat



    # def G_AB_w(self, beta, n_w, eta = 0.1):

