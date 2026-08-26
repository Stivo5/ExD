from triqs.operators.util.hamiltonians import *
from triqs.operators.util import *
from triqs.operators import *
from triqs.gf import *
from triqs.atom_diag import *

from utils import *


import numpy as np
import sympy as sp



class ThreePtCorr:
    '''
    Compute various three-point functions.
    
    Parameters
    __________
    ops = list of pairs (operator, statistic)
            operator is a Triqs operator object, whose correlation is to be computed.
            statistic is 'Boson' or 'Fermion'.
    hamiltonian = (ad, eigensys)
                    ad is the AtomDiag object
                    eigensys is the assorted spectrum, (state, energy, state number)
    max_states: int
                upper cut-off on the number of states
    '''

    def __init__(self, ops, hamiltonian, max_states = 10000):
        '''
        Initialize the matrix elements.
        Compute signs between different operators

        Notes
        -----
        Matrix element convention: 
            A_mat[m][n] = <m|A|n>
        '''

        self.ops = ops
        self.hamiltonian = hamiltonian
        self.max_states = max_states

        ad = hamiltonian[0]
        eigensys = hamiltonian[1]
        
        hilb_dim = ad.full_hilbert_space_dim

        mats = []
        for op in ops:
            mat = []

            for m in range(min(max_states, hilb_dim)):
                mat_row = []

                for n in range(min(max_states, hilb_dim)):

                    m_state = np.zeros(hilb_dim)
                    m_state[eigensys[m][2]] = 1

                    n_state = np.zeros(hilb_dim)
                    n_state[eigensys[n][2]] = 1

                    mat_ele = np.dot(m_state, act(op[0], n_state, ad))
                    
                    mat_row.append(mat_ele)

                mat.append(mat_row)
                
            mats.append(mat)

        self.mats = mats


        mul_sgns = []
        cml_sgns = []
        for m in range(3):
            sgn_row = []
            cml_sgn = 1
            for n in range(3):
                if n == m: sgn_row.append(0) # No need to compute exchange statistic with self
                
                else:
                    s = sgn(ops[m], ops[n])
                
                    sgn_row.append(s)
                    cml_sgn = cml_sgn * s
                
            mul_sgns.append(sgn_row)
            cml_sgns.append(cml_sgn)

        self.sgns = [mul_sgns, cml_sgns]




    def G_div_diff_iw(self, beta, n_iw):
        '''
        Compute the imaginary frequency three-point funtion
        
        n_iw: list of 2 mesh point numbers [n1_iw, n2_iw]
        '''

        mats = self.mats
        max_states = self.max_states
        ad = self.hamiltonian[0]
        eigensys = self.hamiltonian[1]
        sgns = self.sgns

        E = [x[1] for x in eigensys] # Energy
        hilb_dim = ad.full_hilbert_space_dim # Total Hilbert space dimension
        Z = partition_function(atom = ad, beta = beta) # Partition function

        iw_mesh = []
        for n in range(2):
            if self.sgns[1][n] == 1:
                mesh = MeshImFreq(beta = beta, statistic = 'Boson', n_iw = n_iw[n])
            else:
                mesh = MeshImFreq(beta = beta, statistic = 'Fermion', n_iw = n_iw[n])
            iw_mesh.append(mesh)
        iw_prod_mesh = MeshProduct(iw_mesh[0], iw_mesh[1])

        G = []
        for iw1 in iw_mesh[0]:
            G_iw1 = []
            for iw2 in iw_mesh[1]:
                G_iw1iw2 = 0

                # Sum over states
                for m in range(min(max_states, hilb_dim)):
                    for n in range(min(max_states, hilb_dim)):
                        for k in range(min(max_states, hilb_dim)):
                            
                            # tau1 > tau2
                            alp1 = iw1 + E[m] - E[n]
                            alp2 = iw1 + iw2 + E[m] - E[k]
                            I = div_diff_exp_int_1ord([alp1, alp2], beta = beta)
                            G_iw1iw2 += np.exp(-beta * E[m]) * mats[0][m][n] * mats[1][n][k] * mats[2][k][m] * I

                            # tau2 > tau1
                            sgn = sgns[0][0][1]

                            alp1 = iw2 + E[m] - E[n]
                            alp2 = iw1 + iw2 + E[m] - E[k]
                            I = div_diff_exp_int_1ord([alp1, alp2], beta = beta)
                            G_iw1iw2 += sgn * np.exp(-beta * E[m]) * mats[1][m][n] * mats[0][n][k] * mats[2][k][m] * I
                
                G_iw1iw2 = -G_iw1iw2 / Z
                G_iw1.append(G_iw1iw2)
            
            G.append(G_iw1)
        
        data = np.array(G)
        G_iw = Gf(mesh = iw_prod_mesh, data = data)
        return G_iw
                
                
