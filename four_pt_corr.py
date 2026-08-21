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



class FourPtCorr:
    '''
    Compute various four-point functions.
    
    Parameters
    __________
    ops = list of (A, Statistic)
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
        Compute signs of operators

        Notes
        -----
        Matrix element convention: 
            A_mat[m][n] = <m|A|n>
        '''

        self.ops = ops
        self.hamiltonian = hamiltonian
        self.max_states = max_states

        A = ops[0][0]
        B = ops[1][0]
        C = ops[2][0]
        D = ops[3][0]

        # Compute matrix elements
        ad = hamiltonian[0]
        hilb_dim = ad.full_hilbert_space_dim

        eigensys = hamiltonian[1]

        A_mat = []
        B_mat = []
        C_mat = []
        D_mat = []
        for m in range(min(max_states, hilb_dim)):

            A_mat_row = []
            B_mat_row = []
            C_mat_row = []
            D_mat_row = []

            for n in range(min(max_states, hilb_dim)):

                # Bra state
                m_state = np.zeros(hilb_dim)
                m_state[eigensys[m][2]] = 1

                # Ket state
                n_state = np.zeros(hilb_dim)
                n_state[eigensys[n][2]] = 1

                A_mat_ele = np.dot(m_state, act(A, n_state, ad))
                B_mat_ele = np.dot(m_state, act(B, n_state, ad))
                C_mat_ele = np.dot(m_state, act(C, n_state, ad))
                D_mat_ele = np.dot(m_state, act(D, n_state, ad))

                A_mat_row.append(A_mat_ele)
                B_mat_row.append(B_mat_ele)
                C_mat_row.append(C_mat_ele)
                D_mat_row.append(D_mat_ele)
                

            A_mat.append(A_mat_row)
            B_mat.append(B_mat_row)
            C_mat.append(C_mat_row)
            D_mat.append(D_mat_row)
            
        A_mat = np.array(A_mat)
        B_mat = np.array(B_mat)
        C_mat = np.array(C_mat)
        D_mat = np.array(D_mat)

        mat = [A_mat, B_mat, C_mat, D_mat]  
        self.mat = mat

        
        # Compute signs of operators
        sgn_ij = []
        for i in range(4):
            sgn_row = []
            for j in range(4):
                s = sgn(ops[i], ops[j])
                sgn_row.append(s)
            sgn_ij.append(sgn_row)

        sgn_i = []
        for i in range(4):
            s = 1
            for j in range(4):
                if j == i:
                    continue
                s = s * sgn_ij[i][j]
            sgn_i.append(s)
        
        sgns = [sgn_ij, sgn_i]
        self.sgns = sgns


    # def G_AB_w(self, beta, n_w, eta = 0.1):


    def F(self, order, beta, n_iw):
        '''
        A intermediate step in computing the time ordered correlation
        
        Notes
        -----
        order: a permutation of [0,1,2]
                Says we are ordering as ops[order[0]] ops[order[1]] ops[order[2]] ops[3]
        '''
        
        sgn_i = self.sgns[1]
        hilb_dim = self.hamiltonian[0].full_hilbert_space_dim
        eigensys = self.hamiltonian[1]
        mat = self.mat
        E = [x[1] for x in eigensys]        

        # Create mesh first
        meshes = []
        for i in range(3):
            if sgn_i[i] == 1:
                iw_mesh = MeshImFreq(beta = beta, n_iw = n_iw[i], statistic = 'Boson')
                meshes.append(iw_mesh)
            elif sgn_i[i] == -1:
                iw_mesh = MeshImFreq(beta = beta, n_iw = n_iw[i], statistic = 'Fermion')
                meshes.append(iw_mesh)
        
        prod_mesh = MeshProduct(meshes[0], meshes[1], meshes[2])

        # Create data for the Green's function on product mesh
        
        F = []
        for iw1 in meshes[0]:
            F_iw1 = []
            for iw2 in meshes[1]:
                F_iw1iw2 = []
                for iw3 in meshes[2]:
                    
                    # The mesh points are to be shuffuled according to order
                    iw = [iw1,iw2,iw3]
                    
                    # Sum over eigen-states
                    sum = 0
                    for m in range(hilb_dim):
                        for n in range(hilb_dim):
                            for k in range(hilb_dim):
                                for l in range(hilb_dim):
                                    alp = iw[order[0]] + E[m] - E[n]
                                    gam = iw[order[1]] + E[n] - E[k]
                                    kap = iw[order[2]] + E[k] - E[l]
                                    
                                    # print(m,n,k,l,(kap * (gam + kap) * (alp + gam + kap)))
                                    term1 = (
                                        sgn_i[order[0]] * sgn_i[order[1]] * sgn_i[order[2]]
                                        / (kap * (gam + kap) * (alp + gam + kap))
                                        * np.exp(beta * (E[m]) - E[l])
                                    )

                                    term2 = (
                                        sgn_i[order[0]] * sgn_i[order[1]]
                                        / (gam * kap * (alp + gam))
                                        * np.exp(beta * (E[m]) - E[k])
                                    )

                                    term3 = (
                                        sgn_i[order[0]]
                                        / (alp * (gam + kap))
                                        * np.exp(beta * (E[m]) - E[n])
                                    )

                                    term4 = (
                                        1 / (gam + kap)
                                        * (
                                            (alp + 2*gam + kap)
                                            / (gam * (alp + gam) * (alp + gam + kap))
                                            - 1 / alp
                                        )
                                    )

                                    S = (
                                        np.exp(beta * E[m])
                                        * mat[order[0]][m][n]
                                        * mat[order[1]][n][k]
                                        * mat[order[2]][k][l]
                                        * mat[3][k][m]
                                        * (term1 - term2 + term3 + term4)
                                    )

                                    sum += S

                    F_iw1iw2.append(sum)
                F_iw1.append(F_iw1iw2)
            F.append(F_iw1)
        
        data = np.array(F)
        
        F = Gf(mesh = prod_mesh, data = data)

        return F