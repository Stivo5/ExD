from triqs.operators.util.hamiltonians import *
from triqs.operators.util import *
from triqs.operators import *
from triqs.gf import *
from triqs.atom_diag import *

from utils import *


import numba
import numpy as np
import sympy as sp









@numba.njit(inline="always")
def _exp_int(x, beta):
    """
    Integral:
        int_0^beta exp(x*t) dt

    Equivalent to utils.exp_int(), but Numba compatible.
    """
    if abs(x) > 1e-8:
        return np.expm1(x * beta) / x
    else:
        return beta


@numba.njit(inline="always")
def _exp_int_d1(x, beta):
    """
    First derivative of exp_int with respect to x.
    """
    if abs(x) > 1e-8:
        eb = np.exp(x * beta)
        return beta * eb / x - np.expm1(x * beta) / (x * x)
    else:
        return beta * beta / 2.0




@numba.njit(inline="always")
def _exp_int_d2(x, beta):
    """
    Second derivative of exp_int with respect to x.
    """
    if abs(x) > 1e-8:
        eb = np.exp(x * beta)
        return beta * beta * eb / x - 2 * beta * eb / (x * x) + 2 * np.expm1(x * beta) / (x * x * x)
    else:
        return beta * beta * beta / 3.0





@numba.njit(inline="always")
def _div_diff_2(x0, x1, x2, beta):
    """
    Second divided difference of exp_int.
    """
    d01 = x0 - x1
    d12 = x1 - x2
    d02 = x0 - x2

    # 2. Check tolerance flags
    e01 = abs(d01) <= 1e-8
    e12 = abs(d12) <= 1e-8
    e02 = abs(d02) <= 1e-8

    matches = e01 + e12 + e02

    # Case 1: 3rd order pole
    if matches >= 2:
        return _exp_int_d2(x0, beta) / 2

    # Case 2: 1 2nd order pole, and 1 1st order pole
    elif matches == 1:
        if e01 == 1:
            return _exp_int_d1(x0, beta) / d02 + _exp_int(x2, beta) / (d02 * d02)

        elif e02 == 1:
            return _exp_int_d1(x0, beta) / d01 + _exp_int(x1, beta) / (d01 * d01)

        else:
            return _exp_int_d1(x1, beta) / (-d01) + _exp_int(x0, beta) / (d01 * d01)

    # Case 3: 3 1st order poles  
    else:
        return _exp_int(x0, beta) / (d01 * d02) + _exp_int(x1, beta) / (-d01 * d12) + _exp_int(x2, beta) / (d02 * d12)





@numba.njit(parallel=True, fastmath=False)
def _four_point_kernel(
    E,
    M0,
    M1,
    M2,
    M3,
    iw1,
    iw2,
    iw3,
    boltz,
    beta,
    exchange_signs,
):
    """
    Compute the complete four-point frequency grid.
    The output is flattened as (iw1, iw2, iw3).
    exchange_signs is the statistics as the list [01,02,12]
    """

    n_iw1 = len(iw1)
    n_iw2 = len(iw2)
    n_iw3 = len(iw3)
    N = len(E)

    out = np.zeros(n_iw1 * n_iw2 * n_iw3, dtype=np.complex128)

    # Parallelize over frequency points.
    #
    # Each frequency point is independent, which avoids race
    # conditions while still exposing a large amount of parallelism.
    for q in numba.prange(n_iw1 * n_iw2 * n_iw3):

        i = q // (n_iw2 * n_iw3)
        j = (q - i * n_iw2 * n_iw3) // n_iw3
        s = q - i * n_iw2 * n_iw3 - j * n_iw3

        w1 = iw1[i]
        w2 = iw2[j]
        w3 = iw3[s]

        total = 0.0j

        for m in range(N):

            Em = E[m]
            bm = boltz[m]

            for n in range(N):

                Emn = Em - E[n]

                # Matrix elements independent of k.
                a0 = bm * M0[m, n]
                a1 = bm * M1[m, n]
                a2 = bm * M2[m, n]

                for k in range(N):

                    Emk = Em - E[k]

                    for l in range(N):

                        Eml = Em - E[l]
    
                        # --------------------------------------------------
                        # tau1 > tau2 > tau3
                        # --------------------------------------------------
    
                        x0 = w1 + Emn
                        x1 = w1 + w2 + Emk
                        x2 = w1 + w2 + w3 + Eml
    
                        I = _div_diff_2(x0, x1, x2, beta)
    
                        total += (
                            a0
                            * M1[n, k]
                            * M2[k, l]
                            * M3[l, m]
                            * I
                        )
    
                        # --------------------------------------------------
                        # tau1 > tau3 > tau2
                        # --------------------------------------------------
    
                        x0 = w1 + Emn
                        x1 = w1 + w3 + Emk
                        x2 = w1 + w2 + w3 + Eml
    
                        I = _div_diff_2(x0, x1, x2, beta)
    
                        total += (
                            exchange_signs[2]
                            * a0
                            * M2[n, k]
                            * M1[k, l]
                            * M3[l, m]
                            * I
                        )
                        
                        # --------------------------------------------------
                        # tau2 > tau1 > tau3
                        # --------------------------------------------------
    
                        x0 = w2 + Emn
                        x1 = w2 + w1 + Emk
                        x2 = w2 + w1 + w3 + Eml
    
                        I = _div_diff_2(x0, x1, x2, beta)
    
                        total += (
                            exchange_signs[0]
                            * a1
                            * M0[n, k]
                            * M2[k, l]
                            * M3[l, m]
                            * I
                        )
                        
                        # --------------------------------------------------
                        # tau2 > tau3 > tau1
                        # --------------------------------------------------
    
                        x0 = w2 + Emn
                        x1 = w2 + w3 + Emk
                        x2 = w2 + w3 + w1 + Eml
    
                        I = _div_diff_2(x0, x1, x2, beta)
    
                        total += (
                            exchange_signs[0] * exchange_signs[1]
                            * a1
                            * M2[n, k]
                            * M0[k, l]
                            * M3[l, m]
                            * I
                        )
                        
                        # --------------------------------------------------
                        # tau3 > tau1 > tau2
                        # --------------------------------------------------
    
                        x0 = w3 + Emn
                        x1 = w3 + w1 + Emk
                        x2 = w3 + w1 + w2 + Eml
    
                        I = _div_diff_2(x0, x1, x2, beta)
    
                        total += (
                            exchange_signs[1] * exchange_signs[2]
                            * a2
                            * M0[n, k]
                            * M1[k, l]
                            * M3[l, m]
                            * I
                        )
                        
                        # --------------------------------------------------
                        # tau3 > tau2 > tau1
                        # --------------------------------------------------
    
                        x0 = w3 + Emn
                        x1 = w3 + w2 + Emk
                        x2 = w3 + w2 + w1 + Eml
    
                        I = _div_diff_2(x0, x1, x2, beta)
    
                        total += (
                            exchange_signs[0] * exchange_signs[1] * exchange_signs[2]
                            * a2
                            * M1[n, k]
                            * M0[k, l]
                            * M3[l, m]
                            * I
                        )

        out[q] = -total

    return out









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

        ad = hamiltonian[0]
        eigensys = hamiltonian[1]

        hilb_dim = ad.full_hilbert_space_dim

        # ---------------------------------------------------------------
        # Only retain the states that are actually used.
        # ---------------------------------------------------------------

        self.n_states = min(max_states, hilb_dim)

        eigensys = eigensys[:self.n_states]

        self.eigensys = eigensys

        self.state_indices = np.asarray(
            [x[2] for x in eigensys],
            dtype=np.int64,
        )

        self.energies = np.asarray(
            [x[1] for x in eigensys],
            dtype=np.float64,
        )

        # ---------------------------------------------------------------
        # Construct matrix elements.

        mats = []

        for op, statistic in ops:

            mat = np.empty(
                (self.n_states, self.n_states),
                dtype=np.complex128,
            )

            for n in range(self.n_states):

                n_state = np.zeros(
                    hilb_dim,
                    dtype=np.complex128,
                )

                n_state[self.state_indices[n]] = 1.0

                acted_state = act(op, n_state, ad)

                mat[:, n] = np.asarray(
                    acted_state[self.state_indices],
                    dtype=np.complex128,
                )

            mats.append(mat)

        self.mats = mats

        # ---------------------------------------------------------------
        # Exchange signs
        # ---------------------------------------------------------------

        mul_sgns = []
        cml_sgns = []

        for m in range(4):

            sgn_row = []
            cml_sgn = 1

            for n in range(4):

                if n == m:
                    sgn_row.append(0)
                else:
                    s = sgn(ops[m], ops[n])
                    sgn_row.append(s)

                cml_sgn *= (
                    1 if n == m
                    else sgn(ops[m], ops[n])
                )

            mul_sgns.append(sgn_row)
            cml_sgns.append(cml_sgn)

        self.sgns = [mul_sgns, cml_sgns]





    def G_div_diff_iw(self, beta, n_iw):

        """
        Compute the imaginary-frequency three-point function.

        Parameters
        ----------
        beta : float
            Inverse temperature.

        n_iw : list
            [n1_iw, n2_iw, n2_iw]

        Returns
        -------
        Gf
            TRIQS three-frequency Green's function.
        """

        ad = self.hamiltonian[0]

        E = self.energies
        mats = self.mats

        N = self.n_states

        # ---------------------------------------------------------------
        # Partition function
        # ---------------------------------------------------------------

        Z = partition_function(
            atom=ad,
            beta=beta,
        )

        # ---------------------------------------------------------------
        # Construct frequency meshes
        # ---------------------------------------------------------------

        iw_mesh = []

        for n in range(3):

            if self.sgns[1][n] == 1:
                statistic = "Boson"
            else:
                statistic = "Fermion"

            mesh = MeshImFreq(
                beta=beta,
                statistic=statistic,
                n_iw=n_iw[n],
            )

            iw_mesh.append(mesh)

        iw_prod_mesh = MeshProduct(
            iw_mesh[0],
            iw_mesh[1],
            iw_mesh[2]
        )

        # ---------------------------------------------------------------
        # Extract Matsubara frequencies as ordinary complex arrays.
        #
        # Passing TRIQS mesh objects through the numerical kernel would
        # prevent Numba from doing its job.
        # ---------------------------------------------------------------

        iw1 = np.asarray(
            [complex(w) for w in iw_mesh[0]],
            dtype=np.complex128,
        )

        iw2 = np.asarray(
            [complex(w) for w in iw_mesh[1]],
            dtype=np.complex128,
        )

        iw3 = np.asarray(
            [complex(w) for w in iw_mesh[2]],
            dtype=np.complex128,
        )

        # ---------------------------------------------------------------
        # Boltzmann weights
        # ---------------------------------------------------------------

        boltz = np.exp(-beta * E)

        # ---------------------------------------------------------------
        # Main numerical calculation
        # ---------------------------------------------------------------

        print("start numerical calculations")

        data = _four_point_kernel(
            E,
            np.ascontiguousarray(mats[0]),
            np.ascontiguousarray(mats[1]),
            np.ascontiguousarray(mats[2]),
            np.ascontiguousarray(mats[3]),
            iw1,
            iw2,
            iw3,
            boltz,
            beta,
            [self.sgns[0][0][1], self.sgns[0][0][2], self.sgns[0][1][2]],
        )

        data = data.reshape(
            (len(iw1), len(iw2), len(iw3))
        )

        # ---------------------------------------------------------------
        # Normalize by partition function.
        # ---------------------------------------------------------------

        data /= Z

        return Gf(
            mesh=iw_prod_mesh,
            data=data,
        )
