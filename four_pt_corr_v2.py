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





@numba.njit
def _precompute_coefficients(E, M0, M1, M2, M3, boltz, exchange_signs):
    N = len(E)
    ex0 = exchange_signs[0]
    ex1 = exchange_signs[1]
    ex2 = exchange_signs[2]
    
    # Pass 1: Count non-zero transitions to pre-allocate exact arrays
    count = 0
    for m in range(N):
        bm = boltz[m]
        if abs(bm) < 1e-16:
            continue
            
        for n in range(N):
            a0 = bm * M0[m, n]
            a1 = bm * M1[m, n]
            a2 = bm * M2[m, n]
            
            # If all 'a' parameters are zero, skip k and l entirely
            if abs(a0) < 1e-14 and abs(a1) < 1e-14 and abs(a2) < 1e-14:
                continue
                
            for k in range(N):
                for l in range(N):
                    # Every term relies on M3[l, m]. If it's zero, skip!
                    m3_lm = M3[l, m]
                    if abs(m3_lm) < 1e-14:
                        continue
                        
                    c1 = a0 * M1[n, k] * M2[k, l] * m3_lm
                    c2 = ex2 * a0 * M2[n, k] * M1[k, l] * m3_lm
                    c3 = ex0 * a1 * M0[n, k] * M2[k, l] * m3_lm
                    c4 = ex0 * ex1 * a1 * M2[n, k] * M0[k, l] * m3_lm
                    c5 = ex1 * ex2 * a2 * M0[n, k] * M1[k, l] * m3_lm
                    c6 = ex0 * ex1 * ex2 * a2 * M1[n, k] * M0[k, l] * m3_lm
                    
                    if abs(c1)>0 or abs(c2)>0 or abs(c3)>0 or abs(c4)>0 or abs(c5)>0 or abs(c6)>0:
                        count += 1
                        
    # Allocate tight arrays
    out_m = np.zeros(count, dtype=np.int32)
    out_n = np.zeros(count, dtype=np.int32)
    out_k = np.zeros(count, dtype=np.int32)
    out_l = np.zeros(count, dtype=np.int32)
    out_C1 = np.zeros(count, dtype=np.complex128)
    out_C2 = np.zeros(count, dtype=np.complex128)
    out_C3 = np.zeros(count, dtype=np.complex128)
    out_C4 = np.zeros(count, dtype=np.complex128)
    out_C5 = np.zeros(count, dtype=np.complex128)
    out_C6 = np.zeros(count, dtype=np.complex128)
    
    # Pass 2: Fill the pre-allocated arrays
    idx = 0
    for m in range(N):
        bm = boltz[m]
        if abs(bm) < 1e-16:
            continue
            
        for n in range(N):
            a0 = bm * M0[m, n]
            a1 = bm * M1[m, n]
            a2 = bm * M2[m, n]
            
            if abs(a0) < 1e-14 and abs(a1) < 1e-14 and abs(a2) < 1e-14:
                continue
                
            for k in range(N):
                for l in range(N):
                    m3_lm = M3[l, m]
                    if abs(m3_lm) < 1e-14:
                        continue
                        
                    c1 = a0 * M1[n, k] * M2[k, l] * m3_lm
                    c2 = ex2 * a0 * M2[n, k] * M1[k, l] * m3_lm
                    c3 = ex0 * a1 * M0[n, k] * M2[k, l] * m3_lm
                    c4 = ex0 * ex1 * a1 * M2[n, k] * M0[k, l] * m3_lm
                    c5 = ex1 * ex2 * a2 * M0[n, k] * M1[k, l] * m3_lm
                    c6 = ex0 * ex1 * ex2 * a2 * M1[n, k] * M0[k, l] * m3_lm
                    
                    if abs(c1)>0 or abs(c2)>0 or abs(c3)>0 or abs(c4)>0 or abs(c5)>0 or abs(c6)>0:
                        out_m[idx] = m
                        out_n[idx] = n
                        out_k[idx] = k
                        out_l[idx] = l
                        out_C1[idx] = c1
                        out_C2[idx] = c2
                        out_C3[idx] = c3
                        out_C4[idx] = c4
                        out_C5[idx] = c5
                        out_C6[idx] = c6
                        idx += 1
                        
    return out_m, out_n, out_k, out_l, out_C1, out_C2, out_C3, out_C4, out_C5, out_C6









@numba.njit(parallel=True, fastmath=True)
def _four_point_kernel_fast(
    E,
    valid_m, valid_n, valid_k, valid_l,
    C1, C2, C3, C4, C5, C6,
    iw1, iw2, iw3, beta
):
    n_iw1 = len(iw1)
    n_iw2 = len(iw2)
    n_iw3 = len(iw3)
    num_valid = len(valid_m)

    out = np.zeros(n_iw1 * n_iw2 * n_iw3, dtype=np.complex128)

    for q in numba.prange(n_iw1 * n_iw2 * n_iw3):
        i = q // (n_iw2 * n_iw3)
        rem = q % (n_iw2 * n_iw3)
        j = rem // n_iw3
        s = rem % n_iw3

        w1 = iw1[i]
        w2 = iw2[j]
        w3 = iw3[s]

        total = 0.0j

        # We now ONLY loop over the non-zero (m,n,k,l) combinations
        for idx in range(num_valid):
            m = valid_m[idx]
            n = valid_n[idx]
            k = valid_k[idx]
            l = valid_l[idx]

            Emn = E[m] - E[n]
            Emk = E[m] - E[k]
            Eml = E[m] - E[l]

            # We also skip evaluating divided differences if a specific permutation is 0
            
            # tau1 > tau2 > tau3
            if abs(C1[idx]) > 0:
                total += C1[idx] * _div_diff_2(w1 + Emn, w1 + w2 + Emk, w1 + w2 + w3 + Eml, beta)

            # tau1 > tau3 > tau2
            if abs(C2[idx]) > 0:
                total += C2[idx] * _div_diff_2(w1 + Emn, w1 + w3 + Emk, w1 + w2 + w3 + Eml, beta)

            # tau2 > tau1 > tau3
            if abs(C3[idx]) > 0:
                total += C3[idx] * _div_diff_2(w2 + Emn, w2 + w1 + Emk, w2 + w1 + w3 + Eml, beta)

            # tau2 > tau3 > tau1
            if abs(C4[idx]) > 0:
                total += C4[idx] * _div_diff_2(w2 + Emn, w2 + w3 + Emk, w2 + w3 + w1 + Eml, beta)

            # tau3 > tau1 > tau2
            if abs(C5[idx]) > 0:
                total += C5[idx] * _div_diff_2(w3 + Emn, w3 + w1 + Emk, w3 + w1 + w2 + Eml, beta)

            # tau3 > tau2 > tau1
            if abs(C6[idx]) > 0:
                total += C6[idx] * _div_diff_2(w3 + Emn, w3 + w2 + Emk, w3 + w2 + w1 + Eml, beta)

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

        # Numba works best when lists are converted to typed numpy arrays
        exchange_signs_arr = np.array([self.sgns[0][0][1], self.sgns[0][0][2], self.sgns[0][1][2]], dtype=np.float64)

        # 1. Precalculate frequency-independent weights and locate sparse matrices
        sparse_args = _precompute_coefficients(
            E,
            np.ascontiguousarray(mats[0]),
            np.ascontiguousarray(mats[1]),
            np.ascontiguousarray(mats[2]),
            np.ascontiguousarray(mats[3]),
            boltz,
            exchange_signs_arr
        )

        # 2. Compute the 4-point kernel using ONLY the non-zero paths
        data = _four_point_kernel_fast(
            E,
            *sparse_args,
            iw1,
            iw2,
            iw3,
            beta
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
