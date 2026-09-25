from triqs.operators.util.hamiltonians import *
from triqs.operators.util import *
from triqs.operators import *
from triqs.gf import *
from triqs.atom_diag import *

from utils import *


from itertools import product

import numba
import numpy as np
import sympy as sp









_SERIES_RADIUS = 0.5   # use the Taylor series of exp_int when |beta * x| is below this
_SERIES_TERMS = 30     # 0.5^30 / 30! ~ 1e-42, far below double precision


@numba.njit(inline="always")
def _cexpm1(z):
    """
    exp(z) - 1 for complex z without cancellation.

    Numba's np.expm1 on complex numbers is computed as exp(z) - 1, which
    loses ~ 1e-16 / |z| relative accuracy; this uses the real expm1 instead:
        exp(x + iy) - 1 = expm1(x) cos(y) - 2 sin^2(y/2) + i exp(x) sin(y)
    """
    x = z.real
    y = z.imag
    s = np.sin(0.5 * y)
    return complex(np.expm1(x) * np.cos(y) - 2.0 * s * s, np.exp(x) * np.sin(y))


@numba.njit(inline="always")
def _exp_int(x, beta):
    """
    Integral:
        int_0^beta exp(x*t) dt

    Equivalent to utils.exp_int(), but Numba compatible.
    """
    z = x * beta
    if abs(z) < _SERIES_RADIUS:
        # beta * sum_j z^j / (j+1)!
        t = 1.0 + 0.0j
        s = 0.0j
        for j in range(_SERIES_TERMS):
            s += t
            t *= z / (j + 2)
        return beta * s
    return _cexpm1(z) / x


@numba.njit(inline="always")
def _exp_int_d1(x, beta):
    """
    First derivative of exp_int with respect to x.
    """
    z = x * beta
    if abs(z) < _SERIES_RADIUS:
        # beta^2 * sum_j (j+1) z^j / (j+2)!
        t = 0.5 + 0.0j
        s = 0.0j
        for j in range(_SERIES_TERMS):
            s += (j + 1) * t
            t *= z / (j + 3)
        return beta * beta * s
    eb = np.exp(z)
    return beta * eb / x - _cexpm1(z) / (x * x)




@numba.njit(inline="always")
def _exp_int_d2(x, beta):
    """
    Second derivative of exp_int with respect to x.
    """
    z = x * beta
    if abs(z) < _SERIES_RADIUS:
        # beta^3 * sum_j (j+1)(j+2) z^j / (j+3)!
        t = 1.0 / 6.0 + 0.0j
        s = 0.0j
        for j in range(_SERIES_TERMS):
            s += (j + 1) * (j + 2) * t
            t *= z / (j + 4)
        return beta * beta * beta * s
    eb = np.exp(z)
    return beta * beta * eb / x - 2 * beta * eb / (x * x) + 2 * _cexpm1(z) / (x * x * x)





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
            # return _exp_int_d1(x0, beta) / d02 + _exp_int(x2, beta) / (d02 * d02)
            return _exp_int_d1(x0, beta) / d02 + (_exp_int(x2, beta) - _exp_int(x0, beta)) / (d02 * d02)

        elif e02 == 1:
            # return _exp_int_d1(x0, beta) / d01 + _exp_int(x1, beta) / (d01 * d01)
            return _exp_int_d1(x0, beta) / d01 + (_exp_int(x1, beta) - _exp_int(x0, beta)) / (d01 * d01)

        else:
            # return _exp_int_d1(x1, beta) / (-d01) + _exp_int(x0, beta) / (d01 * d01)
            return _exp_int_d1(x1, beta) / (-d01) + (_exp_int(x0, beta) - _exp_int(x1, beta)) / (d01 * d01)

    # Case 3: 3 1st order poles  
    else:
        return _exp_int(x0, beta) / (d01 * d02) + _exp_int(x1, beta) / (-d01 * d12) + _exp_int(x2, beta) / (d02 * d12)





def _nonzero_pattern(mats, tol=1e-14):
    """
    CSR-style sparsity pattern of the union of the nonzeros of `mats`.

    Row r has nonzero columns cols[ptr[r]:ptr[r+1]].
    """
    mask = np.zeros(mats[0].shape, dtype=bool)
    for M in mats:
        mask |= np.abs(M) >= tol
    ptr = np.zeros(mask.shape[0] + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(mask.sum(axis=1))
    cols = np.nonzero(mask)[1].astype(np.int64)
    return ptr, cols


@numba.njit(cache=True)
def _precompute_coefficients(E, M0, M1, M2, M3, boltz, exchange_signs,
                             ptr012, cols012, ptr3T, cols3T):
    """
    Frequency-independent weights of all non-zero (m, n, k, l) paths.

    Instead of a dense N^4 loop, n and k run over the nonzero columns of
    M0|M1|M2 (pattern ptr012/cols012), and l runs over the nonzero rows of
    column m of M3 (pattern of M3^T, ptr3T/cols3T).
    """
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

        for jn in range(ptr012[m], ptr012[m + 1]):
            n = cols012[jn]
            a0 = bm * M0[m, n]
            a1 = bm * M1[m, n]
            a2 = bm * M2[m, n]

            # If all 'a' parameters are zero, skip k and l entirely
            if abs(a0) < 1e-14 and abs(a1) < 1e-14 and abs(a2) < 1e-14:
                continue

            for jk in range(ptr012[n], ptr012[n + 1]):
                k = cols012[jk]
                for jl in range(ptr3T[m], ptr3T[m + 1]):
                    l = cols3T[jl]
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

        for jn in range(ptr012[m], ptr012[m + 1]):
            n = cols012[jn]
            a0 = bm * M0[m, n]
            a1 = bm * M1[m, n]
            a2 = bm * M2[m, n]

            if abs(a0) < 1e-14 and abs(a1) < 1e-14 and abs(a2) < 1e-14:
                continue

            for jk in range(ptr012[n], ptr012[n + 1]):
                k = cols012[jk]
                for jl in range(ptr3T[m], ptr3T[m + 1]):
                    l = cols3T[jl]
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









_TOL = 1e-8   # same degeneracy tolerance as _div_diff_2 / _exp_int

# At a zero bosonic frequency, pole denominators are bare level spacings.
# For spacings that are small but nonzero, the three pole terms are each
# ~ 1/d^2 and cancel only after summing over all paths, which costs
# ~ eps / d^2 relative accuracy. Such pairs (|d| <= _NEAR) are therefore
# also routed to the per-path resonant correction.
_NEAR = 1e-2


def _drop_tol(w):
    """Denominators |d| <= _drop_tol(w) are left to _resonant_correction."""
    return _NEAR if abs(w) <= _TOL else _TOL


def _safe_inv(d, tol=_TOL):
    """1/d, with 0 where |d| <= tol (those terms are handled by the resonant correction)."""
    out = np.zeros_like(d)
    mask = np.abs(d) > tol
    out[mask] = 1.0 / d[mask]
    return out


def _boltz_exp_int(z, zeta, E, eb, beta, tol=_TOL):
    """
    Matrix g[p, q] = exp(-beta E_p) * exp_int(z + E_p - E_q, beta)
                   = (zeta exp(-beta E_q) - exp(-beta E_p)) / (z + E_p - E_q),
    with zeta = exp(beta z) = +-1 for a Matsubara frequency z.
    """
    dE = E[:, None] - E[None, :]
    x = z + dE
    if zeta > 0:
        # exp(-beta E_q) - exp(-beta E_p) cancels for close levels; use expm1 there.
        close = np.abs(beta * dE) < 1.0
        num = np.where(
            close,
            eb[:, None] * np.expm1(beta * np.where(close, dE, 0.0)),
            eb[None, :] - eb[:, None],
        )
    else:
        num = -(eb[None, :] + eb[:, None])
    small = np.abs(x) <= tol
    x_safe = np.where(small, 1.0, x)
    return np.where(small, beta * eb[:, None], num / x_safe)


def _ordered_term_factorized(E, beta, X, Y, Z, D, wa, wb, wc, za, zb, zc):
    """
    Regular (non-resonant) part of

        T[a, b, c] = sum_{mnkl} e^{-beta E_m} X_mn Y_nk Z_kl D_lm
                     * div_diff_2(wa + E_mn, wa + wb + E_mk, wa + wb + wc + E_ml)

    for all frequencies wa[a], wb[b], wc[c].

    The divided difference is split into its three pole terms,
        f(x0)/(d01 d02) - f(x1)/(d01 d12) + f(x2)/(d02 d12),
    where every denominator depends on only two state indices, so each term
    factorizes into N^3 matrix products per frequency instead of a sum over
    all (m, n, k, l) paths per frequency triple.

    Pairs with a vanishing denominator are dropped here (_safe_inv) and
    added back exactly by _resonant_correction.
    """
    na, nb, nc = len(wa), len(wb), len(wc)
    N = len(E)
    eb = np.exp(-beta * E)

    # dE[p, q] = E_q - E_p
    dE = E[None, :] - E[:, None]

    T = np.zeros((na, nb, nc), dtype=np.complex128)

    # Frequency-dependent inverse denominators
    inv_d01 = [_safe_inv(-wb[b] + dE, _drop_tol(wb[b])) for b in range(nb)]   # [n, k]
    inv_d12 = [_safe_inv(-wc[c] + dE, _drop_tol(wc[c])) for c in range(nc)]   # [k, l]

    # ---------------------------------------------------------------
    # Term 0: f(x0) / (d01 d02)
    #   = sum_{nl} P[a]_{ln} Q[b]_{nl} / d02(b + c)_{nl}
    # ---------------------------------------------------------------
    PT = np.empty((na, N * N), dtype=np.complex128)
    for a in range(na):
        P = D @ (X * _boltz_exp_int(wa[a], za[a], E, eb, beta))   # [l, n]
        PT[a] = P.T.ravel()

    Q = [(Y * inv_d01[b]) @ Z for b in range(nb)]                  # [n, l]

    # ---------------------------------------------------------------
    # Term 2 pieces: f(x2) / (d02 d12)
    #   = sum_{nl} H(a + b + c)_{nl} K[c]_{nl} / d02(b + c)_{nl}
    # ---------------------------------------------------------------
    K = [Y @ (Z * inv_d12[c]) for c in range(nc)]                  # [n, l]
    DT = np.ascontiguousarray(D.T)
    XT = np.ascontiguousarray(X.T)
    H_cache = {}

    def H_of(w, zeta):
        key = round(w.imag * beta / np.pi)
        if key not in H_cache:
            H_cache[key] = (XT @ (DT * _boltz_exp_int(w, zeta, E, eb, beta))).ravel()
        return H_cache[key]

    for b in range(nb):
        for c in range(nc):
            inv_d02 = _safe_inv(-(wb[b] + wc[c]) + dE, _drop_tol(wb[b] + wc[c]))   # [n, l]

            # Term 0
            T[:, b, c] += PT @ (Q[b] * inv_d02).ravel()

            # Term 2
            KR = (K[c] * inv_d02).ravel()
            for a in range(na):
                w = wa[a] + wb[b] + wc[c]
                T[a, b, c] += H_of(w, za[a] * zb[b] * zc[c]) @ KR

    # ---------------------------------------------------------------
    # Term 1: -f(x1) / (d01 d12)
    #   = -sum_{mk} g(a + b)_{mk} U[b]_{mk} V[c]_{km}
    # ---------------------------------------------------------------
    VT = np.empty((nc, N * N), dtype=np.complex128)
    for c in range(nc):
        V = (Z * inv_d12[c]) @ D                                   # [k, m]
        VT[c] = V.T.ravel()

    for b in range(nb):
        U = X @ (Y * inv_d01[b])                                   # [m, k]
        for a in range(na):
            G = _boltz_exp_int(wa[a] + wb[b], za[a] * zb[b], E, eb, beta) * U
            T[a, b, :] -= VT @ G.ravel()

    return T


@numba.njit(parallel=True, fastmath=True, cache=True)
def _resonant_correction(dE_mn, dE_mk, dE_ml, C, wa, wb, wc, triples, beta, near):
    """
    For frequency triples where a bosonic combination (wb, wb + wc or wc)
    vanishes, paths with a matching (near-)degeneracy have a zero or small
    pole denominator. _ordered_term_factorized drops those pole terms; here
    the divided difference is restored path by path:

        correction = C * (div_diff_2 - [pole terms that were kept])

    The drop thresholds must match _drop_tol: `near` for a denominator whose
    bosonic frequency is zero, 1e-8 otherwise.
    """
    n_t = triples.shape[0]
    n_p = len(dE_mn)
    out = np.zeros(n_t, dtype=np.complex128)

    for t in numba.prange(n_t):
        a = wa[triples[t, 0]]
        b = wb[triples[t, 1]]
        c = wc[triples[t, 2]]

        t01 = near if abs(b) <= 1e-8 else 1e-8
        t02 = near if abs(b + c) <= 1e-8 else 1e-8
        t12 = near if abs(c) <= 1e-8 else 1e-8

        total = 0.0j
        for p in range(n_p):
            x0 = a + dE_mn[p]
            x1 = a + b + dE_mk[p]
            x2 = a + b + c + dE_ml[p]
            d01 = x0 - x1
            d02 = x0 - x2
            d12 = x1 - x2
            r01 = abs(d01) > t01
            r02 = abs(d02) > t02
            r12 = abs(d12) > t12
            if r01 and r02 and r12:
                continue

            kept = 0.0j
            if r01 and r02:
                kept += _exp_int(x0, beta) / (d01 * d02)
            if r01 and r12:
                kept -= _exp_int(x1, beta) / (d01 * d12)
            if r02 and r12:
                kept += _exp_int(x2, beta) / (d02 * d12)

            total += C[p] * (_div_diff_2(x0, x1, x2, beta) - kept)

        out[t] = total

    return out


@numba.njit(parallel=True, fastmath=True, cache=True)
def _four_point_kernel_fast(
    dE_mn, dE_mk, dE_ml,
    C1, C2, C3, C4, C5, C6,
    iw1, iw2, iw3, beta
):
    n_iw1 = len(iw1)
    n_iw2 = len(iw2)
    n_iw3 = len(iw3)
    num_valid = len(dE_mn)

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

        # We now ONLY loop over the distinct non-zero energy triples
        for idx in range(num_valid):
            Emn = dE_mn[idx]
            Emk = dE_mk[idx]
            Eml = dE_ml[idx]

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





    @staticmethod
    def _factorized(E, beta, mats, exchange_signs, iws, dE_mn, dE_mk, dE_ml, C):
        """
        Sum the six time orderings with _ordered_term_factorized, plus the
        resonant corrections. Returns data[i, j, s] (before 1/Z).
        """
        ex0, ex1, ex2 = exchange_signs
        M0, M1, M2, M3 = mats

        # exp(beta * iw) = +1 (Boson) / -1 (Fermion)
        zetas = [np.round(np.exp(beta * w).real) for w in iws]

        # (sign, (X, Y, Z), frequency order) for each ordering, matching C1..C6
        orderings = [
            (1.0,             (M0, M1, M2), (0, 1, 2)),   # tau1 > tau2 > tau3
            (ex2,             (M0, M2, M1), (0, 2, 1)),   # tau1 > tau3 > tau2
            (ex0,             (M1, M0, M2), (1, 0, 2)),   # tau2 > tau1 > tau3
            (ex0 * ex1,       (M1, M2, M0), (1, 2, 0)),   # tau2 > tau3 > tau1
            (ex1 * ex2,       (M2, M0, M1), (2, 0, 1)),   # tau3 > tau1 > tau2
            (ex0 * ex1 * ex2, (M2, M1, M0), (2, 1, 0)),   # tau3 > tau2 > tau1
        ]

        # (Near-)degeneracies that make d01, d02, d12 small at a zero bosonic
        # frequency (a looser tolerance only makes the path sets larger).
        deg01 = np.abs(dE_mn - dE_mk) <= 2 * _NEAR
        deg02 = np.abs(dE_mn - dE_ml) <= 2 * _NEAR
        deg12 = np.abs(dE_mk - dE_ml) <= 2 * _NEAR

        data = np.zeros(tuple(len(w) for w in iws), dtype=np.complex128)

        for p, (sign, (X, Y, Z), order) in enumerate(orderings):
            wa, wb, wc = (iws[o] for o in order)
            za, zb, zc = (zetas[o] for o in order)

            T = _ordered_term_factorized(E, beta, X, Y, Z, M3, wa, wb, wc, za, zb, zc)

            # Frequency triples where a pole denominator can vanish
            A, B, Cc = np.meshgrid(
                np.arange(len(wa)), np.arange(len(wb)), np.arange(len(wc)), indexing="ij"
            )
            z01 = np.abs(wb[B]) <= _TOL
            z02 = np.abs(wb[B] + wc[Cc]) <= _TOL
            z12 = np.abs(wc[Cc]) <= _TOL

            # Group resonant triples by which denominators can vanish, and
            # only visit paths of this ordering with the matching degeneracy.
            for f01, f02, f12 in product((False, True), repeat=3):
                if not (f01 or f02 or f12):
                    continue
                res = (z01 == f01) & (z02 == f02) & (z12 == f12)
                if not res.any():
                    continue
                paths = (C[p] != 0) & ((f01 & deg01) | (f02 & deg02) | (f12 & deg12))
                if not paths.any():
                    continue

                triples = np.stack([A[res], B[res], Cc[res]], axis=1).astype(np.int64)
                # C[p] already contains the exchange sign; T does not.
                T[res] += sign * _resonant_correction(
                    *(np.ascontiguousarray(d[paths]) for d in (dE_mn, dE_mk, dE_ml)),
                    np.ascontiguousarray(C[p][paths]),
                    wa, wb, wc, triples, beta, _NEAR,
                )

            # T axes are (order[0], order[1], order[2]) -> back to (0, 1, 2)
            data -= sign * T.transpose(np.argsort(order))

        return data


    def G_div_diff_iw(self, beta, n_iw, method="direct"):

        """
        Compute the imaginary-frequency four-point function.

        Parameters
        ----------
        beta : float
            Inverse temperature.

        n_iw : list
            [n1_iw, n2_iw, n2_iw]

        method : str
            "direct" (default): sum over all non-zero (m, n, k, l) paths for
            every frequency triple.
            "factorized": pole terms of the divided difference are summed
            with matrix products, cost ~ N^3 per frequency. Agrees with
            "direct" to ~1e-13 for fermionic G2 on the Hubbard dimer and
            plaquette, ~1e-10 for density correlators with nearly degenerate
            levels at zero bosonic frequency (validate_factorized.py).

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

        # print("start numerical calculations")

        # Numba works best when lists are converted to typed numpy arrays
        exchange_signs_arr = np.array([self.sgns[0][0][1], self.sgns[0][0][2], self.sgns[0][1][2]], dtype=np.float64)

        M0, M1, M2, M3 = (np.ascontiguousarray(M) for M in mats)
        ptr012, cols012 = _nonzero_pattern([M0, M1, M2])
        ptr3T, cols3T = _nonzero_pattern([M3.T])

        # 1. Precalculate frequency-independent weights and locate sparse matrices
        valid_m, valid_n, valid_k, valid_l, *C = _precompute_coefficients(
            E,
            M0, M1, M2, M3,
            boltz,
            exchange_signs_arr,
            ptr012, cols012, ptr3T, cols3T,
        )

        dE_mn = E[valid_m] - E[valid_n]
        dE_mk = E[valid_m] - E[valid_k]
        dE_ml = E[valid_m] - E[valid_l]

        if method == "direct":
            # 2. Compute the 4-point kernel path by path (slow reference)
            data = _four_point_kernel_fast(
                dE_mn, dE_mk, dE_ml,
                *C,
                iw1,
                iw2,
                iw3,
                beta
            )

            data = data.reshape(
                (len(iw1), len(iw2), len(iw3))
            )

        else:
            data = self._factorized(
                E, beta, [M0, M1, M2, M3], exchange_signs_arr,
                [iw1, iw2, iw3],
                dE_mn, dE_mk, dE_ml, C,
            )

        # ---------------------------------------------------------------
        # Normalize by partition function.
        # ---------------------------------------------------------------

        data /= Z

        return Gf(
            mesh=iw_prod_mesh,
            data=data,
        )
