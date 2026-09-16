from triqs.operators.util.hamiltonians import *
from triqs.operators.util import *
from triqs.operators import *
from triqs.gf import *
from triqs.atom_diag import *

from utils import *

import numpy as np

try:
    import numba
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


# ---------------------------------------------------------------------------
# Fast numerical kernels
# ---------------------------------------------------------------------------

if _HAS_NUMBA:

    @numba.njit(inline="always")
    def _exp_int(x, beta):
        """
        Integral:
            int_0^beta exp(x*t) dt

        Equivalent to utils.exp_int(), but Numba compatible.
        """
        if abs(x) > 1e-8:
            return np.expm1(x * beta) / x
            # return (np.exp(x * beta)-1) / x
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
            # return beta * eb / x - (np.exp(x * beta)-1) / (x * x)
        else:
            return beta * beta / 2.0


    @numba.njit(inline="always")
    def _div_diff_1(x0, x1, beta):
        """
        First divided difference of exp_int.

        This is equivalent to:

            div_diff_exp_int_1ord([x0, x1], beta)

        from utils.py.
        """
        dx = x0 - x1

        if abs(dx) > 1e-8:
            return (
                _exp_int(x0, beta) - _exp_int(x1, beta)
            ) / dx
        else:
            return _exp_int_d1(x0, beta)


    @numba.njit(parallel=True, fastmath=False)
    def _three_point_kernel(
        E, M0, M1, M2, iw1, iw2, boltz, beta, exchange_sign,
    ):
        """Same formula and frequency ordering, with O(N) scratch per worker.

        Cache exp_int(w1+w2+E[m]-E[k]) once per (q,m,k) and
        exp_int(w{1,2}+E[m]-E[n]) once per (q,m,n), rather than
        recomputing all three inside every state triple. Exact zeros
        are skipped; no tolerance-based truncation or fastmath is used.
        """
        N = len(E)
        n_iw2 = len(iw2)

        # Sorted nonzero row indices of each M2 column. Rebuilt per call
        # so edits to self.mats continue to take effect.
        offsets = np.zeros(N + 1, dtype=np.int64)
        for m in range(N):
            count = 0
            for k in range(N):
                if M2[k, m] != 0:
                    count += 1
            offsets[m + 1] = offsets[m] + count
        indices = np.empty(offsets[N], dtype=np.int64)
        for m in range(N):
            pos = offsets[m]
            for k in range(N):
                if M2[k, m] != 0:
                    indices[pos] = k
                    pos += 1

        out = np.empty(len(iw1) * n_iw2, dtype=np.complex128)
        for q in numba.prange(len(out)):
            i = q // n_iw2
            j = q - i * n_iw2
            w1, w2 = iw1[i], iw2[j]
            xsum = np.empty(N, dtype=np.complex128)
            fsum = np.empty(N, dtype=np.complex128)
            total = 0.0j
            for m in range(N):
                start, stop = offsets[m], offsets[m + 1]
                if start == stop:
                    continue
                Em, bm = E[m], boltz[m]
                for pos in range(start, stop):
                    k = indices[pos]
                    xsum[k] = w1 + w2 + (Em - E[k])
                    fsum[k] = _exp_int(xsum[k], beta)

                for n in range(N):
                    a0 = bm * M0[m, n]
                    a1 = bm * M1[m, n]
                    if a0 == 0 and a1 == 0:
                        continue
                    Emn = Em - E[n]
                    x0 = w1 + Emn
                    y0 = w2 + Emn
                    f0 = _exp_int(x0, beta) if a0 != 0 else 0.0j
                    g0 = _exp_int(y0, beta) if a1 != 0 else 0.0j
                    have_d0 = False
                    have_d1 = False
                    d0 = 0.0j
                    d1 = 0.0j
                    for pos in range(start, stop):
                        k = indices[pos]
                        tail = M2[k, m]
                        if a0 != 0 and M1[n, k] != 0:
                            dx = x0 - xsum[k]
                            if abs(dx) > 1e-8:
                                I = (f0 - fsum[k]) / dx
                            else:
                                if not have_d0:
                                    d0 = _exp_int_d1(x0, beta)
                                    have_d0 = True
                                I = d0
                            total += a0 * M1[n, k] * tail * I
                        if a1 != 0 and M0[n, k] != 0:
                            dx = y0 - xsum[k]
                            if abs(dx) > 1e-8:
                                I = (g0 - fsum[k]) / dx
                            else:
                                if not have_d1:
                                    d1 = _exp_int_d1(y0, beta)
                                    have_d1 = True
                                I = d1
                            total += exchange_sign * a1 * M0[n, k] * tail * I
            out[q] = -total
        return out


# ---------------------------------------------------------------------------
# Three point correlation class
# ---------------------------------------------------------------------------

class ThreePtCorr:

    """
    Compute various three-point functions.

    Parameters
    ----------
    ops : list
        List of pairs (operator, statistic).

        operator is a TRIQS operator object.

        statistic is 'Boson' or 'Fermion'.

    hamiltonian : tuple
        (ad, eigensys)

        ad is the AtomDiag object.
        eigensys is the assorted spectrum:
        (state, energy, state number)

    max_states : int
        Upper cutoff on the number of states.
    """

    def __init__(self, ops, hamiltonian, max_states=10000):

        self.ops = ops
        self.hamiltonian = hamiltonian
        self.max_states = max_states

        ad = hamiltonian[0]
        eigensys = hamiltonian[1]

        hilb_dim = ad.full_hilbert_space_dim

        # ---------------------------------------------------------------
        # Only retain the states that are actually used.
        # ---------------------------------------------------------------

        self.n_states = min(max_states, hilb_dim, len(eigensys))

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
        #
        # OLD:
        #
        #   for m:
        #       for n:
        #           act(op, |n>)
        #
        # NEW:
        #
        #   for n:
        #       v = act(op, |n>)
        #       matrix[:, n] = v[state_indices]
        #
        # This removes an entire factor of N from initialization.
        # ---------------------------------------------------------------

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

        for m in range(3):

            sgn_row = []
            cml_sgn = 1

            for n in range(3):

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

        # ---------------------------------------------------------------
        # Warn once if Numba isn't available.
        # ---------------------------------------------------------------

        if not _HAS_NUMBA:
            print(
                "WARNING: numba is not installed. "
                "G_div_diff_iw requires numba."
            )


    def G_div_diff_iw(self, beta, n_iw):

        """
        Compute the imaginary-frequency three-point function.

        Parameters
        ----------
        beta : float
            Inverse temperature.

        n_iw : list
            [n1_iw, n2_iw]

        Returns
        -------
        Gf
            TRIQS two-frequency Green's function.
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

        for n in range(2):

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

        # ---------------------------------------------------------------
        # Boltzmann weights.
        #
        # OLD:
        #
        #     exp(-beta * E[m])
        #
        # was evaluated inside the innermost k loop.
        #
        # NEW:
        #     calculated exactly once per state.
        # ---------------------------------------------------------------

        boltz = np.exp(-beta * E)

        # ---------------------------------------------------------------
        # Main numerical calculation
        # ---------------------------------------------------------------

        if _HAS_NUMBA:

            data = _three_point_kernel(
                E,
                np.ascontiguousarray(mats[0]),
                np.ascontiguousarray(mats[1]),
                np.ascontiguousarray(mats[2]),
                iw1,
                iw2,
                boltz,
                beta,
                self.sgns[0][0][1],
            )

            data = data.reshape(
                (len(iw1), len(iw2))
            )

        else:
            raise ImportError("ThreePtCorr requires numba")

        # ---------------------------------------------------------------
        # Normalize by partition function.
        #
        # Numba kernel deliberately leaves Z out so that it doesn't
        # perform an unnecessary division inside the triple loop.
        # ---------------------------------------------------------------

        data /= Z

        return Gf(
            mesh=iw_prod_mesh,
            data=data,
        )

