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
        E,
        M0,
        M1,
        M2,
        iw1,
        iw2,
        boltz,
        beta,
        exchange_sign,
    ):
        """
        Compute the complete three-point frequency grid.

        This replaces the Python loops over:

            iw1
            iw2
            m
            n
            k

        The output is flattened as (iw1, iw2).
        """

        n_iw1 = len(iw1)
        n_iw2 = len(iw2)
        N = len(E)

        out = np.zeros(n_iw1 * n_iw2, dtype=np.complex128)

        # Parallelize over frequency points.
        #
        # Each frequency point is independent, which avoids race
        # conditions while still exposing a large amount of parallelism.
        for q in numba.prange(n_iw1 * n_iw2):

            i = q // n_iw2
            j = q - i * n_iw2

            w1 = iw1[i]
            w2 = iw2[j]

            total = 0.0j

            for m in range(N):

                Em = E[m]
                bm = boltz[m]

                for n in range(N):

                    Emn = Em - E[n]

                    # Matrix elements independent of k.
                    a0 = bm * M0[m, n]
                    a1 = bm * M1[m, n]

                    for k in range(N):

                        Emk = Em - E[k]

                        # --------------------------------------------------
                        # tau1 > tau2
                        # --------------------------------------------------

                        x0 = w1 + Emn
                        x1 = w1 + w2 + Emk

                        I = _div_diff_1(x0, x1, beta)

                        total += (
                            a0
                            * M1[n, k]
                            * M2[k, m]
                            * I
                        )

                        # --------------------------------------------------
                        # tau2 > tau1
                        # --------------------------------------------------

                        x0 = w2 + Emn
                        x1 = w1 + w2 + Emk

                        I = _div_diff_1(x0, x1, beta)

                        total += (
                            exchange_sign
                            * a1
                            * M0[n, k]
                            * M2[k, m]
                            * I
                        )

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
                "G_div_diff_iw will use the slower Python implementation."
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

            # # -----------------------------------------------------------
            # # Fallback implementation.
            # #
            # # Still considerably faster than the original because:
            # #   - everything is NumPy arrays
            # #   - exp(-beta E) is precomputed
            # #   - matrix indexing is local
            # # -----------------------------------------------------------

            # M0 = mats[0]
            # M1 = mats[1]
            # M2 = mats[2]

            # data = np.zeros(
            #     (len(iw1), len(iw2)),
            #     dtype=np.complex128,
            # )

            # exchange_sign = self.sgns[0][0][1]

            # for ii, w1 in enumerate(iw1):

            #     for jj, w2 in enumerate(iw2):

            #         total = 0.0j

            #         for m in range(N):

            #             Em = E[m]
            #             bm = boltz[m]

            #             for n in range(N):

            #                 Emn = Em - E[n]

            #                 a0 = bm * M0[m, n]
            #                 a1 = bm * M1[m, n]

            #                 for k in range(N):

            #                     Emk = Em - E[k]

            #                     I = div_diff_exp_int_1ord(
            #                         [
            #                             w1 + Emn,
            #                             w1 + w2 + Emk,
            #                         ],
            #                         beta=beta,
            #                     )

            #                     total += (
            #                         a0
            #                         * M1[n, k]
            #                         * M2[k, m]
            #                         * I
            #                     )

            #                     I = div_diff_exp_int_1ord(
            #                         [
            #                             w2 + Emn,
            #                             w1 + w2 + Emk,
            #                         ],
            #                         beta=beta,
            #                     )

            #                     total += (
            #                         exchange_sign
            #                         * a1
            #                         * M0[n, k]
            #                         * M2[k, m]
            #                         * I
            #                     )
            

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
