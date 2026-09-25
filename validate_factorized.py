"""
Compare FourPtCorr.G_div_diff_iw(method="factorized") against method="direct".

    python validate_factorized.py dimer [n_iw]
    python validate_factorized.py plaquette [n_iw]
"""
import sys, time
from itertools import product
import numpy as np
from triqs.atom_diag import AtomDiagComplex
from triqs.operators import n, c, c_dag
from sort import sort_states_full
from four_pt_corr_v2 import FourPtCorr

system = sys.argv[1] if len(sys.argv) > 1 else "dimer"
n_iw = int(sys.argv[2]) if len(sys.argv) > 2 else 3
beta = 2.0
t, U, h_field = 0.5, 4.0, 0.05
spin_names = ("up", "dn")

if system == "dimer":
    eps = [-1.9, -2.1]
    bonds = [(0, 1)]
else:
    eps = [-1.9, -2.1, -1.9, -2.1]
    bonds = [(0, 1), (1, 3), (3, 2), (2, 0)]
atoms = tuple(range(len(eps)))
fops = list(product(spin_names, atoms))

H = sum(e * n(s, a) for (a, e), s in product(zip(atoms, eps), spin_names))
H += sum(-h_field * (n('up', a) - n('dn', a)) for a in atoms)
H += U * sum(n('up', a) * n('dn', a) for a in atoms)
H += t * sum(c_dag(sp, i) * c(sp, j) + c_dag(sp, j) * c(sp, i)
             for sp in spin_names for i, j in bonds)

ad = AtomDiagComplex(H, fops)
eigensys = sort_states_full(spin_names, atoms, ad)

cases = {
    "G2 up,dn":  [(c_dag('up', 0), 'Fermion'), (c('up', 1), 'Fermion'),
                  (c_dag('dn', 0), 'Fermion'), (c('dn', 1), 'Fermion')],
    "G2 up,up":  [(c_dag('up', 0), 'Fermion'), (c('up', 0), 'Fermion'),
                  (c_dag('up', 1), 'Fermion'), (c('up', 1), 'Fermion')],
    "density":   [(n('up', 0), 'Boson'), (n('dn', 0), 'Boson'),
                  (n('up', 1), 'Boson'), (n('dn', 1), 'Boson')],
    "mixed":     [(c_dag('up', 0), 'Fermion'), (c('up', 1), 'Fermion'),
                  (n('dn', 0), 'Boson'), (n('up', 1), 'Boson')],
}

worst = 0.0
for name, ops in cases.items():
    corr = FourPtCorr(ops=ops, hamiltonian=(ad, eigensys))
    t0 = time.time()
    gf = corr.G_div_diff_iw(beta=beta, n_iw=[n_iw] * 3, method="factorized").data
    t1 = time.time()
    gd = corr.G_div_diff_iw(beta=beta, n_iw=[n_iw] * 3, method="direct").data
    t2 = time.time()
    err = np.abs(gf - gd).max() / max(np.abs(gd).max(), 1e-300)
    worst = max(worst, err)
    print(f"{system:9s} {name:9s} n_iw={n_iw}: factorized {t1-t0:7.2f}s  direct {t2-t1:7.2f}s  "
          f"max|G| {np.abs(gd).max():.3e}  rel. max diff {err:.2e}", flush=True)

print("PASS" if worst < 1e-8 else "FAIL", f"(worst rel. diff {worst:.2e})")
