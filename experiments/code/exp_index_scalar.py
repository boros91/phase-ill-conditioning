"""Phase condition number Pi versus the measured phase delay, scalar problems.

For each problem, SciPy method and tolerance (rtol = tol, atol = tol/100)
the exact delay D of the first accepted step with x >= xe is computed from
the exact time function, and compared with tol * Pi.
Output: ../results/index_scalar.csv, ../results/index_scalar_cases.csv
"""
import csv
import numpy as np
import phase_common as pc

RATIO = 1e-2   # atol/rtol, as in the benchmark setting
cases = ([pc.flame_q(2, d) for d in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)]
         + [pc.flame_q(3, d) for d in (1e-1, 1e-2, 1e-3)]
         + [pc.ghost(e) for e in (1e-2, 1e-4, 1e-6, 1e-8)]
         + [pc.logistic(x) for x in (1e-1, 1e-3, 1e-5, 1e-7)])
tols = (1e-4, 1e-6, 1e-8)

rows, crows = [], []
for p in cases:
    Pi, xstar = p.Pi(RATIO)
    Pi_rel, _ = p.Pi(0.0)
    the = p.Theta(np.array([p.xe]), RATIO)[0]
    crows.append(dict(case=p.name, x0=p.x0, xe=p.xe, Theta_e=the, Pi=Pi, x_argmax=xstar,
                      Pi_relative=Pi_rel, T_event=p.T(p.xe)))
    for m in pc.METHODS:
        for tol in tols:
            ts, xs, nfev, n, ok = pc.run_to_level(p, m, tol, tol * RATIO)
            D = ts[-1] - p.T(xs[-1]) if ok else np.nan
            rows.append(dict(case=p.name, method=m, tol=tol, Pi=Pi, tolPi=tol * Pi,
                             Theta_e=the, D=D, eta=abs(D) / the if ok else np.nan,
                             steps=n, nfev=nfev, ok=ok))
            print(f"{p.name:22s} {m:6s} {tol:7.0e}  Pi={Pi:9.3e}  D={D:+11.3e}  "
                  f"eta={abs(D)/the:9.3e}  tolPi={tol*Pi:9.3e}  steps={n}")

with open('../results/index_scalar.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
with open('../results/index_scalar_cases.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(crows[0]))
    w.writeheader(); w.writerows(crows)
