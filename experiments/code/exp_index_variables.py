"""Phase condition number of the flame problem y' = y^q (1-y) in different
unknowns u = phi(y) and error measures.  A perturbation e of u at the level y
moves the event (y = 1/2) by e/|u'|, u' = phi'(y) f(y); with the scale s_u the
timing sensitivity is Theta_u = s_u / |u'| and Pi_u = max Theta_u / Theta_u(1/2).
Output: ../results/index_variables.csv
"""
import csv
import numpy as np

rows = []
for q in (2, 3):
    for d in ((1e-2, 1e-4, 1e-6) if q == 2 else (1e-1, 1e-2, 1e-3)):
        y = np.unique(np.concatenate([np.geomspace(d, 0.5, 20001), np.linspace(d, 0.5, 20001)]))
        f = y**q * (1 - y)
        variants = {
            'y, s = |y| + 1e-2 (atol = rtol/100)': (np.abs(y) + 1e-2) / f,
            'y, s = |y| (relative only)': y / f,
            'ln y, s = 1 (absolute only)': 1.0 / (f / y),
            'y^(1-q), s = |u| (relative only)': (y**(1 - q)) / ((q - 1) * y**(-q) * f),
            'y^(1-q), s = 1 (absolute only)': 1.0 / ((q - 1) * y**(-q) * f),
        }
        for name, th in variants.items():
            Pi = th.max() / th[-1]
            rows.append(dict(q=q, delta=d, unknown=name, Pi=Pi))
            print(f"q={q} delta={d:.0e}  {name:38s} Pi = {Pi:.3e}")
with open('../results/index_variables.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
