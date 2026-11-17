"""Leave-one-family-out validation of the one-step model (M0: eta = c tol Pi) and of the integrated law
(M2: eta = K tol^alpha Lambda_alpha) on the scalar runs of exp_delay_law_fit.py.
For every solver and every scalar family (flame q=2, flame q=3, ghost, logistic), the constants (c; K and
alpha) are fitted on the other three families and the held-out family is predicted.
Input : ../results/delay_law_fit_runs.csv (runs, Pi) and Lambda_alpha recomputed on the alpha grid
Output: ../results/delay_law_lofo.csv, ../results/delay_law_lofo.txt
"""
import numpy as np
import pandas as pd
import delay_law_common as L
from exp_delay_law_fit_lib import problem, ALPHAS

R = pd.read_csv('../results/delay_law_fit_runs.csv')
R = R[R.res_M0.notna()].reset_index(drop=True)
cache = {}
LAM = np.zeros((len(R), len(ALPHAS)))
for i, r in R.iterrows():
    key = (r.problem, r.par, round(np.log10(r.ratio), 6), r.conv)
    if key not in cache:
        f, fp, x0, xe, pts = problem(r.problem, r.par)
        cache[key] = np.array([L.Lambda(f, fp, x0, xe, a, r.ratio, r.conv, pts) for a in ALPHAS])
    LAM[i] = cache[key]

rows = []
for (env, solver), G in R.groupby(['env', 'solver']):
    for fam in G.problem.unique():
        tr = G[G.problem != fam]; te = G[G.problem == fam]
        if len(tr) < 5 or len(te) == 0:
            continue
        # M0
        logc = np.mean(np.log10(tr.eta / (tr.tol * tr.Pi)))
        r0 = np.log10(te.eta / (10**logc * te.tol * te.Pi))
        # M2
        best = None
        for j, a in enumerate(ALPHAS):
            z = np.log10(tr.eta.values) - a * np.log10(tr.tol.values) - np.log10(LAM[tr.index, j])
            if best is None or z.std() < best[0]:
                best = (z.std(), j, a, z.mean())
        _, j, a, k = best
        r2 = np.log10(te.eta.values) - a * np.log10(te.tol.values) - np.log10(LAM[te.index, j]) - k
        for v0, v2 in zip(r0, r2):
            rows.append(dict(env=env, solver=solver, held_out=fam, res_M0=v0, res_M2=v2, alpha=a))
O = pd.DataFrame(rows)
O.to_csv('../results/delay_law_lofo.csv', index=False)
txt = ['Leave-one-family-out prediction of the held-out family (log10 measured/predicted eta)', '']
for fam, G in O.groupby('held_out'):
    txt.append(f'{fam:9s} n={len(G):4d}  M0: median|res| {G.res_M0.abs().median():.2f}, within 1 decade {np.mean(G.res_M0.abs()<1):.2f}   '
               f'M2: median|res| {G.res_M2.abs().median():.2f}, within 1 decade {np.mean(G.res_M2.abs()<1):.2f}')
txt.append(f'all       n={len(O):4d}  M0: median|res| {O.res_M0.abs().median():.2f}, within 1 decade {np.mean(O.res_M0.abs()<1):.2f}   '
           f'M2: median|res| {O.res_M2.abs().median():.2f}, within 1 decade {np.mean(O.res_M2.abs()<1):.2f}')
open('../results/delay_law_lofo.txt', 'w').write('\n'.join(txt) + '\n')
print('\n'.join(txt))
