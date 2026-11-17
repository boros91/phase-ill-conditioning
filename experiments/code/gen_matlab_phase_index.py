"""Reads the output of matlab/run_phase_index.m (results/matlab/phase_index_stats.csv,
phase_index_traj.csv) and computes, for every run, the exact delay at the event, the phase
condition number with MATLAB's error scale s = max(|x|, AbsTol/RelTol), eta = |D|/Theta(T_e)
and c = eta/(RelTol*Pi).  Morris-Lecar: delays of the 1st and 10th spike.
Outputs: ../results/matlab_phase_index.csv, ../results/matlab_phase_index_summary.txt
Usage: python gen_matlab_phase_index.py [stats.csv traj.csv]
"""
import sys
import numpy as np
import pandas as pd
import phase_common as pc
import sir_common as sc
import ml_common as ml

fs = sys.argv[1] if len(sys.argv) > 2 else '../results/matlab/phase_index_stats.csv'
ft = sys.argv[2] if len(sys.argv) > 2 else '../results/matlab/phase_index_traj.csv'
st = pd.read_csv(fs)
tr = pd.read_csv(ft)
ML_X0 = {1e-1: np.array([9.9999999999707612, 0.06124350123828786]),
         1e-2: np.array([10.000000000021291, 0.061324963909461364])}


def scalar_problem(name, par):
    return pc.ghost(par) if name == 'ghost' else pc.logistic(par)


def pi_scalar(p, ratio):
    th = np.maximum(np.abs(p.grid), ratio) / np.abs(p.f(p.grid))
    the = max(abs(p.xe), ratio) / abs(p.f(p.xe))
    return th.max() / the, the


cache = {}
rows = []
for r in st.itertuples():
    x = tr[tr.run == r.run].sort_values('k')
    ratio = r.AbsTol / r.RelTol
    out = dict(run=r.run, problem=r.problem, param=r.param, solver=r.solver, RelTol=r.RelTol,
               AbsTol=r.AbsTol, nsteps=r.nsteps, event=r.event, D=np.nan, D10=np.nan)
    if r.problem in ('ghost', 'logistic'):
        p = scalar_problem(r.problem, r.param)
        Pi, the = pi_scalar(p, ratio)
        k = np.where(x.y1.values >= p.xe)[0]
        if len(k):
            k = k[0]; out['D'] = x.t.values[k] - p.T(x.y1.values[k])
    elif r.problem == 'sir':
        key = ('sir', r.param)
        if key not in cache:
            cache[key] = sc.adjoint_on_grid(r.param)
        A = cache[key]
        Pi, the = sc.Pi(A, ratio, conv='matlab')
        S = x.y1.values
        k = np.where(S <= sc.Se)[0]
        if len(k):
            k = k[0]; out['D'] = x.t.values[k] - A['T'](S[k])
    elif r.problem == 'ml':
        key = ('ml', r.param)
        if key not in cache:
            cache[key] = ml.reference(r.param, ML_X0[r.param])
        R = cache[key]
        Pi, the = ml.Pi_from(R, ratio, conv='matlab')
        V, t = x.y1.values, x.t.values
        D = []; below = False; n = 0
        for i in range(1, len(V)):
            if V[i] < 9.0:
                below = True
            if below and V[i] >= 10.0:
                below = False; n += 1
                if n > len(R['tsp']):
                    break
                D.append(t[i] - R['tref_level'](n, V[i]))
        if D:
            out['D'] = D[0]
        if len(D) >= 10:
            out['D10'] = D[9]
    out.update(Pi=Pi, Theta_e=the, tolPi=r.RelTol * Pi, eta=abs(out['D']) / the)
    out['c'] = out['eta'] / out['tolPi']
    rows.append(out)

d = pd.DataFrame(rows)
d.to_csv('../results/matlab_phase_index.csv', index=False)
lines = [f'{len(d)} runs, {int(np.isfinite(d.eta).sum())} with an event']
small = d[(d.tolPi < 1e-3) & np.isfinite(d.eta)]; big = d[(d.tolPi >= 1) & np.isfinite(d.eta)]
lines.append(f'tol*Pi < 1e-3: {len(small)} runs, max eta {small.eta.max():.3g}; '
             f'tol*Pi >= 1: {len(big)} runs, median eta {big.eta.median():.3g}')
noev = d[~np.isfinite(d.eta)]
lines.append('without event: ' + '; '.join(f'{a.problem} {a.param:.0e} {a.solver} {a.RelTol:.0e}' for a in noev.itertuples()))
g = d[(d.tolPi >= 1e-3) & np.isfinite(d.c)].groupby('solver').c
lines.append(pd.DataFrame(dict(n=g.size(), median=g.median(), q10=g.quantile(.1), q90=g.quantile(.9)))
             .sort_values('median', ascending=False).to_string(float_format=lambda v: f'{v:.3g}'))
m = d[d.problem == 'ml']
if len(m):
    lines.append(m[['param', 'solver', 'RelTol', 'Pi', 'D', 'D10', 'eta']].to_string(float_format=lambda v: f'{v:.3g}'))
txt = '\n\n'.join(lines)
open('../results/matlab_phase_index_summary.txt', 'w').write(txt)
print(txt)
