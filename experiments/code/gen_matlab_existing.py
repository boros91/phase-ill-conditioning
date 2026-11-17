"""Phase condition number against the delays of the MATLAB R2023b runs already made for Paper A:
 - results/matlab/moler_matlab.csv   (8 solvers; q = 2, 3; RelTol/AbsTol = 1e-4/1e-6, 1e-8/1e-12, 1e-12/1e-20)
 - results/matlab/tol_sweep_matlab.csv (part A: 5 solvers; q = 2; RelTol 1e-3 ... 100u, AbsTol = RelTol/100)
 - results/matlab/other_sir.csv, other_problems_stats.csv (SIR, I0 = 1e-7, 1e-9; variants std, def, atol, log)
MATLAB error scale: s = max(|y|, AbsTol/RelTol).  eta = |D|/Theta(T_e) with D the exact delay of the
first accepted step past the event (D_half in the Paper A files).
Outputs: ../results/matlab_pi_flame.csv, ../results/matlab_pi_sir.csv, ../results/matlab_constants.csv
"""
import numpy as np
import pandas as pd
import sir_common as sc

M = '../results/matlab/'


def pi_flame(q, delta, ratio):
    y = np.unique(np.concatenate([np.geomspace(delta, 0.5, 20001), np.linspace(delta, 0.5, 20001)]))
    f = y**q * (1 - y)
    th = np.maximum(y, ratio) / f
    return th.max() / th[-1], th[-1]


rows = []
m = pd.read_csv(M + 'moler_matlab.csv'); m = m[m.warm == 0].copy(); m['source'] = 'moler'
t = pd.read_csv(M + 'tol_sweep_matlab.csv'); t = t[(t.part == 'A') & (t.warm == 0)].copy(); t['source'] = 'tol_sweep'
cols = ['source', 'q', 'delta', 'solver', 'RelTol', 'AbsTol', 'nsteps', 'D_half', 'completed']
d = pd.concat([m[cols], t[cols]], ignore_index=True)
cache = {}
for i, r in d.iterrows():
    key = (r.q, r.delta, r.AbsTol / r.RelTol)
    if key not in cache:
        cache[key] = pi_flame(int(r.q), r.delta, r.AbsTol / r.RelTol)
    Pi, the = cache[key]
    d.loc[i, 'Pi'] = Pi
    d.loc[i, 'Theta_e'] = the
d['tolPi'] = d.RelTol * d.Pi
d['eta'] = np.abs(d.D_half) / d.Theta_e
d['c'] = d.eta / d.tolPi
d.to_csv('../results/matlab_pi_flame.csv', index=False)

# SIR
st = pd.read_csv(M + 'other_problems_stats.csv'); st = st[st.example == 'sir']
tr = pd.read_csv(M + 'other_sir.csv')
srows = []
adj = {}
for r in st.itertuples():
    I0 = r.param
    if I0 not in adj:
        adj[I0] = sc.adjoint_on_grid(I0)
    A = adj[I0]
    ratio = r.AbsTol / r.RelTol
    unknown = 'log' if r.variant == 'log' else 'x'
    Pi, the = sc.Pi(A, ratio, conv='matlab', unknown=unknown)
    x = tr[tr.run == r.run]
    S = np.exp(x.y1.values) if unknown == 'log' else x.y1.values
    k = np.where(S <= sc.Se)[0]
    if len(k) == 0:
        D = np.nan
    else:
        k = k[0]
        D = x.t.values[k] - A['T'](S[k])
    srows.append(dict(I0=I0, variant=r.variant, solver=r.solver, RelTol=r.RelTol, AbsTol=r.AbsTol,
                      Pi=Pi, tolPi=r.RelTol * Pi, Theta_e=the, D=D, eta=abs(D) / the, c=abs(D) / the / (r.RelTol * Pi),
                      Te=A['Te'], nsteps=r.nsteps))
s = pd.DataFrame(srows)
s.to_csv('../results/matlab_pi_sir.csv', index=False)

# summaries
out = []
ok = d[np.isfinite(d.eta) & (d.completed == 1)]
out.append(f'flame runs used: {len(ok)} (moler {sum(ok.source=="moler")}, tol_sweep {sum(ok.source=="tol_sweep")})')
small = ok[ok.tolPi < 1e-3]; big = ok[ok.tolPi >= 1]
out.append(f'tol*Pi < 1e-3: {len(small)} runs, max eta {small.eta.max():.2e}; tol*Pi >= 1: {len(big)} runs, median eta {big.eta.median():.2f}')
cons = []
for solver, g in ok[ok.tolPi >= 1e-3].groupby('solver'):
    cons.append(dict(solver=solver, n=len(g), median=g.c.median(), q10=g.c.quantile(.1), q90=g.c.quantile(.9)))
cons = pd.DataFrame(cons).sort_values('median', ascending=False)
cons.to_csv('../results/matlab_constants.csv', index=False)
out.append(cons.to_string(float_format=lambda v: f'{v:.3g}'))
out.append(s[['I0', 'variant', 'solver', 'RelTol', 'AbsTol', 'Pi', 'tolPi', 'D', 'eta']].to_string(float_format=lambda v: f'{v:.3g}'))
txt = '\n\n'.join(out)
open('../results/matlab_pi_summary.txt', 'w').write(txt)
print(txt)
