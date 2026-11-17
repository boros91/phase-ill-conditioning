"""Which indicator predicts the realised delay of adaptive codes best?  Scalar problems only.

Models for eta = |D|/Theta(T_e) (Theta in the solver's error scale):
  M0  eta = c   * tol         * Pi             (phase condition number, article v2)
  M1  eta = K   * tol^alpha   * Pi             (same, exponent of tol fitted)
  M2  eta = K   * tol^alpha   * Lambda_alpha   (integrated law, sigma = s|f'|/|f|)
  M3  eta = K   * tol^alpha   * t_*/Theta_e    (phase ratio)
For each solver, alpha and K are fitted by least squares in log eta; the spread of the residuals
(standard deviation and 10-90% range of log10) measures how well the model explains the runs.

Inputs : ../../experiments/results/index_scalar.csv          (SciPy, 240 runs)
         ../../experiments/results/matlab_pi_flame.csv       (MATLAB R2023b, flame runs of Paper A)
         ../../experiments/results/matlab_phase_index.csv    (MATLAB R2023b, ghost and logistic runs)
Outputs: ../results/delay_law_fit.csv, ../results/delay_law_fit_runs.csv, ../results/delay_law_fit.txt
"""
import re
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import delay_law_common as L

OLD = '../../experiments/results/'
ALPHAS = np.round(np.arange(0.30, 1.501, 0.05), 2)
cache = {}


def problem(name, par):
    if name == 'flame2':
        f, fp = L.flame(2); return f, fp, par, 0.5, None
    if name == 'flame3':
        f, fp = L.flame(3); return f, fp, par, 0.5, None
    if name == 'ghost':
        f, fp = L.ghost(par); return f, fp, -1.0, 1.0, [0.0]
    if name == 'logistic':
        f, fp = L.logistic(); return f, fp, par, 0.5, None
    raise ValueError(name)


def indicators(name, par, ratio, conv):
    key = (name, par, round(np.log10(ratio), 6) if ratio > 0 else -99, conv)
    if key not in cache:
        f, fp, x0, xe, pts = problem(name, par)
        lam = np.array([L.Lambda(f, fp, x0, xe, a, ratio, conv, pts) for a in ALPHAS])
        cache[key] = dict(lam=lam, pr=L.phase_ratio(f, x0, xe, ratio, conv, pts))
    return cache[key]


rows = []
# ---- SciPy ----
d = pd.read_csv(OLD + 'index_scalar.csv')
for _, r in d.iterrows():
    m = re.match(r'(flame_q2|flame_q3|ghost|logistic)_[dex](.*)', r.case)
    name = {'flame_q2': 'flame2', 'flame_q3': 'flame3', 'ghost': 'ghost', 'logistic': 'logistic'}[m.group(1)]
    par = float(m.group(2))
    rows.append(dict(env='SciPy', solver=r.method, problem=name, par=par, tol=r.tol, ratio=1e-2, conv='scipy',
                     Pi=r.Pi, eta=abs(r.eta) if r.ok and np.isfinite(r.eta) else np.nan))
# ---- MATLAB, flame runs of Paper A ----
d = pd.read_csv(OLD + 'matlab_pi_flame.csv')
d = d[d.completed == 1]
for _, r in d.iterrows():
    rows.append(dict(env='MATLAB', solver=r.solver, problem=f'flame{int(r.q)}', par=r.delta, tol=r.RelTol,
                     ratio=r.AbsTol / r.RelTol, conv='matlab', Pi=r.Pi, eta=abs(r.eta)))
# ---- MATLAB, new runs: ghost and logistic ----
d = pd.read_csv(OLD + 'matlab_phase_index.csv')
d = d[d.problem.isin(['ghost', 'logistic'])]
for _, r in d.iterrows():
    rows.append(dict(env='MATLAB', solver=r.solver, problem=r.problem, par=r.param, tol=r.RelTol,
                     ratio=r.AbsTol / r.RelTol, conv='matlab', Pi=r.Pi, eta=abs(r.eta)))

R = pd.DataFrame(rows)
R = R[np.isfinite(R.eta) & (R.eta > 0)].reset_index(drop=True)
LAM = np.zeros((len(R), len(ALPHAS))); PR = np.zeros(len(R))
for i, r in R.iterrows():
    I = indicators(r.problem, r.par, r.ratio, r.conv)
    LAM[i] = I['lam']; PR[i] = I['pr']
R['phase_ratio'] = PR
# keep runs whose delay is clearly above roundoff: eta > 1e-10, and tol*Pi >= 1e-6
# (the bulk of tiny delays at tight tolerances is dominated by the event-location error)
use = (R.eta > 1e-10).values


def fit(y, logtol, logind):
    """Least squares in log10 eta = logK + alpha*logtol + logind; returns alpha, logK, residuals."""
    A = np.vstack([np.ones_like(logtol), logtol]).T
    c, *_ = np.linalg.lstsq(A, y - logind, rcond=None)
    res = y - logind - A @ c
    return c[1], c[0], res


out, runs = [], []
for (env, solver), G in R[use].groupby(['env', 'solver']):
    idx = G.index.values
    y = np.log10(G.eta.values); lt = np.log10(G.tol.values)
    # M0: alpha = 1, K free
    r0 = y - lt - np.log10(G.Pi.values); r0 = r0 - r0.mean()
    a1, k1, r1 = fit(y, lt, np.log10(G.Pi.values))
    a3, k3, r3 = fit(y, lt, np.log10(G.phase_ratio.values))
    # M2: alpha enters Lambda_alpha; scan alpha, use the same alpha for tol and Lambda
    best = None
    for j, a in enumerate(ALPHAS):
        z = y - a * lt - np.log10(LAM[idx, j]); res = z - z.mean()
        sd = res.std()
        if best is None or sd < best[0]:
            best = (sd, a, z.mean(), res, j)
    sd2, a2, k2, r2, j2 = best
    q = lambda r: (r.std(), np.percentile(r, 90) - np.percentile(r, 10))
    out.append(dict(env=env, solver=solver, n=len(G),
                    sd_M0=q(r0)[0], r90_M0=q(r0)[1],
                    alpha_M1=a1, sd_M1=q(r1)[0], r90_M1=q(r1)[1],
                    alpha_M2=a2, K_M2=10**k2, sd_M2=q(r2)[0], r90_M2=q(r2)[1],
                    alpha_M3=a3, sd_M3=q(r3)[0], r90_M3=q(r3)[1]))
    for i, rr0, rr2 in zip(idx, r0, r2):
        runs.append(dict(i=i, res_M0=rr0, res_M2=rr2, alpha=a2, Lambda=LAM[i, j2]))

O = pd.DataFrame(out)
Rr = R.join(pd.DataFrame(runs).set_index('i'))
O.to_csv('../results/delay_law_fit.csv', index=False)
Rr.to_csv('../results/delay_law_fit_runs.csv', index=False)
pd.set_option('display.width', 250)
txt = ['Spread of log10 residuals (sd, and 10-90% range) for each model, per solver', '',
       O.round(3).to_string(index=False), '',
       f'runs used: {use.sum()} of {len(R)} with an event; SciPy {(R[use].env=="SciPy").sum()}, MATLAB {(R[use].env=="MATLAB").sum()}']
for env in ('SciPy', 'MATLAB'):
    o = O[O.env == env]
    w = o.n / o.n.sum()
    txt.append(f'{env}: weighted mean sd  M0 {np.sum(w*o.sd_M0):.3f}  M1 {np.sum(w*o.sd_M1):.3f}  '
               f'M2 {np.sum(w*o.sd_M2):.3f}  M3 {np.sum(w*o.sd_M3):.3f}')
open('../results/delay_law_fit.txt', 'w').write('\n'.join(txt) + '\n')
print('\n'.join(txt))
