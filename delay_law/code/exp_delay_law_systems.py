"""Out-of-sample test of the integrated delay law on systems (SIR, Morris-Lecar).

For a system the scalar quantities generalise as
    Theta(t) = sum_i s_i |lambda_i|                       (timing sensitivity, as before)
    omega(t) = max_i |(J f)_i|/s_i / max_i |f_i|/s_i       (local rate: |x''|/|x'| in the error scale;
                                                              = |f'| for a scalar equation)
    sigma(t) = Theta(t) * omega(t),   Lambda_alpha = int_0^Te sigma^alpha dt / Theta(T_e).
The exponent alpha and the constant K of each solver are taken from the fit on the scalar problems
(exp_delay_law_fit.py, model M2); for M0 (eta = c tol Pi) the constant c is the scalar fit as well.
Nothing is fitted to the systems.

Inputs : ../results/delay_law_fit.csv, ../results/delay_law_fit_runs.csv
         ../../experiments/results/sir_runs.csv, ml_runs.csv, matlab_phase_index.csv, matlab_pi_sir.csv
Outputs: ../results/delay_law_systems_runs.csv, ../results/delay_law_systems.txt
"""
import sys
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
sys.path.insert(0, '.')
import sir_common as sc
import ml_common as ml

OLD = '../../experiments/results/'
fit = pd.read_csv('../results/delay_law_fit.csv')
fr = pd.read_csv('../results/delay_law_fit_runs.csv')


def scale(X, ratio, conv):
    return np.abs(X) + ratio if conv == 'scipy' else np.maximum(np.abs(X), ratio)


def lam_ind(tg, X, LAM, F, JF, ratio, conv, alpha):
    S = scale(X, ratio, conv)
    th = (np.abs(LAM) * S).sum(0)
    om = (np.abs(JF) / S).max(0) / (np.abs(F) / S).max(0)
    sig = th * om
    return np.trapezoid(sig**alpha, tg) / th[-1], th.max() / th[-1], th[-1]


# ---------------- references ----------------
refs = {}
for I0 in (1e-2, 1e-4, 1e-6, 1e-7, 1e-8, 1e-9):
    A = sc.adjoint_on_grid(I0)
    F = np.array([sc.rhs(0, A['X'][:, k]) for k in range(len(A['tg']))]).T
    JF = np.array([sc.jac(0, A['X'][:, k]) @ F[:, k] for k in range(len(A['tg']))]).T
    refs[('sir', I0)] = (A['tg'], A['X'], A['LAM'], F, JF)
    print('SIR reference', I0, flush=True)

Ic, _ = ml.snic_current()
for Delta in (1e-1, 1e-2, 1e-3):
    f, J = ml.make(Ic + Delta)
    pre = solve_ivp(f, [0, 40000], [-30.0, 0.1], method='DOP853', rtol=1e-13, atol=1e-15, dense_output=True)
    V = pre.y[0]; up = np.where((V[:-1] < 10.0) & (V[1:] >= 10.0))[0]; k = up[-2]
    ts = brentq(lambda t: pre.sol(t)[0] - 10.0, pre.t[k], pre.t[k + 1], xtol=1e-14)
    R = ml.reference(Delta, pre.sol(ts), nsp=1)
    F = np.array([f(0, R['X'][:, k]) for k in range(len(R['tg']))]).T
    JF = np.array([J(0, R['X'][:, k]) @ F[:, k] for k in range(len(R['tg']))]).T
    refs[('ml', Delta)] = (R['tg'], R['X'], R['LAM'], F, JF)
    print('ML reference', Delta, flush=True)

# ---------------- runs ----------------
rows = []
d = pd.read_csv(OLD + 'sir_runs.csv')
for _, r in d.iterrows():
    rows.append(dict(env='SciPy', solver=r.method, problem='sir', par=r.I0, tol=r.tol, ratio=1e-2, conv='scipy',
                     eta=abs(r.eta) if r.ok and np.isfinite(r.eta) and r.minI > 0 else np.nan))
d = pd.read_csv(OLD + 'ml_runs.csv'); d = d[d.variant == 'standard']
for _, r in d.iterrows():
    rows.append(dict(env='SciPy', solver=r.method, problem='ml', par=r.Delta, tol=r.tol, ratio=1e-2, conv='scipy',
                     eta=abs(r.eta1)))
d = pd.read_csv(OLD + 'matlab_phase_index.csv'); d = d[d.problem.isin(['sir', 'ml'])]
for _, r in d.iterrows():
    rows.append(dict(env='MATLAB', solver=r.solver, problem=r.problem, par=r.param, tol=r.RelTol,
                     ratio=r.AbsTol / r.RelTol, conv='matlab', eta=abs(r.eta) if r.event == 1 else np.nan))
d = pd.read_csv(OLD + 'matlab_pi_sir.csv'); d = d[d.variant == 'std']
for _, r in d.iterrows():
    rows.append(dict(env='MATLAB', solver=r.solver, problem='sir', par=r.I0, tol=r.RelTol,
                     ratio=r.AbsTol / r.RelTol, conv='matlab', eta=abs(r.eta)))
Rn = pd.DataFrame(rows)
Rn = Rn[np.isfinite(Rn.eta) & (Rn.eta > 1e-10)].reset_index(drop=True)

out = []
for i, r in Rn.iterrows():
    fs = fit[(fit.env == r.env) & (fit.solver == r.solver)].iloc[0]
    alpha, K = fs.alpha_M2, fs.K_M2
    # constant c of model M0 from the scalar runs: geometric mean of eta/(tol Pi)
    g = fr[(fr.env == r.env) & (fr.solver == r.solver)]
    c = 10**np.mean(np.log10(g.eta / (g.tol * g.Pi)))
    tg, X, LAM, F, JF = refs[(r.problem, r.par)]
    lam, Pi, the = lam_ind(tg, X, LAM, F, JF, r.ratio, r.conv, alpha)
    out.append(dict(r, alpha=alpha, Lambda=lam, Pi=Pi, Theta_e=the,
                    pred_M2=K * r.tol**alpha * lam, pred_M0=c * r.tol * Pi))
O = pd.DataFrame(out)
O['res_M0'] = np.log10(O.eta / O.pred_M0)
O['res_M2'] = np.log10(O.eta / O.pred_M2)
O.to_csv('../results/delay_law_systems_runs.csv', index=False)
txt = ['Systems, out of sample (alpha, K, c from the scalar fit). log10(eta/prediction):', '']
for (env, prob), G in O.groupby(['env', 'problem']):
    txt.append(f'{env:6s} {prob:4s} n={len(G):3d}  M0: mean {G.res_M0.mean():+.2f} sd {G.res_M0.std():.2f} '
               f'within 1 decade {np.mean(np.abs(G.res_M0) < 1):.2f}   '
               f'M2: mean {G.res_M2.mean():+.2f} sd {G.res_M2.std():.2f} within 1 decade {np.mean(np.abs(G.res_M2) < 1):.2f}')
txt.append('')
txt.append(f'all n={len(O)}: M0 median |res| {O.res_M0.abs().median():.2f}, within 1 decade {np.mean(O.res_M0.abs()<1):.2f}; '
           f'M2 median |res| {O.res_M2.abs().median():.2f}, within 1 decade {np.mean(O.res_M2.abs()<1):.2f}')
open('../results/delay_law_systems.txt', 'w').write('\n'.join(txt) + '\n')
print('\n'.join(txt))
