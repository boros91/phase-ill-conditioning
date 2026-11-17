"""Systems: three definitions of the event error, the first-order representation with defects at their
physical insertion times, and the cancellation of time defects (SIR and Morris-Lecar, SciPy).

For every run:
  D_end  : phase defect at the first accepted step past the event, t_n - T_ref(c_n), where T_ref(c) is the
           time at which the reference reaches the computed value c of the event coordinate
           (S for SIR, V on the upstroke for Morris-Lecar)   [the definition used in the article]
  D_root : error of the event time located on the solver's dense output, t_root - T_e
  D_hat  : - sum_k lambda(t_{k+1}) . d_k, d_k = X_{k+1} - phi_{h_k}(X_k) the true defect of step k
           (phi from DOP853, rtol 1e-12), lambda the finite-event adjoint at the physical time t_{k+1}
  A      : sum_k |lambda(t_{k+1}) . d_k|  (no cancellation)
Morris-Lecar: first spike, adjoint over one period with the event-normal terminal condition.
Output: ../results/event_error_check.csv, ../results/event_error_check.txt
"""
import time
import warnings
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp, RK45, DOP853, LSODA, BDF, Radau
from scipy.optimize import brentq
import sir_common as sc
import ml_common as ml

warnings.simplefilter('ignore')
METHODS = {'RK45': RK45, 'DOP853': DOP853, 'LSODA': LSODA, 'BDF': BDF, 'Radau': Radau}
RATIO = 1e-2
rows = []


def lam_interp(tg, LAM, t):
    t = min(max(t, tg[0]), tg[-1])
    return np.array([np.interp(t, tg, LAM[i]) for i in range(LAM.shape[0])])


# ---------------- SIR ----------------
for I0 in (1e-2, 1e-4, 1e-6, 1e-8):
    A_ = sc.adjoint_on_grid(I0)
    tg, LAM, T, Te = A_['tg'], A_['LAM'], A_['T'], A_['Te']
    for m, cls in METHODS.items():
        for tol in (1e-4, 1e-6, 1e-8):
            kw = dict(rtol=tol, atol=tol * RATIO)
            if m in ('BDF', 'Radau', 'LSODA'):
                kw['jac'] = sc.jac
            s = cls(sc.rhs, 0.0, np.array([1 - I0, I0]), 1e4, **kw)
            ts, xs = [0.0], [np.array([1 - I0, I0])]
            while s.status == 'running' and s.y[0] > sc.Se and len(ts) < 200000:
                s.step(); ts.append(s.t); xs.append(s.y.copy())
            if s.y[0] > sc.Se or min(x[1] for x in xs) <= 0:
                rows.append(dict(problem='sir', par=I0, solver=m, tol=tol, ok=False))
                continue
            D_end = ts[-1] - T(xs[-1][0])
            do = s.dense_output()
            troot = brentq(lambda t: do(t)[0] - sc.Se, ts[-2], ts[-1], xtol=1e-14)
            D_root = troot - Te
            Dh = 0.0; Aa = 0.0
            for k in range(len(ts) - 1):
                ph = solve_ivp(sc.rhs, [ts[k], ts[k + 1]], xs[k], method='DOP853', rtol=1e-12, atol=1e-20).y[:, -1]
                v = lam_interp(tg, LAM, ts[k + 1]) @ (xs[k + 1] - ph)
                Dh -= v; Aa += abs(v)
            rows.append(dict(problem='sir', par=I0, solver=m, tol=tol, ok=True, steps=len(ts) - 1,
                             D_end=D_end, D_root=D_root, D_hat=Dh, A=Aa))
            print(f'SIR {I0:.0e} {m:6s} {tol:.0e}: D_end={D_end:+.4e} D_root={D_root:+.4e} D_hat={Dh:+.4e} A={Aa:.3e}',
                  flush=True)

# ---------------- Morris-Lecar ----------------
Ic, _ = ml.snic_current()
for Delta in (1e-1, 1e-2):
    f, J = ml.make(Ic + Delta)
    pre = solve_ivp(f, [0, 40000], [-30.0, 0.1], method='DOP853', rtol=1e-13, atol=1e-15, dense_output=True)
    V = pre.y[0]; up = np.where((V[:-1] < 10.0) & (V[1:] >= 10.0))[0]; k = up[-2]
    ts0 = brentq(lambda t: pre.sol(t)[0] - 10.0, pre.t[k], pre.t[k + 1], xtol=1e-14)
    x0 = pre.sol(ts0)
    R = ml.reference(Delta, x0, nsp=1)
    tg, LAM, P = R['tg'], R['LAM'], R['P']
    for m, cls in METHODS.items():
        for tol in (1e-4, 1e-6, 1e-8):
            t0 = time.time()
            kw = dict(rtol=tol, atol=tol * RATIO)
            if m in ('BDF', 'Radau', 'LSODA'):
                kw['jac'] = J
            s = cls(f, 0.0, x0.copy(), 1e9, **kw)
            ts, xs = [0.0], [x0.copy()]
            below = False
            while s.status == 'running' and len(ts) < 2_000_000:
                s.step(); ts.append(s.t); xs.append(s.y.copy())
                if s.y[0] < 9.0:
                    below = True
                if below and s.y[0] >= 10.0:
                    break
            D_end = ts[-1] - R['tref_level'](1, xs[-1][0])
            do = s.dense_output()
            troot = brentq(lambda t: do(t)[0] - 10.0, ts[-2], ts[-1], xtol=1e-13)
            D_root = troot - P
            Dh = 0.0; Aa = 0.0
            for k in range(len(ts) - 1):
                ph = solve_ivp(f, [ts[k], ts[k + 1]], xs[k], method='DOP853', rtol=1e-12, atol=1e-14).y[:, -1]
                v = lam_interp(tg, LAM, ts[k + 1]) @ (xs[k + 1] - ph)
                Dh -= v; Aa += abs(v)
            rows.append(dict(problem='ml', par=Delta, solver=m, tol=tol, ok=True, steps=len(ts) - 1,
                             D_end=D_end, D_root=D_root, D_hat=Dh, A=Aa))
            print(f'ML {Delta:.0e} {m:6s} {tol:.0e}: D_end={D_end:+.4e} D_root={D_root:+.4e} D_hat={Dh:+.4e} '
                  f'A={Aa:.3e} steps={len(ts)-1} ({time.time()-t0:.1f}s)', flush=True)

O = pd.DataFrame(rows)
O.to_csv('../results/event_error_check.csv', index=False)
O = O[O.ok == True].copy()
O['rel_end_root'] = np.abs(O.D_end - O.D_root) / np.abs(O.D_root)
O['rel_hat_root'] = np.abs(O.D_hat - O.D_root) / np.abs(O.D_root)
O['cancel'] = np.abs(O.D_root) / O.A
txt = ['Event-error definitions and first-order representation (SciPy runs)', '']
for p, G in O.groupby('problem'):
    big = G[np.abs(G.D_root) > 1e-6 * (100 if p == 'sir' else 1)]
    txt.append(f'{p}: runs {len(G)} (of {len(rows) if False else (O.problem == p).sum()} completed); |D_end-D_root|/|D_root|: '
               f'median {G.rel_end_root.median():.2e}, max {G.rel_end_root.max():.2e}')
    txt.append(f'   |D_hat-D_root|/|D_root|: median {G.rel_hat_root.median():.2e}, 90% {G.rel_hat_root.quantile(0.9):.2e}, '
               f'max {G.rel_hat_root.max():.2e}; same sign {np.mean(np.sign(G.D_hat) == np.sign(G.D_root)):.2f}')
    txt.append(f'   |D_root|/A (1 = no cancellation): median {G.cancel.median():.3f}, 10-90% '
               f'{G.cancel.quantile(0.1):.3f}-{G.cancel.quantile(0.9):.3f}')
    for s_, H in G.groupby('solver'):
        txt.append(f'     {s_:6s}: |D_root|/A median {H.cancel.median():.3f}')
open('../results/event_error_check.txt', 'w').write('\n'.join(txt) + '\n')
print('\n'.join(txt))
