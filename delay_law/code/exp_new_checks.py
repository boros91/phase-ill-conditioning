"""Further checks for article v3.
A. Pure relative control (AbsTol = 1e-30): delay relative to the passage time, D/t_*, against the
   integrated law D/t_* ~ K tol^alpha <|E|^alpha> (out of sample: alpha, K from the scalar fit).
B. The origin of the error scale: the bottleneck x' = eps + (x-a)^2 shifted to the centre a.
C. Fixed number of nodes: Heun with N = 2000 steps on [0, 2/delta] for the flame problem.
D. A transverse mechanism: x' = x^2(1-x) + c z, z' = -k z; event-time sensitivities by the adjoint,
   by finite differences and the asymptotic values 1/eps^2 and (c/k)/eps^2; Pi and Pi_*.
Outputs: ../results/new_checks_A.csv, _B.csv, _C.csv, _D.csv and ../results/new_checks.txt
"""
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq
import phase_common as pc
import delay_law_common as L
from exp_fixed_step_lib import run_scalar

fit = pd.read_csv('../results/delay_law_fit.csv')
fit = fit[fit.env == 'SciPy'].set_index('solver')
txt = []


def ghost_shift(eps, a):
    se = np.sqrt(eps)
    f = lambda x: eps + (x - a)**2
    fp = lambda x: 2 * (x - a)
    T = lambda x: (np.arctan((x - a) / se) + np.arctan(1.0 / se)) / se
    return pc.Scalar(f'ghost_a{a}', f, fp, a - 1.0, a + 1.0, T, None, '')


def run(prob, m, rtol, atol, pts=None):
    ts, xs, nfev, n, ok = pc.run_to_level(prob, m, rtol, atol)
    return ts[-1] - prob.T(xs[-1]), n, ok


# ---------------- A ----------------
rows = []
probs = [('flame2', p, pc.flame_q(2, p)) for p in (1e-2, 1e-4, 1e-6)] + \
        [('flame3', p, pc.flame_q(3, p)) for p in (1e-1, 1e-2, 1e-3)] + \
        [('ghost', p, pc.ghost(p)) for p in (1e-2, 1e-4, 1e-6)] + \
        [('logistic', p, pc.logistic(p)) for p in (1e-2, 1e-6, 1e-10)]
for name, par, P in probs:
    tstar = P.T(P.xe)
    pts = [0.0] if name == 'ghost' else None
    for m in ('RK45', 'BDF', 'Radau'):
        a, K = fit.loc[m, 'alpha_M2'], fit.loc[m, 'K_M2']
        lam = L.Lambda(P.f, P.fp, P.x0, P.xe, a, 0.0, 'scipy', pts)
        the = abs(P.xe) / abs(P.f(P.xe))
        mean_sig = lam * the / tstar                      # <sigma^alpha> along the passage
        D, n, ok = run(P, m, 1e-6, 1e-30)
        rows.append(dict(problem=name, par=par, solver=m, tstar=tstar, D=D, steps=n, ok=ok,
                         D_over_t=D / tstar, pred_abs=K * 1e-6**a * mean_sig))
        print(f'A {name:8s} {par:.0e} {m:5s}: t*={tstar:.4g} D={D:+.3e} D/t*={D/tstar:+.2e} '
              f'pred |D|/t*={K*1e-6**a*mean_sig:.2e} steps={n}', flush=True)
A = pd.DataFrame(rows); A.to_csv('../results/new_checks_A.csv', index=False)
for m, G in A[A.ok].groupby('solver'):
    r = np.log10(np.abs(G.D_over_t) / G.pred_abs)
    txt.append(f'A {m:5s}: |D|/t* from {np.abs(G.D_over_t).min():.2e} to {np.abs(G.D_over_t).max():.2e} '
               f'(t* from {G.tstar.min():.3g} to {G.tstar.max():.3g}); log10(measured/pred) mean {r.mean():+.2f}, '
               f'range {r.min():+.2f}..{r.max():+.2f}; signs {"".join("+" if v > 0 else "-" for v in G.D)}')

# ---------------- B ----------------
rows = []
for a in (0.0, 10.0, 1000.0):
    P = ghost_shift(1e-4, a)
    Pi_rel = max(abs(x) / P.f(x) for x in np.linspace(a - 1, a + 1, 200001)) / (abs(P.xe) / P.f(P.xe))
    for m in ('RK45', 'BDF'):
        al, K = fit.loc[m, 'alpha_M2'], fit.loc[m, 'K_M2']
        lam = L.Lambda(P.f, P.fp, P.x0, P.xe, al, 0.0, 'scipy', [a])
        the = abs(P.xe) / abs(P.f(P.xe))
        D, n, ok = run(P, m, 1e-6, 1e-30)
        tstar = P.T(P.xe)
        rows.append(dict(a=a, solver=m, tstar=tstar, D=D, D_over_t=D / tstar, eta=abs(D) / the, Theta_e=the,
                         Pi=Pi_rel, Lambda=lam, pred_eta=K * 1e-6**al * lam, steps=n, ok=ok))
        print(f'B a={a:6g} {m:5s}: D/t*={D/tstar:+.2e} eta={abs(D)/the:.2e} Pi={Pi_rel:.3g} '
              f'pred eta={K*1e-6**al*lam:.2e} steps={n}', flush=True)
B = pd.DataFrame(rows); B.to_csv('../results/new_checks_B.csv', index=False)
txt.append('')
for _, r in B.iterrows():
    txt.append(f'B a={r.a:6g} {r.solver:5s}: D/t*={r.D_over_t:+.2e}  eta={r.eta:.2e}  Pi={r.Pi:.3g}  '
               f'Lambda pred eta={r.pred_eta:.2e}  steps={r.steps}')

# ---------------- C ----------------
rows = []
for d in (1e-1, 1e-2, 1e-3):
    P = pc.flame_q(2, d)
    h = (2 / d) / 2000
    for m in ('EE', 'Heun', 'RK4'):
        D, n = run_scalar(P, m, h, 0.5)
        rows.append(dict(delta=d, method=m, h=h, D=D, steps=n, D_delta2=D * d * d))
C = pd.DataFrame(rows); C.to_csv('../results/new_checks_C.csv', index=False)
txt.append('')
for _, r in C.iterrows():
    txt.append(f'C delta={r.delta:.0e} {r.method:4s} N=2000 on [0,2/delta]: h={r.h:g} D={r.D:+.3e} D*delta^2={r.D_delta2:.3e} steps to event={r.steps}')

# ---------------- D ----------------
c, k, xs = 0.5, 1.0, 0.5
rows = []
for eps in (1e-1, 1e-2, 1e-3):
    F = lambda t, u: np.array([u[0]**2 * (1 - u[0]) + c * u[1], -k * u[1]])
    Jm = lambda u: np.array([[2 * u[0] - 3 * u[0]**2, c], [0.0, -k]])

    def tau(x0, z0):
        ev = lambda t, u: u[0] - xs; ev.terminal = True; ev.direction = 1
        s = solve_ivp(F, [0, 1e7], [x0, z0], method='DOP853', rtol=1e-13, atol=1e-16, events=ev)
        return s.t_events[0][0]
    te = tau(eps, 0.0)
    ref = solve_ivp(F, [0, te], [eps, 0.0], method='DOP853', rtol=1e-13, atol=1e-16, dense_output=True)
    fe = F(te, ref.sol(te))
    adj = solve_ivp(lambda t, l: -Jm(ref.sol(t)).T @ l, [te, 0], [1 / fe[0], 0.0], method='DOP853',
                    rtol=1e-12, atol=1e-14, dense_output=True)
    lam0 = adj.sol(0.0)
    hx, hz = 1e-6 * eps, 1e-6
    fd_x = (tau(eps + hx, 0) - tau(eps - hx, 0)) / (2 * hx)
    fd_z = (tau(eps, hz) - tau(eps, -hz)) / (2 * hz)
    # Pi and Pi_* in the SciPy scale with AbsTol/RelTol = 1e-2
    tg = np.linspace(0, te, 20001)
    X = ref.sol(tg); LAM = adj.sol(tg); S = np.abs(X) + 1e-2
    th = (np.abs(LAM) * S).sum(0)
    Fg = np.array([F(0, X[:, j]) for j in range(len(tg))]).T
    mu = (np.abs(Fg) / S).max(0)
    rows.append(dict(eps=eps, te=te, dtau_dx0_adj=-lam0[0], dtau_dx0_fd=fd_x, dtau_dz0_adj=-lam0[1], dtau_dz0_fd=fd_z,
                     asym_x=-1 / eps**2, asym_z=-(c / k) / eps**2, Pi=th.max() / th[-1], Pi_star=(1 / mu).max() / th[-1]))
Dd = pd.DataFrame(rows); Dd.to_csv('../results/new_checks_D.csv', index=False)
txt.append('')
for _, r in Dd.iterrows():
    txt.append(f'D eps={r.eps:.0e}: dtau/dx0 adj {r.dtau_dx0_adj:.6g} fd {r.dtau_dx0_fd:.6g} (eps*: {r.eps*r.dtau_dx0_adj:.4g}); '
               f'dtau/dz0 adj {r.dtau_dz0_adj:.6g} fd {r.dtau_dz0_fd:.6g} (eps*: {r.eps*r.dtau_dz0_adj:.4g}); '
               f'asymptotic {r.asym_x:.4g}, {r.asym_z:.4g}; Pi={r.Pi:.4g} Pi_*={r.Pi_star:.4g}')
open('../results/new_checks.txt', 'w').write('\n'.join(txt) + '\n')
print('\n'.join(txt))
