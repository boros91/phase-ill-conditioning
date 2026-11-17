"""SIR epidemic started from a small infected fraction: phase condition number
from the adjoint (event-time sensitivity), measured delays of the SciPy
integrators, and the a posteriori estimate D_hat = -sum_k lambda_k . le_k.

S' = -b S I, I' = b S I - g I, b = 0.3, g = 0.1, S(0) = 1 - I0, I(0) = I0.
Event: S = g/b (peak of I).  Exact time to reach S by quadrature.
Outputs: ../results/sir_cases.csv, ../results/sir_runs.csv, ../results/sir_theta.csv
"""
import csv
import numpy as np
from scipy.integrate import quad, solve_ivp, RK45, DOP853, LSODA, BDF, Radau

b, g = 0.3, 0.1
c = g / b
Se = c
RATIO = 1e-2
METHODS = {'RK45': RK45, 'DOP853': DOP853, 'LSODA': LSODA, 'BDF': BDF, 'Radau': Radau}


def make_T(I0):
    S0 = 1.0 - I0

    def Ifun(z):                       # I as a function of z = S0 - S on the orbit
        return I0 + z + c * np.log1p(-z / S0)

    def T(S):
        Z = S0 - S
        if Z <= 0:
            return 0.0
        U = np.log1p(Z / I0)
        h = lambda u: (I0 * np.exp(u)) / (b * (S0 - I0 * np.expm1(u)) * Ifun(I0 * np.expm1(u)))
        pts = np.linspace(0, U, 9)[1:-1]
        val, _ = quad(h, 0.0, U, points=pts, limit=1000, epsabs=1e-13, epsrel=1e-13)
        return val
    return S0, T


def rhs(t, x):
    S, I = x
    return np.array([-b * S * I, b * S * I - g * I])


def jac(t, x):
    S, I = x
    return np.array([[-b * I, -b * S], [b * I, b * S - g]])


def reference(I0, Te):
    S0 = 1.0 - I0
    lr = lambda t, u: np.array([-b * np.exp(u[1]), b * np.exp(u[0]) - g])   # u = (ln S, ln I)
    sol = solve_ivp(lr, [0, Te], [np.log(S0), np.log(I0)], method='DOP853',
                    rtol=1e-13, atol=1e-13, dense_output=True)
    X = lambda t: np.exp(sol.sol(t))
    xe = X(Te)
    fe = rhs(Te, xe)
    lamT = np.array([1.0 / fe[0], 0.0])
    adj = lambda t, lam: -jac(t, X(t)).T @ lam
    sa = solve_ivp(adj, [Te, 0.0], lamT, method='DOP853', rtol=1e-11, atol=1e-30, dense_output=True)
    return X, sa.sol, xe, fe


def Theta(lam, x, ratio):
    return np.abs(lam[0]) * (np.abs(x[0]) + ratio) + np.abs(lam[1]) * (np.abs(x[1]) + ratio)


def run(I0, S0, T, method, tol, Te, lam_of_S, estimate=True):
    kw = dict(rtol=tol, atol=tol * RATIO)
    if method in ('BDF', 'Radau', 'LSODA'):
        kw['jac'] = jac
    s = METHODS[method](rhs, 0.0, np.array([S0, I0]), 50 * Te, **kw)
    ts, xs = [0.0], [s.y.copy()]
    while s.status == 'running':
        s.step()
        ts.append(s.t); xs.append(s.y.copy())
        if s.y[0] <= Se or s.t > 10 * Te:
            break
    ts, xs = np.array(ts), np.array(xs)
    ok = xs[-1, 0] <= Se
    if not ok:
        return dict(ok=False, steps=len(ts) - 1, nfev=s.nfev, D=np.nan, Dhat=np.nan,
                    minI=xs[:, 1].min(), t_end=ts[-1])
    D = ts[-1] - T(xs[-1, 0])
    Dhat = np.nan
    if estimate and xs[:, 1].min() > 0:
        Dhat = 0.0
        for k in range(len(ts) - 1):
            ref = solve_ivp(rhs, [ts[k], ts[k + 1]], xs[k], method='DOP853',
                            rtol=1e-13, atol=1e-22)
            le = xs[k + 1] - ref.y[:, -1]
            Dhat -= lam_of_S(xs[k + 1, 0]) @ le
    return dict(ok=True, steps=len(ts) - 1, nfev=s.nfev, D=D, Dhat=Dhat,
                minI=xs[:, 1].min(), t_end=ts[-1])


cases, runs, throws = [], [], []
for I0 in (1e-2, 1e-4, 1e-6, 1e-8):
    S0, T = make_T(I0)
    Te = T(Se)
    X, LAM, xe, fe = reference(I0, Te)
    tg = np.unique(np.concatenate([np.linspace(0, Te, 20001), Te * (1 - np.geomspace(1e-9, 1, 2000))]))
    lam = LAM(tg); xx = X(tg)
    th = Theta(lam, xx, RATIO); th_rel = Theta(lam, xx, 0.0)
    the = Theta(LAM(Te), xe, RATIO); the_rel = Theta(LAM(Te), xe, 0.0)
    Pi, Pi_rel = th.max() / the, th_rel.max() / the_rel
    # adjoint-free lower bound: Theta(t) >= 1/mu(t), mu = max_i |f_i| / s_i
    F = np.array([rhs(0, xx[:, k]) for k in range(xx.shape[1])]).T
    mu = np.max(np.abs(F) / (np.abs(xx) + RATIO), axis=0)
    mu_rel = np.max(np.abs(F) / np.abs(xx), axis=0)
    Pi_star, Pi_star_rel = (1 / mu).max() / the, (1 / mu_rel).max() / the_rel
    # lambda as a function of S on the reference orbit (S is decreasing in t)
    Sg = xx[0]
    order = np.argsort(Sg)
    def lam_of_S(S, Sg=Sg[order], L=lam[:, order], T=T, LAM=LAM):
        return LAM(min(max(T(S), 0.0), Te))
    cases.append(dict(I0=I0, Te=Te, Theta_e=the, Pi=Pi, t_argmax=tg[np.argmax(th)],
                      Pi_relative=Pi_rel, Pi_star=Pi_star, Pi_star_relative=Pi_star_rel, lamS0=lam[0, 0], lamI0=lam[1, 0], Imax=xx[1].max()))
    for k in range(0, len(tg), 200):
        throws.append(dict(I0=I0, t=tg[k], S=xx[0, k], I=xx[1, k], lamS=lam[0, k], lamI=lam[1, k],
                           Theta=th[k], Theta_rel=th_rel[k]))
    print(f"I0={I0:.0e}: Te={Te:.6f}  Theta_e={the:.4f}  Pi={Pi:.3e}  Pi_rel={Pi_rel:.3e}  "
          f"Pi_star={Pi_star:.3e}  Pi_star_rel={Pi_star_rel:.3e}  argmax t={tg[np.argmax(th)]:.3g}")
    for m in METHODS:
        for tol in (1e-4, 1e-6, 1e-8):
            r = run(I0, S0, T, m, tol, Te, lam_of_S)
            r.update(I0=I0, method=m, tol=tol, Pi=Pi, tolPi=tol * Pi, Theta_e=the,
                     eta=abs(r['D']) / the if r['ok'] else np.nan)
            runs.append(r)
            print(f"   {m:6s} {tol:6.0e}  ok={r['ok']!s:5s} D={r['D']:+.3e}  Dhat={r['Dhat']:+.3e}  "
                  f"eta={r['eta']:.2e} tolPi={tol*Pi:.2e} steps={r['steps']} minI={r['minI']:.1e}")

for name, rows in (('sir_cases', cases), ('sir_runs', runs), ('sir_theta', throws)):
    with open(f'../results/{name}.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
