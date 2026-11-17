"""Fixed-step methods: the delay at the event is a phase error as for adaptive codes, but its size is
set by the step size.  Leading term for explicit Euler (order 1):
    D ~ (h/2) * int_0^{T_e} lambda(t) . J(t) f(x(t)) dt,
which for a scalar equation equals (h/2) ln( f(x_e) / f(x_0) ) (the log of the speed contrast).
Methods: explicit Euler (EE), implicit Euler (IE, scalar only), Heun, RK4.
Problems: logistic x0 = 1e-1..1e-7; ghost eps = 1e-2..1e-8 (events x = 1 and x = 0.1);
SIR I0 = 1e-2..1e-8; Morris-Lecar Delta = 1e-1, 1e-2 (first and tenth spike).
Outputs: ../results/fixed_step.csv, ../results/fixed_step.log
"""
import csv, time
import numpy as np
from scipy.optimize import brentq
import phase_common as pc
import sir_common as sc
import ml_common as ml

rows = []


def step_scalar(meth, f, fp, x, h):
    if meth == 'EE':
        return x + h * f(x)
    if meth == 'Heun':
        k1 = f(x); k2 = f(x + h * k1); return x + h / 2 * (k1 + k2)
    if meth == 'RK4':
        k1 = f(x); k2 = f(x + h / 2 * k1); k3 = f(x + h / 2 * k2); k4 = f(x + h * k3)
        return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    if meth == 'IE':
        y = x + h * f(x)
        for _ in range(50):
            dy = (y - x - h * f(y)) / (1 - h * fp(y)); y -= dy
            if abs(dy) < 1e-15 * max(1.0, abs(y)):
                break
        return y


def run_scalar(p, meth, h, xe, nmax=5_000_000):
    x, t, n = p.x0, 0.0, 0
    while x < xe and n < nmax:
        x = step_scalar(meth, p.f, p.fp, x, h); t += h; n += 1
    return t - p.T(x), n


# ---------------- scalar problems ----------------
h = 0.25
for kind, pars, levels in (('logistic', (1e-1, 1e-3, 1e-5, 1e-7), (0.5,)),
                           ('ghost', (1e-2, 1e-4, 1e-6, 1e-8), (0.1, 1.0))):
    for par in pars:
        p = pc.logistic(par) if kind == 'logistic' else pc.ghost(par)
        for xe in levels:
            pred = h / 2 * np.log(p.f(xe) / p.f(p.x0))
            for meth in ('EE', 'IE', 'Heun', 'RK4'):
                t0 = time.time()
                D, n = run_scalar(p, meth, h, xe)
                rows.append(dict(problem=kind, param=par, event=xe, method=meth, h=h, D=D, steps=n,
                                 pred_euler=pred if meth in ('EE', 'IE') else np.nan,
                                 ratio=D / (pred if meth == 'EE' else -pred) if meth in ('EE', 'IE') and pred != 0 else np.nan))
                print(f'{kind:8s} {par:.0e} xe={xe:<4} {meth:4s} h={h}: D={D:+.4e}  pred={pred if meth in ("EE","IE") else np.nan:+.4e}'
                      f'  steps={n}  ({time.time()-t0:.1f}s)', flush=True)

# ---------------- SIR ----------------
for I0 in (1e-2, 1e-4, 1e-6, 1e-8):
    A = sc.adjoint_on_grid(I0)
    tg, X, LAM = A['tg'], A['X'], A['LAM']
    integrand = np.array([LAM[:, k] @ (sc.jac(0, X[:, k]) @ sc.rhs(0, X[:, k])) for k in range(len(tg))])
    I_int = np.trapezoid(integrand, tg)
    for hh in (0.1, 0.05):
        pred = hh / 2 * I_int
        for meth in ('EE', 'Heun', 'RK4'):
            x = np.array([1 - I0, I0]); t = 0.0; n = 0
            while x[0] > sc.Se and n < 10_000_000:
                f = sc.rhs
                if meth == 'EE':
                    x = x + hh * f(0, x)
                elif meth == 'Heun':
                    k1 = f(0, x); k2 = f(0, x + hh * k1); x = x + hh / 2 * (k1 + k2)
                else:
                    k1 = f(0, x); k2 = f(0, x + hh / 2 * k1); k3 = f(0, x + hh / 2 * k2); k4 = f(0, x + hh * k3)
                    x = x + hh / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                t += hh; n += 1
            D = t - A['T'](x[0])
            rows.append(dict(problem='sir', param=I0, event=sc.Se, method=meth, h=hh, D=D, steps=n,
                             pred_euler=pred if meth == 'EE' else np.nan,
                             ratio=D / pred if meth == 'EE' else np.nan))
            print(f'sir      {I0:.0e} {meth:4s} h={hh}: D={D:+.4e}  pred={pred if meth=="EE" else np.nan:+.4e}  steps={n}', flush=True)

# ---------------- Morris-Lecar ----------------
X0 = {1e-1: np.array([9.9999999999707612, 0.06124350123828786]),
      1e-2: np.array([10.000000000021291, 0.061324963909461364])}
Ic, _ = ml.snic_current()
for Dl in (1e-1, 1e-2):
    R = ml.reference(Dl, X0[Dl])
    f, J = ml.make(Ic + Dl)
    integrand = np.array([R['LAM'][:, k] @ (J(0, R['X'][:, k]) @ f(0, R['X'][:, k])) for k in range(len(R['tg']))])
    per = np.trapezoid(integrand, R['tg'])          # one period, from spike to spike
    for hh in (0.05, 0.025):
        pred1 = hh / 2 * per
        for meth in ('EE', 'Heun', 'RK4'):
            t0 = time.time()
            x = X0[Dl].copy(); t = 0.0; n = 0; below = False; k = 0; Ds = []
            while k < 10 and n < 50_000_000:
                if meth == 'EE':
                    xn = x + hh * f(0, x)
                elif meth == 'Heun':
                    k1 = f(0, x); k2 = f(0, x + hh * k1); xn = x + hh / 2 * (k1 + k2)
                else:
                    k1 = f(0, x); k2 = f(0, x + hh / 2 * k1); k3 = f(0, x + hh / 2 * k2); k4 = f(0, x + hh * k3)
                    xn = x + hh / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                x = xn; t += hh; n += 1
                if x[0] < 9.0:
                    below = True
                if below and x[0] >= 10.0:
                    below = False; k += 1
                    Ds.append(t - R['tref_level'](k, x[0]))
            rows.append(dict(problem='ml', param=Dl, event='spike 1', method=meth, h=hh, D=Ds[0], steps=n,
                             pred_euler=pred1 if meth == 'EE' else np.nan,
                             ratio=Ds[0] / pred1 if meth == 'EE' else np.nan))
            rows.append(dict(problem='ml', param=Dl, event='spike 10', method=meth, h=hh, D=Ds[-1], steps=n,
                             pred_euler=10 * pred1 if meth == 'EE' else np.nan,
                             ratio=Ds[-1] / (10 * pred1) if meth == 'EE' else np.nan))
            print(f'ml       {Dl:.0e} {meth:4s} h={hh}: D1={Ds[0]:+.4e} D10={Ds[-1]:+.4e}  pred1={pred1 if meth=="EE" else np.nan:+.4e}'
                  f'  steps={n}  ({time.time()-t0:.1f}s)', flush=True)

with open('../results/fixed_step.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
