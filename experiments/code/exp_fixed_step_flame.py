"""Fixed-step methods on the flame problem y' = y^2 - y^3, y(0) = delta, event y = 1/2,
h = 0.25 (independent Python replication of the fixed-step experiment of Paper A),
and the leading terms of the delay for explicit Euler and Heun on all scalar problems.

Leading terms (scalar x' = f(x), local error le = method - exact, time shift -le/f per step):
  explicit Euler: le = -(h^2/2) f' f                ->  D ~ (h/2) ln(f(x_e)/f(x_0))
  implicit Euler: le = +(h^2/2) f' f                ->  D ~ -(h/2) ln(f(x_e)/f(x_0))
  Heun:           le = h^3 (f'' f^2/12 - f'^2 f/6)  ->  D ~ h^2 int_{x_0}^{x_e} (f'^2/(6f) - f''/12) dx
Outputs: ../results/fixed_step_flame.csv, ../results/fixed_step_heun_pred.csv
"""
import csv, time
import numpy as np
from scipy.integrate import quad
import phase_common as pc
from exp_fixed_step_lib import run_scalar

h = 0.25
rows = []
for delta in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
    p = pc.flame_q(2, delta)
    pred = h / 2 * np.log(p.f(0.5) / p.f(delta))
    for meth in ('EE', 'IE', 'Heun', 'RK4'):
        t0 = time.time()
        D, n = run_scalar(p, meth, h, 0.5)
        rows.append(dict(problem='flame', param=delta, event=0.5, method=meth, h=h, D=D, steps=n,
                         pred_euler=pred if meth in ('EE', 'IE') else np.nan))
        print(f'flame {delta:.0e} {meth:4s} h={h}: D={D:+.4e}  pred_EE={pred:+.4e}  steps={n}  ({time.time()-t0:.1f}s)',
              flush=True)
with open('../results/fixed_step_flame.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


# ---- leading term for Heun on the scalar problems ----
def heun_pred(f, fp, fpp, x0, xe, pts=None):
    g = lambda x: fp(x)**2 / (6 * f(x)) - fpp(x) / 12
    val, err = quad(g, x0, xe, points=pts, limit=500, epsabs=1e-13, epsrel=1e-11)
    return h**2 * val


out = []
for x0 in (1e-1, 1e-3, 1e-5, 1e-7):
    out.append(dict(problem='logistic', param=x0, event=0.5,
                    pred_heun=heun_pred(lambda x: x * (1 - x), lambda x: 1 - 2 * x, lambda x: -2.0, x0, 0.5)))
for eps in (1e-2, 1e-4, 1e-6, 1e-8):
    for xe in (0.1, 1.0):
        out.append(dict(problem='ghost', param=eps, event=xe,
                        pred_heun=heun_pred(lambda x: eps + x * x, lambda x: 2 * x, lambda x: 2.0, -1.0, xe, pts=[0.0])))
for delta in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
    out.append(dict(problem='flame', param=delta, event=0.5,
                    pred_heun=heun_pred(lambda y: y * y * (1 - y), lambda y: 2 * y - 3 * y * y, lambda y: 2 - 6 * y,
                                        delta, 0.5)))
for r in out:
    print(f"Heun leading term  {r['problem']:8s} {r['param']:.0e} xe={r['event']}: {r['pred_heun']:+.4e}")
with open('../results/fixed_step_heun_pred.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
