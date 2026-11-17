"""H2-air ignition, additional runs for article v6.
(a) SciPy default tolerances (rtol 1e-3, atol 1e-6): the runs without ignition before t = 1e-3 s are
    continued to t = 1 s; final temperature and smallest mass fraction are reported.
(b) rtol = 1e-4 with a scalar atol = 1e-6 against componentwise absolute tolerances
    (1e-6 on T, 1e-14 on every mass fraction), SciPy LSODA, BDF and Radau.
(c) Cantera's own reactor network (IdealGasConstPressureReactor, ReactorNet/CVODES) with its default
    tolerances and with rtol = 1e-4, atol = 1e-6 and rtol = 1e-3, atol = 1e-6.
Delay: D = t_n - t_ref(T_n) at the first step with T >= T0 + 400 K (as in the article).
The right-hand side of h2_common.py passes the mass fractions to Cantera unnormalized and unclipped;
negative values are kept (Cantera's set_unnormalized_mass_fractions); a state Cantera cannot evaluate
returns NaN, which makes the integrator reject the step.
Output: ../results/h2_review.csv, ../results/h2_review.txt
"""
import time
import warnings
import numpy as np
import pandas as pd
import cantera as ct
from scipy.integrate import solve_ivp, LSODA, BDF, Radau, RK45
from scipy.optimize import brentq
import h2_common as h

warnings.simplefilter('ignore')
TL = h.T0 + 400.0
ref = solve_ivp(h.rhs, [0, 1e-3], h.x0, method='Radau', rtol=1e-12, atol=1e-24, dense_output=True,
                jac=lambda t, x: h.jac_fd(t, x))
i = np.argmax(ref.y[0] >= TL)
Te = brentq(lambda t: ref.sol(t)[0] - TL, ref.t[i - 1], ref.t[i], xtol=1e-18)
print(f'reference t_e = {Te:.10e}', flush=True)


def t_ref_of_T(Tv):
    lo = 0.0 if ref.sol(Te * 0.5)[0] > Tv else Te * 0.5
    return brentq(lambda t: ref.sol(t)[0] - Tv, lo, 1e-3, xtol=1e-18)


rows = []
M = {'LSODA': LSODA, 'BDF': BDF, 'Radau': Radau, 'RK45': RK45}


def run_scipy(m, rtol, atol, t_end, label):
    kw = dict(rtol=rtol, atol=atol)
    if m != 'RK45':
        kw['jac'] = lambda t, x: h.jac_fd(t, x)
    s = M[m](h.rhs, 0.0, h.x0.copy(), t_end, **kw)
    n = 0; err = ''
    try:
        while s.status == 'running':
            s.step(); n += 1
            if s.y[0] >= TL or n > 400000:
                break
    except Exception as e:
        err = str(e).strip().splitlines()[-1][:60]
    ign = s.y[0] >= TL
    D = s.t - t_ref_of_T(s.y[0]) if ign else np.nan
    r = dict(group=label, code=m, rtol=rtol, atol=str(atol if np.isscalar(atol) else 'componentwise'),
             t_end=t_end, ignited=bool(ign), D=D, rel_delay=D / Te if ign else np.nan, steps=n,
             t_final=s.t, T_final=s.y[0], minY=s.y[1:].min(), status=s.status if not err else 'error: ' + err)
    rows.append(r)
    print(f"{label:22s} {m:6s} rtol={rtol:.0e}: ignited={ign} D/t_e={r['rel_delay']:+.3e} steps={n} "
          f"t={s.t:.3e} T={s.y[0]:.1f} minY={r['minY']:.2e} {r['status']}", flush=True)


# (a) SciPy defaults, window 1 s
for m in ('LSODA', 'BDF', 'Radau', 'RK45'):
    run_scipy(m, 1e-3, 1e-6, 1.0, 'defaults, t_end=1 s')
# (b) scalar vs componentwise atol
atol_vec = np.concatenate([[1e-6], np.full(h.N, 1e-14)])
for m in ('LSODA', 'BDF', 'Radau'):
    run_scipy(m, 1e-4, 1e-6, 1e-3, 'rtol 1e-4, scalar atol')
    run_scipy(m, 1e-4, atol_vec, 1e-3, 'rtol 1e-4, species atol')
# (c) Cantera reactor network
for rt, at, label in ((None, None, 'Cantera defaults'), (1e-4, 1e-6, 'Cantera rtol 1e-4'), (1e-3, 1e-6, 'Cantera rtol 1e-3')):
    gas = ct.Solution('h2o2.yaml'); gas.TPX = h.T0, h.P0, 'H2:2, O2:1, N2:3.76'
    r = ct.IdealGasConstPressureReactor(gas)
    net = ct.ReactorNet([r])
    if rt is not None:
        net.rtol = rt; net.atol = at
    n = 0; t0 = time.time()
    while net.time < 1e-3 and r.T < TL and n < 200000:
        net.step(); n += 1
    ign = r.T >= TL
    D = net.time - t_ref_of_T(r.T) if ign else np.nan
    rows.append(dict(group=label, code='ReactorNet', rtol=net.rtol, atol=str(net.atol), t_end=1e-3, ignited=bool(ign),
                     D=D, rel_delay=D / Te if ign else np.nan, steps=n, t_final=net.time, T_final=r.T,
                     minY=r.thermo.Y.min(), status='ok'))
    print(f"{label:22s} ReactorNet rtol={net.rtol:.0e} atol={net.atol:.0e}: ignited={ign} D/t_e={D/Te:+.3e} steps={n}",
          flush=True)
R = pd.DataFrame(rows)
R.to_csv('../results/h2_review.csv', index=False)
txt = [f'reference t_e = {Te:.6e} s; Cantera {ct.__version__}', ''] + \
      [f"{r.group:24s} {r.code:10s} rtol={r.rtol:.0e} atol={r.atol:>13s}: ignited={r.ignited!s:5s} D/t_e={r.rel_delay:+.2e} "
       f"steps={r.steps:6d} t_final={r.t_final:.2e} T_final={r.T_final:.1f} minY={r.minY:.2e} {r.status}" for r in R.itertuples()]
open('../results/h2_review.txt', 'w').write('\n'.join(txt) + '\n')
print('\n'.join(txt))
