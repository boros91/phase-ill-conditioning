"""Check for article v8: is the NaN branch of the H2 right-hand side ever reached?
h2_common.rhs returns NaN when Cantera cannot evaluate a state. This script repeats every SciPy run
reported for the ignition problem, with the same settings, and counts the right-hand-side calls
(including those of the finite-difference Jacobian) that return NaN:
  (1) the solver runs of ../../experiments/code/exp_h2_ignition.py (Tables 11 and the orbit test):
      atol/rtol = 1e-2, 1e-6, 1e-10; RK45, LSODA, BDF, Radau; rtol 1e-4, 1e-6, 1e-8 (RK45 not below 1e-6);
      SciPy defaults; orbit test BDF and LSODA at rtol and rtol/10 on [0, 2 t_e];
  (2) the SciPy runs of exp_h2_review.py (Table 12).
The numbers of accepted steps are compared with the saved runs to confirm that the runs are the same.
Output: ../results/h2_nan_check.csv, ../results/h2_nan_check.txt
"""
import warnings
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp, RK45, LSODA, BDF, Radau
import h2_common as h

warnings.simplefilter('ignore')
TL = h.T0 + 400.0
Te = pd.read_csv('../../experiments/results/h2_cases.csv').Te.iloc[0]
_rhs = h.rhs
CNT = dict(calls=0, nan=0)


def counted(t, x):
    v = _rhs(t, x)
    CNT['calls'] += 1
    if not np.all(np.isfinite(v)):
        CNT['nan'] += 1
    return v


h.rhs = counted          # h.jac_fd looks up rhs in the module, so it is counted as well
M = {'RK45': RK45, 'LSODA': LSODA, 'BDF': BDF, 'Radau': Radau}
rows = []


def stepper(label, m, rtol, atol, t_end):
    CNT.update(calls=0, nan=0)
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
    rows.append(dict(source=label, method=m, rtol=rtol, atol=str(atol if np.isscalar(atol) else 'componentwise'),
                     steps=n, ignited=bool(s.y[0] >= TL), rhs_calls=CNT['calls'], nan_returns=CNT['nan'], error=err))
    print(rows[-1], flush=True)


# (1) stage-2 runs
for ratio in (1e-2, 1e-6, 1e-10):
    for m in M:
        for tol in (1e-4, 1e-6, 1e-8):
            if m == 'RK45' and tol < 1e-6:
                continue
            stepper(f'stage2 ratio={ratio:.0e}', m, tol, tol * ratio, 1e-3)
for m in M:
    stepper('stage2 defaults', m, 1e-3, 1e-6, 1e-3)
for m in ('BDF', 'LSODA'):
    for tol in (1e-4, 1e-6, 1e-8):
        for tt in (tol, tol / 10):
            CNT.update(calls=0, nan=0)
            s = solve_ivp(h.rhs, (0.0, 2 * Te), h.x0, method=m, rtol=tt, atol=tt * 1e-2, dense_output=True,
                          jac=lambda t, x: h.jac_fd(t, x))
            rows.append(dict(source='stage2 orbit test', method=m, rtol=tt, atol=str(tt * 1e-2), steps=s.t.size - 1,
                             ignited=bool((s.y[0] >= TL).any()), rhs_calls=CNT['calls'], nan_returns=CNT['nan'],
                             error='' if s.success else s.message[:60]))
            print(rows[-1], flush=True)
# (2) review runs
for m in ('LSODA', 'BDF', 'Radau', 'RK45'):
    stepper('review defaults, t_end=1 s', m, 1e-3, 1e-6, 1.0)
atol_vec = np.concatenate([[1e-6], np.full(h.N, 1e-14)])
for m in ('LSODA', 'BDF', 'Radau'):
    stepper('review rtol 1e-4, scalar atol', m, 1e-4, 1e-6, 1e-3)
    stepper('review rtol 1e-4, species atol', m, 1e-4, atol_vec, 1e-3)

R = pd.DataFrame(rows)
# comparison with the saved step counts
S2 = pd.read_csv('../../experiments/results/h2_runs.csv')
RV = pd.read_csv('../results/h2_review.csv')
match = []
for r in R.itertuples():
    if r.source.startswith('stage2 ratio'):
        ratio = float(r.source.split('=')[1])
        o = S2[(np.isclose(S2.ratio, ratio)) & (S2.method == r.method) & (np.isclose(S2.tol, r.rtol))]
    elif r.source == 'stage2 defaults':
        o = S2[(np.isclose(S2.ratio, 1e-3)) & (S2.method == r.method)]
    elif r.source.startswith('review'):
        o = RV[(RV.group == r.source[7:]) & (RV.code == r.method)]
    else:
        o = []
    match.append(bool(len(o) and int(o.steps.iloc[0]) == r.steps) if len(o) else None)
R['steps_match_saved'] = match
R.to_csv('../results/h2_nan_check.csv', index=False)
txt = [f'runs {len(R)}; runs with NaN returns {int((R.nan_returns > 0).sum())}; total NaN returns {int(R.nan_returns.sum())} '
       f'of {int(R.rhs_calls.sum())} right-hand-side calls; step counts compared {int(R.steps_match_saved.notna().sum())}, '
       f'equal {int((R.steps_match_saved == True).sum())}', '']
txt += [f'{r.source:32s} {r.method:6s} rtol={r.rtol:.0e} atol={r.atol:>13s} steps={r.steps:6d} ignited={r.ignited!s:5s} '
        f'rhs={r.rhs_calls:7d} NaN={r.nan_returns:4d} saved-steps-match={r.steps_match_saved} {r.error}' for r in R.itertuples()]
open('../results/h2_nan_check.txt', 'w').write('\n'.join(txt) + '\n')
print('\n'.join(txt))
