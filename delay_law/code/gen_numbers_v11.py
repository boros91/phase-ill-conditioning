"""Numbers added in article v11: the two SciPy runs without an event in the Pi table (BDF, tol = 1e-4).
(1) logistic x0 = 1e-7: repeated with phase_common.run_to_level settings (t_bound = 1e12); final status,
    time, state and number of steps;
(2) SIR I0 = 1e-8: read from ../../experiments/results/sir_runs.csv (stopped at t > 10 T_e).
Output: ../results/summary_article_v11.txt
"""
import numpy as np
import pandas as pd
from scipy.integrate import BDF
import phase_common as pc

out = ['Numbers added in article v11', '']
p = pc.logistic(1e-7)
s = BDF(lambda t, y: np.array([p.f(y[0])]), 0.0, np.array([p.x0]), 1e12, rtol=1e-4, atol=1e-6,
        jac=lambda t, y: np.array([[p.fp(y[0])]]))
n = 0
while s.status == 'running' and n < 2_000_000:
    s.step(); n += 1
    if s.y[0] >= p.xe or not np.isfinite(s.y[0]):
        break
out.append(f'logistic x0=1e-7 BDF tol=1e-4 atol=1e-6: status {s.status}, steps {n}, t={s.t:.3e}, x={s.y[0]:.3e}')
r = pd.read_csv('../../experiments/results/sir_runs.csv')
q = r[(~r.ok) & (r.method == 'BDF')].iloc[0]
c = pd.read_csv('../../experiments/results/sir_cases.csv')
Te = c[np.isclose(c.I0, q.I0)].Te.iloc[0]
out.append(f'SIR I0={q.I0:.0e} BDF tol={q.tol:.0e}: steps {q.steps}, t_end={q.t_end:.4g} = {q.t_end/Te:.2f} T_e, '
           f'min I={q.minI:.3e}')
open('../results/summary_article_v11.txt', 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
