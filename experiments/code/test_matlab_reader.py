"""Checks gen_matlab_phase_index.py on files in the format of run_phase_index.m, produced here
with SciPy (RK45 and BDF stand in for the MATLAB solvers).  Writes to the scratch folder."""
import os, sys, subprocess
import numpy as np, pandas as pd
from scipy.integrate import solve_ivp
import phase_common as pc, sir_common as sc, ml_common as ml
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
stats, traj = [], []
j = 0
X0 = {1e-1: [9.9999999999707612, 0.06124350123828786]}
for name, par in (('ghost', 1e-4), ('logistic', 1e-5), ('sir', 1e-6), ('ml', 1e-1)):
    for m in ('RK45', 'BDF'):
        j += 1
        if name in ('ghost', 'logistic'):
            p = pc.ghost(par) if name == 'ghost' else pc.logistic(par)
            f = lambda t, y, p=p: [p.f(y[0])]; y0 = [p.x0]; tend = 2 * p.T(p.xe)
        elif name == 'sir':
            f = sc.rhs; y0 = [1 - par, par]; tend = 200
        else:
            Ic, _ = ml.snic_current(); f, _J = ml.make(Ic + par); y0 = X0[par]; tend = 11 * 582.124
        s = solve_ivp(f, [0, tend], y0, method=m, rtol=1e-6, atol=1e-8)
        Y = s.y.T
        if Y.shape[1] == 1:
            Y = np.hstack([Y, np.full((len(Y), 1), np.nan)])
        for k in range(len(s.t)):
            traj.append((j, k, s.t[k], Y[k, 0], Y[k, 1]))
        stats.append(dict(run=j, problem=name, param=par, solver=m, RelTol=1e-6, AbsTol=1e-8,
                          nsteps=len(s.t) - 1, nfailed=0, nfevals=s.nfev, time_s=0, event=1, warning='none'))
pd.DataFrame(stats).to_csv(f'{out}/stats.csv', index=False)
pd.DataFrame(traj, columns=['run', 'k', 't', 'y1', 'y2']).to_csv(f'{out}/traj.csv', index=False)
