"""Numbers added in article v8.
(1) Morris-Lecar, time as a dependent variable: the relative tolerance 1e-13 on t against the absolute
    tolerance tol*Theta(T_e). The run ends at the tenth spike, t <= 11 periods (the initial point is a spike),
    so the relative part of the error weight of t is at most 1e-13 * 11 P / (tol * Theta_e).
(2) H2 ignition: summary of the NaN-branch check (exp_h2_nan_check.py).
Inputs : ../../experiments/results/ml_cases.csv, ml_runs.csv, ../results/h2_nan_check.csv
Output : ../results/summary_article_v8.txt
"""
import numpy as np
import pandas as pd

out = ['Numbers added in article v8', '']
C = pd.read_csv('../../experiments/results/ml_cases.csv')
R = pd.read_csv('../../experiments/results/ml_runs.csv'); R = R[R.variant == 'time control']
q = []
for r in R.itertuples():
    P = C[np.isclose(C.Delta, r.Delta)].period.iloc[0]
    q.append(1e-13 * 11 * P / (r.tol * r.Theta_e))
    out.append(f'ML time control {r.method:5s} Delta={r.Delta:.0e} tol={r.tol:.0e}: period {P:.1f} ms, '
               f'rtol_t*11P/atol_t = {q[-1]:.2e}')
out.append(f'ML time control: largest relative/absolute weight ratio for t: {max(q):.2e}')
out.append('')
H = pd.read_csv('../results/h2_nan_check.csv')
out.append(f'H2 NaN check: {len(H)} runs, {int((H.nan_returns > 0).sum())} with NaN returns, all of them '
           f'{sorted(set(H[H.nan_returns > 0].method))}; NaN returns per such run '
           f'{int(H[H.nan_returns > 0].nan_returns.min())}-{int(H[H.nan_returns > 0].nan_returns.max())} of '
           f'{int(H[H.nan_returns > 0].rhs_calls.min())}-{int(H[H.nan_returns > 0].rhs_calls.max())} calls; '
           f'implicit codes (LSODA, BDF, Radau): {int(H[H.method != "RK45"].nan_returns.sum())} NaN returns in '
           f'{int((H.method != "RK45").sum())} runs; saved step counts reproduced: '
           f'{int((H.steps_match_saved == True).sum())} of {int(H.steps_match_saved.notna().sum())}')
open('../results/summary_article_v8.txt', 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
