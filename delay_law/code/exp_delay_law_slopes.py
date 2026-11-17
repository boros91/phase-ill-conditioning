"""Scaling exponents: how does the delay grow along a family (delta, eps, x0) at a fixed solver and
tolerance, and which indicator predicts the exponent?  For every group (env, solver, tol, AbsTol/RelTol,
problem family) with at least three parameter values, the slope of log eta against log(parameter) is
compared with the slopes of log Pi (model M0) and of log Lambda_alpha (model M2, alpha of the solver).

Input : ../results/delay_law_fit_runs.csv      Output: ../results/delay_law_slopes.csv, .txt
"""
import numpy as np
import pandas as pd

R = pd.read_csv('../results/delay_law_fit_runs.csv')
R = R[R.res_M0.notna()]
R['ratio_r'] = np.round(np.log10(R.ratio), 3)
rows = []
for (env, solver, tol, rr, prob), G in R.groupby(['env', 'solver', 'tol', 'ratio_r', 'problem']):
    if G.par.nunique() < 3:
        continue
    lp = np.log10(G.par.values)
    s_eta = np.polyfit(lp, np.log10(G.eta.values), 1)[0]
    s_pi = np.polyfit(lp, np.log10(G.Pi.values), 1)[0]
    s_lam = np.polyfit(lp, np.log10(G.Lambda.values), 1)[0]
    rows.append(dict(env=env, solver=solver, tol=tol, ratio=10**rr, problem=prob, n=len(G),
                     slope_eta=s_eta, slope_Pi=s_pi, slope_Lambda=s_lam))
S = pd.DataFrame(rows)
S['err_Pi'] = S.slope_eta - S.slope_Pi
S['err_Lambda'] = S.slope_eta - S.slope_Lambda
S.to_csv('../results/delay_law_slopes.csv', index=False)
txt = ['Slopes of log eta against log parameter (fixed solver, tolerance, ratio): measured minus predicted', '']
for prob, G in S.groupby('problem'):
    txt.append(f'{prob:9s} groups={len(G):3d}  slope eta median {G.slope_eta.median():+.2f}  '
               f'Pi: median slope {G.slope_Pi.median():+.2f}, median |error| {G.err_Pi.abs().median():.2f}   '
               f'Lambda: median slope {G.slope_Lambda.median():+.2f}, median |error| {G.err_Lambda.abs().median():.2f}')
txt.append('')
txt.append(f'all groups={len(S)}: median |error| Pi {S.err_Pi.abs().median():.3f}, Lambda {S.err_Lambda.abs().median():.3f}; '
           f'|error|<0.1: Pi {np.mean(S.err_Pi.abs()<0.1):.2f}, Lambda {np.mean(S.err_Lambda.abs()<0.1):.2f}')
gh = S[S.problem == 'ghost']
txt.append(f'ghost groups (AbsTol = RelTol/100): measured slope range {gh.slope_eta.min():.2f} .. {gh.slope_eta.max():.2f}, '
           f'Pi slope {gh.slope_Pi.median():.2f}, Lambda slope range {gh.slope_Lambda.min():.2f} .. {gh.slope_Lambda.max():.2f}')
open('../results/delay_law_slopes.txt', 'w').write('\n'.join(txt) + '\n')
print('\n'.join(txt))
