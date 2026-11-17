"""Numbers added in article v7.
(1) Morris-Lecar, decomposition of the prediction error on the matched SciPy runs (Table 6):
        log10(|D|/P) = log10(A/P) + log10(|D|/A),
    P the integrated-law prediction (scalar fit, no refitting), A the sum of absolute time defects.
(2) Cost of time as a dependent variable for Morris-Lecar: accepted steps and function evaluations
    relative to the standard run of the same code, tolerance and Delta.
Inputs : ../results/event_error_check.csv, ../results/delay_law_systems_runs.csv,
         ../../experiments/results/ml_runs.csv
Output : ../results/summary_article_v7.txt
"""
import numpy as np
import pandas as pd

EV = pd.read_csv('../results/event_error_check.csv'); EV = EV[EV.ok == True]
SY = pd.read_csv('../results/delay_law_systems_runs.csv'); SY = SY[SY.env == 'SciPy']
M = EV.merge(SY[['problem', 'par', 'solver', 'tol', 'pred_M2', 'Theta_e']], on=['problem', 'par', 'solver', 'tol'])
M['P'] = M.pred_M2 * M.Theta_e
M['lDP'] = np.log10(np.abs(M.D_end) / M.P)
M['lAP'] = np.log10(M.A / M.P)
M['lDA'] = np.log10(np.abs(M.D_end) / M.A)
out = ['Numbers added in article v7', '']
for p, G in M.groupby('problem'):
    out.append(f'{p}: matched runs {len(G)}; mean log10(|D|/P) {G.lDP.mean():+.3f} = mean log10(A/P) {G.lAP.mean():+.3f} '
               f'+ mean log10(|D|/A) {G.lDA.mean():+.3f}; medians {G.lDP.median():+.3f}, {G.lAP.median():+.3f}, {G.lDA.median():+.3f}; '
               f'share of the mean due to cancellation {G.lDA.mean()/G.lDP.mean():.2f}')
    for s, H in G.groupby('solver'):
        out.append(f'   {s:6s}: mean log10(|D|/P) {H.lDP.mean():+.2f}, log10(A/P) {H.lAP.mean():+.2f}, log10(|D|/A) {H.lDA.mean():+.2f}')
out.append('')
R = pd.read_csv('../../experiments/results/ml_runs.csv')
rows = []
for (D, m, tol), G in R.groupby(['Delta', 'method', 'tol']):
    if set(G.variant) >= {'standard', 'time control'}:
        a = G[G.variant == 'standard'].iloc[0]; b = G[G.variant == 'time control'].iloc[0]
        rows.append(dict(Delta=D, method=m, tol=tol, steps_ratio=b.steps / a.steps, nfev_ratio=b.nfev / a.nfev,
                         D10_std=a.D10, D10_tc=b.D10))
C = pd.DataFrame(rows)
for m, G in C.groupby('method'):
    out.append(f'time control {m}: steps ratio {G.steps_ratio.min():.2f}-{G.steps_ratio.max():.2f}, '
               f'f-eval ratio {G.nfev_ratio.min():.2f}-{G.nfev_ratio.max():.2f} (n={len(G)}); '
               f'max |D10| {G.D10_tc.abs().max():.3f} ms against standard max {G.D10_std.abs().max():.3f} ms')
for (m, tol), G in C.groupby(['method', 'tol']):
    out.append(f'time control {m} tol={tol:.0e}: max |D10| {G.D10_tc.abs().max():.2e} ms')
for (v, m), G in R.groupby(['variant', 'method']):
    out.append(f'max |D10| over Delta and tol, {v} {m}: {G.D10.abs().max():.3f} ms')
for r in C.itertuples():
    out.append(f'   {r.method:5s} Delta={r.Delta:.0e} tol={r.tol:.0e}: steps x{r.steps_ratio:.2f} f-evals x{r.nfev_ratio:.2f} '
               f'D10 {r.D10_std:+.4f} -> {r.D10_tc:+.4f}')
open('../results/summary_article_v7.txt', 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
