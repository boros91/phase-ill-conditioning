"""Further numbers quoted in the paper.
(1) Pooled threshold counts over the 1215-run suite (Sects. 7.2 and 12): the 240 SciPy scalar runs
    and 60 SIR runs, 539 completed MATLAB flame runs, 40 first-set MATLAB SIR runs and 336 second-set
    MATLAB runs, in the bands rtol*Pi < 1e-3, 1e-3 <= rtol*Pi < 1, rtol*Pi >= 1; and how each run
    without a recorded crossing ended (clean completion, stopping rule, solver failure).
(2) Held-out scalar families (Sect. 7.4): the fixed-power model c r Pi, a two-parameter power model
    K r^alpha Pi refitted on the three training families (free least-squares slope, and the slope
    chosen on the grid used for the integrated law), and the archived integrated-law residuals.
(3) Systems out of sample: the scalar-trained models on SIR and Morris-Lecar, all runs and the
    SciPy Morris-Lecar runs.
Run from additional/code.  Inputs: ../../experiments/results, ../../delay_law/results.
Output: ../results/summary_v18.txt
"""
import numpy as np
import pandas as pd

E = '../../experiments/results/'
L = '../../delay_law/results/'
out = ['Further numbers quoted in the paper (gen_numbers_v18.py)', '']

# ---------------------------------------------------------------- (1) pooled bands
frames = []
for name in ('index_scalar', 'sir_runs', 'matlab_pi_flame', 'matlab_pi_sir', 'matlab_phase_index'):
    r = pd.read_csv(E + name + '.csv')
    if name == 'matlab_pi_flame':
        r = r[r.completed == 1]
    if name == 'matlab_phase_index':
        r.loc[r.event == 0, 'eta'] = np.nan
    r = r[['tolPi', 'eta']].copy(); r['set'] = name
    frames.append(r)
P = pd.concat(frames, ignore_index=True)
band = pd.cut(P.tolPi, [-np.inf, 1e-3, 1.0, np.inf], right=False, labels=['low', 'mid', 'high'])
out.append(f'pooled runs: {len(P)} (' + ', '.join(f'{k} {len(g)}' for k, g in P.groupby("set", sort=False)) + ')')
for b, g in P.groupby(band, observed=True):
    e = g.eta.dropna()
    out.append(f'  band {b}: runs {len(g)}, recorded crossings {len(e)}, no crossing {len(g)-len(e)}; '
               f'eta median {e.median():.4g}, max {e.max():.4g}')
hi = P[band == 'high'].eta.dropna()
out.append(f'  high band: crossings with eta < 1: {(hi < 1).sum()}, with eta < 0.1: {(hi < 0.1).sum()} (of {len(hi)})')
# how the runs without a crossing ended
nc = []
s = pd.read_csv(E + 'index_scalar.csv'); s = s[~s.ok]
for r in s.itertuples():
    nc.append(('SciPy scalar', r.case, r.method, 'clean completion (solver reached t = 1e12 without a crossing)'))
s = pd.read_csv(E + 'sir_runs.csv'); s = s[~s.ok]
for r in s.itertuples():
    nc.append(('SciPy SIR', f'I0={r.I0:.0e}', r.method, 'stopping rule (first step past t = 10 T_e)'))
st = pd.read_csv(E + 'matlab/phase_index_stats.csv')
for r in st[st.event == 0].itertuples():
    kind = 'clean completion' if str(r.warning).strip() == 'none' else 'solver failure: ' + str(r.warning)[:60]
    nc.append(('MATLAB second set', f'{r.problem} {r.param:.0e}', r.solver, kind))
ms = pd.read_csv(E + 'matlab_pi_sir.csv'); ms = ms[ms.eta.isna()]
os_ = pd.read_csv(E + 'matlab/other_problems_stats.csv')
for r in ms.itertuples():
    m = os_[(os_.example == 'sir') & np.isclose(os_.param, r.I0) & (os_.variant == r.variant) & (os_.form == 'std')
            & (os_.solver == r.solver) & np.isclose(os_.RelTol, r.RelTol)]
    if len(m):
        w = str(m.warning.iloc[0]).strip(); c = int(m.completed.iloc[0])
        kind = ('clean completion' if c == 1 and w == 'none' else f'completed={c}, warning: {w[:60]}')
    else:
        kind = 'not found in other_problems_stats.csv'
    nc.append(('MATLAB first set SIR', f'I0={r.I0:.0e} {r.variant}', r.solver, kind))
out.append(f'runs without a recorded crossing: {len(nc)}')
for x in nc:
    out.append('   ' + ' | '.join(x))
kinds = pd.Series([x[3].split(':')[0].split(' (')[0] for x in nc]).value_counts()
out.append('   summary: ' + ', '.join(f'{k} {v}' for k, v in kinds.items()))
out.append('')

# ---------------------------------------------------------------- (2) held-out families
runs = pd.read_csv(L + 'delay_law_fit_runs.csv').dropna(subset=['res_M0'])
saved = pd.read_csv(L + 'delay_law_lofo.csv')
fixed, free, gridded = [], [], []
grid = np.round(np.arange(0.30, 1.501, 0.05), 2)
for _, g in runs.groupby(['env', 'solver']):
    for fam in g.problem.unique():
        tr, te = g[g.problem != fam], g[g.problem == fam]
        if len(tr) < 5 or te.empty:
            continue
        x, y = np.log10(tr.tol), np.log10(tr.eta / tr.Pi)
        xt, yt = np.log10(te.tol), np.log10(te.eta / te.Pi)
        a, b = np.polyfit(x, y, 1)
        free.extend(yt - (a * xt + b))
        fixed.extend(yt - xt - (y - x).mean())
        ag = min(grid, key=lambda v: np.std(y - v * x))
        gridded.extend(yt - ag * xt - (y - ag * x).mean())
assert np.allclose(fixed, saved.res_M0, rtol=1e-10, atol=1e-12)


def score(label, res):
    e = np.abs(np.asarray(res))
    return (f'  {label}: {np.sum(e < 1)}/{len(e)} = {100*np.mean(e < 1):.1f}% within a factor 10; '
            f'median |log10 residual| {np.median(e):.3f} (factor {10**np.median(e):.2f})')


out.append('held-out scalar families (leave one family out):')
out.append(score('c r Pi (one fitted parameter; archived residuals reproduced)', fixed))
out.append(score('K r^alpha Pi (two fitted parameters, least-squares slope)', free))
out.append(score('K r^alpha Pi (two fitted parameters, slope on the grid of the integrated law)', gridded))
out.append(score('K r^alpha Lambda_alpha (archived)', saved.res_M2))
out.append('')

# ---------------------------------------------------------------- (3) systems
sy = pd.read_csv(L + 'delay_law_systems_runs.csv')
out.append('systems out of sample (scalar-trained models):')
out.append(score('one-step model, all runs', sy.res_M0))
out.append(score('integrated law, all runs', sy.res_M2))
ml = sy[(sy.env == 'SciPy') & (sy.problem == 'ml')]
out.append(score('one-step model, SciPy Morris-Lecar', ml.res_M0))
out.append(score('integrated law, SciPy Morris-Lecar', ml.res_M2))
open('../results/summary_v18.txt', 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
