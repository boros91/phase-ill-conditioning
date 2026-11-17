"""Table and numbers for v3: the new MATLAB R2023b runs of matlab/run_phase_index.m
(ghost, logistic, SIR, Morris-Lecar; eight solvers; RelTol 1e-4, 1e-6, 1e-8; AbsTol = RelTol/100).
Reads ../results/matlab_phase_index.csv (gen_matlab_phase_index.py) and matlab_pi_flame.csv.
Outputs: ../results/table_matlab_v3.tex, ../results/summary_v3.txt"""
import numpy as np
import pandas as pd

R = '../results/'
n = pd.read_csv(R + 'matlab_phase_index.csv')
old = pd.read_csv(R + 'matlab_pi_flame.csv'); old = old[np.isfinite(old.eta) & (old.completed == 1)]


def fmt(v, sig=2):
    if v is None or not np.isfinite(v):
        return '--'
    e = int(np.floor(np.log10(abs(v)))); m = v / 10**e
    if -2 <= e <= 2:
        return f'{v:.{max(sig, e + 1)}g}'
    return f'{m:.1f}\\cdot10^{{{e}}}'


order = ['ode15s_BDF', 'ode15s', 'ode23t', 'ode23tb', 'ode23s', 'ode113', 'ode23', 'ode45']
L = [r'\begin{tabular}{lcccccc}', r'\toprule',
     r'& \multicolumn{3}{c}{flame, runs of Paper A} & \multicolumn{3}{c}{ghost, logistic, SIR, Morris--Lecar}\\',
     r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',
     r'solver & runs & median & 10\%--90\% & runs & median & 10\%--90\%\\', r'\midrule']
for sv in order:
    a = old[(old.solver == sv) & (old.tolPi >= 1e-3)].c
    b = n[(n.solver == sv) & (n.tolPi >= 1e-3) & np.isfinite(n.c)].c
    L.append(f"\\code{{{sv.replace('_', ' ')}}} & {len(a)} & ${fmt(a.median())}$ & ${fmt(a.quantile(.1))}$--${fmt(a.quantile(.9))}$ & "
             f"{len(b)} & ${fmt(b.median())}$ & ${fmt(b.quantile(.1))}$--${fmt(b.quantile(.9))}$ \\\\")
L += [r'\bottomrule', r'\end{tabular}']
open(R + 'table_matlab_v3.tex', 'w').write('\n'.join(L) + '\n')

out = []
ok = n[np.isfinite(n.eta)]
out.append(f'new MATLAB runs: {len(n)}, with event {len(ok)}; MATLAB version from stats file')
s = ok[ok.tolPi < 1e-3]; b = ok[ok.tolPi >= 1]
out.append(f'tolPi<1e-3: {len(s)} runs, max eta {s.eta.max():.3g}; tolPi>=1: {len(b)} runs, median eta {b.eta.median():.3g}')
ne = n[~np.isfinite(n.eta)]
out.append('no event: ' + '; '.join(f'{r.problem} {r.param:.0e} {r.solver} RelTol {r.RelTol:.0e} tolPi {r.tolPi:.3g}' for r in ne.itertuples()))
for p in ('ghost', 'logistic', 'sir', 'ml'):
    x = ok[ok.problem == p]
    out.append(f'{p}: Pi {x.Pi.min():.3g}-{x.Pi.max():.3g}; max |D| {x.D.abs().max():.3g}; median c (tolPi>=1e-3) '
               f'{x[x.tolPi >= 1e-3].c.median():.3g}')
# class II scaling: delay depends on AbsTol/I0 only
sir = ok[ok.problem == 'sir'].copy(); sir['a_over_x0'] = sir.AbsTol / sir.param
g = sir.groupby(['solver', 'a_over_x0']).D.agg(['min', 'max', 'count'])
g = g[g['count'] > 1]
g['spread'] = (g['max'] - g['min']) / g[['max', 'min']].abs().max(axis=1)
out.append('SIR: groups with equal AbsTol/I0 (>=2 runs): relative spread of D, median '
           f'{g.spread.median():.3g}, max {g.spread.max():.3g} over {len(g)} groups')
big = g[g.index.get_level_values('a_over_x0') >= 1e-2]
out.append(f'SIR, AbsTol/I0 >= 1e-2: relative spread of D within groups: max {big.spread.max():.3g} over {len(big)} groups')
out.append(g.to_string())
m = ok[ok.problem == 'ml']
r = m.D10 / m.D
out.append(f'ML: D10/D1 median {r.median():.3g}, quartiles {r.quantile(.25):.3g}-{r.quantile(.75):.3g}; worst D10 '
           f'{m.D10.abs().max():.3g} ms ({m.loc[m.D10.abs().idxmax(), "solver"]}, Delta {m.loc[m.D10.abs().idxmax(), "param"]}, '
           f'RelTol {m.loc[m.D10.abs().idxmax(), "RelTol"]})')
sgn = ok[ok.solver.isin(['ode15s', 'ode15s_BDF', 'ode23t', 'ode23tb', 'ode23s']) & (ok.eta > 1e-6) & (ok.problem != 'ml')]
out.append(f'implicit codes (eta > 1e-6, not ML): {int((sgn.D < 0).sum())} of {len(sgn)} early')
txt = '\n\n'.join(out)
open(R + 'summary_v3.txt', 'w').write(txt)
print(txt)
