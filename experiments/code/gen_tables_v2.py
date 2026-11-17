"""LaTeX tables and quoted numbers for v2 of the note: MATLAB evidence and the two applications.
Outputs: ../results/table_matlab.tex, table_h2.tex, table_ml.tex, summary_v2.txt"""
import numpy as np
import pandas as pd

R = '../results/'
out = []


def fmt(v, sig=2):
    if v is None or not np.isfinite(v):
        return '--'
    if v == 0:
        return '0'
    e = int(np.floor(np.log10(abs(v)))); m = v / 10**e
    if -2 <= e <= 2:
        return f'{v:.{max(sig, e + 1)}g}'
    return f'{m:.1f}\\cdot10^{{{e}}}'


# ---------------- MATLAB (existing runs) ----------------
d = pd.read_csv(R + 'matlab_pi_flame.csv'); d = d[np.isfinite(d.eta) & (d.completed == 1)]
s = pd.read_csv(R + 'matlab_pi_sir.csv')
order = ['ode15s_BDF', 'ode15s', 'ode23t', 'ode23tb', 'ode23s', 'ode113', 'ode23', 'ode45']
lines = [r'\begin{tabular}{lccc}', r'\toprule',
         r'solver & runs & median $c_M$ & 10\%--90\% \\', r'\midrule']
for sv in order:
    g = d[(d.solver == sv) & (d.tolPi >= 1e-3)].c
    lines.append(f"\\code{{{sv.replace('_', ' ')}}} & {len(g)} & ${fmt(g.median())}$ & ${fmt(g.quantile(.1))}$--${fmt(g.quantile(.9))}$ \\\\")
lines += [r'\bottomrule', r'\end{tabular}']
open(R + 'table_matlab.tex', 'w').write('\n'.join(lines) + '\n')
small = d[d.tolPi < 1e-3]; big = d[d.tolPi >= 1]
out.append(f'MATLAB flame: {len(d)} runs; tolPi<1e-3: {len(small)} runs max eta {small.eta.max():.3g}; '
           f'tolPi>=1: {len(big)} runs median eta {big.eta.median():.3g}; q values {sorted(d.q.unique())}')
for v in ('std', 'def', 'atol', 'log'):
    g = s[s.variant == v]
    out.append(f"MATLAB SIR {v}: Pi {g.Pi.min():.3g}-{g.Pi.max():.3g}; tolPi {g.tolPi.min():.3g}-{g.tolPi.max():.3g}; "
               f"max |D| {np.nanmax(np.abs(g.D)):.3g} days; no peak: {int(g.D.isna().sum())} of {len(g)}")

# ---------------- H2 ----------------
hc = pd.read_csv(R + 'h2_cases.csv'); hr = pd.read_csv(R + 'h2_runs.csv')
out.append(hc.to_string())
Te = hc.Te.iloc[0]
lines = [r'\begin{tabular}{lcccccc}', r'\toprule',
         r'& \multicolumn{2}{c}{$\AbsTol/\RelTol=10^{-2}$} & \multicolumn{2}{c}{$10^{-6}$} & \multicolumn{2}{c}{$10^{-10}$}\\',
         r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}',
         r'& \multicolumn{2}{c}{$\Pi=' + fmt(hc[hc.ratio == 1e-2].Pi.iloc[0]) + r'$} & \multicolumn{2}{c}{$\Pi=' +
         fmt(hc[hc.ratio == 1e-6].Pi.iloc[0]) + r'$} & \multicolumn{2}{c}{$\Pi=' + fmt(hc[hc.ratio == 1e-10].Pi.iloc[0]) + r'$}\\',
         r'solver, $\RelTol$ & $D/t_e$ & steps & $D/t_e$ & steps & $D/t_e$ & steps\\', r'\midrule']
for m in ('RK45', 'LSODA', 'BDF', 'Radau'):
    for tol in ((1e-4, 1e-6) if m == 'RK45' else (1e-4, 1e-6, 1e-8)):
        cells = []
        for ratio in (1e-2, 1e-6, 1e-10):
            q = hr[(hr.method == m) & (hr.tol == tol) & (hr.ratio == ratio)]
            if len(q) == 0:
                cells += ['--', '--']; continue
            q = q.iloc[0]
            if q.ok:
                cells += [f'${fmt(q.D / Te)}$', f'{int(q.steps)}']
            else:
                o = str(q.outcome)
                cells += ['no ignition' if o.startswith('no ignition') else 'failure', f'{int(q.steps)}']
        lines.append(f"{m if tol == 1e-4 else ''} & " if False else f"{m} $10^{{{int(np.log10(tol))}}}$ & " + ' & '.join(cells) + r' \\')
lines += [r'\bottomrule', r'\end{tabular}']
open(R + 'table_h2.tex', 'w').write('\n'.join(lines) + '\n')
out.append(hr[['ratio', 'method', 'tol', 'tolPi', 'D', 'rel_delay', 'eta', 'steps', 'outcome']].to_string())
ho = pd.read_csv(R + 'h2_orbit.csv'); out.append(ho.to_string())

# ---------------- Morris-Lecar ----------------
mc = pd.read_csv(R + 'ml_cases.csv'); mr = pd.read_csv(R + 'ml_runs.csv')
out.append(mc.to_string())
lines = [r'\begin{tabular}{lcccccc}', r'\toprule',
         r'& \multicolumn{2}{c}{$\Delta=10^{-1}$} & \multicolumn{2}{c}{$\Delta=10^{-2}$} & \multicolumn{2}{c}{$\Delta=10^{-3}$}\\',
         r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}',
         r'& \multicolumn{2}{c}{$P=' + f'{mc.period.iloc[0]:.0f}' + r',\ \Pi=' + fmt(mc.Pi.iloc[0]) + r'$} & \multicolumn{2}{c}{$P=' +
         f'{mc.period.iloc[1]:.0f}' + r',\ \Pi=' + fmt(mc.Pi.iloc[1]) + r'$} & \multicolumn{2}{c}{$P=' + f'{mc.period.iloc[2]:.0f}' +
         r',\ \Pi=' + fmt(mc.Pi.iloc[2]) + r'$}\\',
         r'solver, $\RelTol$ & $D_1$ & $D_{10}$ & $D_1$ & $D_{10}$ & $D_1$ & $D_{10}$\\', r'\midrule']
for variant, ms in (('standard', ('RK45', 'DOP853', 'LSODA', 'BDF', 'Radau')), ('time control', ('RK45', 'LSODA'))):
    if variant == 'time control':
        lines.append(r'\midrule')
        lines.append(r'\multicolumn{7}{l}{time as dependent variable, absolute tolerance $\RelTol\cdot\Theta(T_e)$ on $t$}\\')
    for m in ms:
        for tol in ((1e-4, 1e-6, 1e-8) if variant == 'standard' else (1e-4, 1e-6)):
            cells = []
            for Dl in (1e-1, 1e-2, 1e-3):
                q = mr[(mr.method == m) & (mr.tol == tol) & (mr.Delta == Dl) & (mr.variant == variant)]
                q = q.iloc[0]
                cells += [f'${fmt(q.D1)}$', f'${fmt(q.D10)}$']
            lines.append(f"{m} $10^{{{int(np.log10(tol))}}}$ & " + ' & '.join(cells) + r' \\')
lines += [r'\bottomrule', r'\end{tabular}']
open(R + 'table_ml.tex', 'w').write('\n'.join(lines) + '\n')
st = mr[mr.variant == 'standard'].copy()
st['c1'] = st.eta1 / st.tolPi
out.append('ML c = eta1/(tol Pi) by method (median): ' + st.groupby('method').c1.median().to_string())
ok = st[np.isfinite(st.D10_over_D1) & (np.abs(st.D1) > 1e-6)]
out.append(f'ML D10/D1: median {ok.D10_over_D1.median():.3g}, 25-75% {ok.D10_over_D1.quantile(.25):.3g}-{ok.D10_over_D1.quantile(.75):.3g}')
txt = '\n\n'.join(out)
open(R + 'summary_v2.txt', 'w').write(txt)
print(txt)
