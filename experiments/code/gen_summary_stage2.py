"""Numbers and LaTeX tables quoted in the stage-2 note (reads ../results)."""
import numpy as np
import pandas as pd

R = '../results/'
out = []
d = pd.read_csv(R + 'index_scalar.csv'); d = d[d.ok].copy()
s = pd.read_csv(R + 'sir_runs.csv'); s = s[s.ok].copy()
s['case'] = 'sir_I' + s.I0.map(lambda v: f'{v:.0e}')
d['c'] = d.eta / d.tolPi; s['c'] = s.eta / s.tolPi
d['kind'] = 'scalar'; s['kind'] = 'SIR'
a = pd.concat([d[['case', 'method', 'tol', 'Pi', 'tolPi', 'eta', 'c', 'kind']],
               s[['case', 'method', 'tol', 'Pi', 'tolPi', 'eta', 'c', 'kind']]])
a.to_csv(R + 'method_constants_all.csv', index=False)
rows = []
for kind in ('scalar', 'SIR'):
    for m in ('BDF', 'LSODA', 'RK45', 'DOP853', 'Radau'):
        x = a[(a.kind == kind) & (a.method == m) & (a.tolPi >= 1e-3)].c
        rows.append(dict(kind=kind, method=m, n=len(x), median=x.median(), q10=x.quantile(0.1),
                         q90=x.quantile(0.9), min=x.min(), max=x.max()))
mc = pd.DataFrame(rows); mc.to_csv(R + 'method_constants.csv', index=False)
out.append('method constants c = eta/(tol*Pi), runs with tol*Pi >= 1e-3:\n' + mc.to_string(float_format=lambda v: f'{v:.2g}'))

# runs with tol*Pi < 1e-3 versus >= 1: relative error at event
small = a[a.tolPi < 1e-3]; big = a[a.tolPi >= 1]
out.append(f"tol*Pi < 1e-3: max eta = {small.eta.max():.2e} ({len(small)} runs); "
           f"tol*Pi >= 1: median eta = {big.eta.median():.2f}, min {big.eta.min():.2e} ({len(big)} runs)")
fails = pd.read_csv(R + 'index_scalar.csv'); fails = fails[~fails.ok]
sf = pd.read_csv(R + 'sir_runs.csv'); sf = sf[~sf.ok]
out.append('runs without event: ' + '; '.join(f"{r.case} {r.method} {r.tol:.0e}" for r in fails.itertuples())
           + '; ' + '; '.join(f"SIR I0={r.I0:.0e} {r.method} {r.tol:.0e} (min I = {r.minI:.1e})" for r in sf.itertuples()))

# SIR estimator
e = s[np.isfinite(s.Dhat)].copy()
e['rel'] = np.abs(e.Dhat - e.D) / np.abs(e.D)
out.append(f"SIR estimator: {len(e)} runs; median rel. discrepancy {e.rel.median():.2e}; "
           f"fraction within 5%: {(e.rel < 0.05).mean():.2f}; within 25%: {(e.rel < 0.25).mean():.2f}")
out.append('largest discrepancies:\n' + e.sort_values('rel', ascending=False).head(6)[
    ['I0', 'method', 'tol', 'D', 'Dhat', 'rel']].to_string())

# Table: cases
c = pd.read_csv(R + 'index_scalar_cases.csv')
cs = pd.read_csv(R + 'sir_cases.csv')
out.append(c[['case', 'Pi', 'Pi_relative', 'x_argmax', 'Theta_e', 'T_event']].to_string())
out.append(cs[['I0', 'Te', 'Theta_e', 'Pi', 'Pi_relative', 'Pi_star', 'Pi_star_relative']].to_string())

# LaTeX table of Pi and measured eta (BDF and RK45 at tol 1e-4 and 1e-8) per case
mech = {'flame_q2': 'I', 'flame_q3': 'I', 'ghost': 'III', 'logistic': 'II'}
def fmt(v):
    if not np.isfinite(v):
        return '--'
    if v == 0:
        return '0'
    e_ = int(np.floor(np.log10(abs(v)))); m_ = v / 10**e_
    if -2 <= e_ <= 2:
        return f'{v:.3g}'
    return f'{m_:.1f}\\cdot10^{{{e_}}}'
lines = []
allc = list(c.case) + [f'sir_I{v:.0e}' for v in cs.I0]
for case in allc:
    if case.startswith('sir'):
        r0 = cs[cs.I0 == float(case[5:])].iloc[0]; Pi, Pir = r0.Pi, r0.Pi_relative
        lab = rf"SIR, $I_0=10^{{{int(np.log10(float(case[5:])))}}}$"; mk = 'II'
    else:
        r0 = c[c.case == case].iloc[0]; Pi, Pir = r0.Pi, r0.Pi_relative
        key = '_'.join(case.split('_')[:2]) if case.startswith('flame') else case.split('_')[0]
        mk = mech[key]
        par = case.split('_')[-1]
        val = float(par[1:]) if not par.startswith('x') else float(par[1:])
        ex = int(round(np.log10(val)))
        if case.startswith('flame_q2'): lab = rf"$y'=y^2(1-y)$, $\delta=10^{{{ex}}}$"
        elif case.startswith('flame_q3'): lab = rf"$y'=y^3(1-y)$, $\delta=10^{{{ex}}}$"
        elif case.startswith('ghost'): lab = rf"$x'=\varepsilon+x^2$, $\varepsilon=10^{{{ex}}}$"
        else: lab = rf"$x'=x(1-x)$, $x_0=10^{{{ex}}}$"
    cells = []
    for m in ('BDF', 'RK45'):
        for tol in (1e-4, 1e-8):
            q = a[(a.case == case) & (a.method == m) & (a.tol == tol)]
            cells.append('$' + fmt(q.eta.iloc[0]) + '$' if len(q) else 'no event')
    lines.append(f"{lab} & {mk} & ${fmt(Pi)}$ & ${fmt(Pir)}$ & " + ' & '.join(cells) + r' \\')
open(R + 'table_index.tex', 'w').write('\n'.join(lines) + '\n')

v = pd.read_csv(R + 'index_variables.csv')
vl = []
for (q, dl), g in v.groupby(['q', 'delta'], sort=False):
    vl.append(f"{q} & $10^{{{int(round(np.log10(dl)))}}}$ & " + ' & '.join('$' + fmt(x) + '$' for x in g.Pi) + r' \\')
open(R + 'table_variables.tex', 'w').write('\n'.join(vl) + '\n')
txt = '\n\n'.join(out)
open(R + 'summary_stage2.txt', 'w').write(txt)
print(txt)

# ---- LaTeX table of method constants ----
def f2(v):
    e_ = int(np.floor(np.log10(abs(v)))); m_ = v / 10**e_
    return f'{v:.2g}' if -2 <= e_ <= 1 else f'{m_:.1f}\\cdot10^{{{e_}}}'
tl = [r'\begin{tabular}{lcccc}', r'\toprule',
      r'& \multicolumn{2}{c}{scalar problems} & \multicolumn{2}{c}{SIR}\\',
      r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
      r'method & median & 10\%--90\% & median & 10\%--90\% \\', r'\midrule']
for m in ('BDF', 'LSODA', 'RK45', 'DOP853', 'Radau'):
    x = mc[(mc.kind == 'scalar') & (mc.method == m)].iloc[0]
    y = mc[(mc.kind == 'SIR') & (mc.method == m)].iloc[0]
    tl.append(f"{m} & ${f2(x['median'])}$ & ${f2(x.q10)}$--${f2(x.q90)}$ & "
              f"${f2(y['median'])}$ & ${f2(y.q10)}$--${f2(y.q90)}$ \\\\")
tl += [r'\midrule', f"runs & \\multicolumn{{2}}{{c}}{{{int(mc[mc.kind=='scalar'].n.sum())}}} & "
       f"\\multicolumn{{2}}{{c}}{{{int(mc[mc.kind=='SIR'].n.sum())}}}\\\\", r'\bottomrule', r'\end{tabular}']
open(R + 'table_constants.tex', 'w').write('\n'.join(tl) + '\n')

# ---- orbit test table ----
import os
if os.path.exists(R + 'orbit_test.csv'):
    o = pd.read_csv(R + 'orbit_test.csv')
    names = {'flame_q2_d1e-04': r"$y'=y^2(1-y)$, $\delta=10^{-4}$", 'flame_q3_d1e-02': r"$y'=y^3(1-y)$, $\delta=10^{-2}$",
             'ghost_e1e-06': r"$x'=\varepsilon+x^2$, $\varepsilon=10^{-6}$", 'logistic_x1e-01': r"$x'=x(1-x)$, $x_0=10^{-1}$",
             'logistic_x1e-07': r"$x'=x(1-x)$, $x_0=10^{-7}$", 'sir_I1e-06': r"SIR, $I_0=10^{-6}$", 'sir_I1e-08': r"SIR, $I_0=10^{-8}$",
             'lorenz': r"Lorenz (control)"}
    ol = [r'\begin{tabular}{llcccccc}', r'\toprule',
          r'& & \multicolumn{3}{c}{BDF} & \multicolumn{3}{c}{RK45}\\',
          r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}',
          r'problem & tol & $E_{\rm time}$ & $E_{\rm orbit}$ & $D_{12}$ & $E_{\rm time}$ & $E_{\rm orbit}$ & $D_{12}$\\', r'\midrule']
    for case, lab in names.items():
        for tol in (1e-4, 1e-6):
            cells = []
            for m in ('BDF', 'RK45'):
                q = o[(o.case == case) & (o.method == m) & (o.tol == tol)]
                if len(q) == 0:
                    cells += ['\\multicolumn{3}{c}{solver failure}']; continue
                q = q.iloc[0]
                eo = q.Eorbit if q.Eorbit >= 1e-15 else 0.0
                d12 = ('--' if case == 'lorenz' else ('no event' if not np.isfinite(q.D12_event)
                                                      else f"${fmt(q.D12_event)}$"))
                cells += [f"${fmt(q.Etime)}$", '$<10^{-15}$' if eo == 0 else f"${fmt(eo)}$", d12]
            ol.append(f"{lab if tol == 1e-4 else ''} & $10^{{{int(np.log10(tol))}}}$ & " + ' & '.join(cells) + r' \\')
    ol += [r'\bottomrule', r'\end{tabular}']
    open(R + 'table_orbit.tex', 'w').write('\n'.join(ol) + '\n')
    o['phase'] = o.verdict == 'phase error'
    print('\norbit test verdicts:\n', o.groupby(['case', 'verdict']).size().to_string())
    print('\nmax Eorbit/Etime for phase verdicts:', o[o.phase].ratio.max())
    print(o[['case', 'method', 'tol', 'Etime', 'Eorbit', 'ratio', 'D12_event', 'verdict']].to_string())
