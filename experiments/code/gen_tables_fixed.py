"""Table and summary numbers for the fixed-step experiment (article v2).
Inputs:  ../results/fixed_step.csv, fixed_step_flame.csv, fixed_step_heun_pred.csv,
         index_scalar.csv, index_scalar_cases.csv, sir_runs.csv, sir_cases.csv, ml_runs.csv
Outputs: ../tables/table_fixed.tex, ../results/summary_fixed.txt
"""
import numpy as np
import pandas as pd

R = '../results/'
fx = pd.concat([pd.read_csv(R + 'fixed_step.csv'), pd.read_csv(R + 'fixed_step_flame.csv')], ignore_index=True)
fx['event'] = fx['event'].astype(str)
hp = pd.read_csv(R + 'fixed_step_heun_pred.csv'); hp['event'] = hp['event'].astype(str)
idx = pd.read_csv(R + 'index_scalar.csv')
idc = pd.read_csv(R + 'index_scalar_cases.csv')
sir = pd.read_csv(R + 'sir_runs.csv'); sirc = pd.read_csv(R + 'sir_cases.csv')
mlr = pd.read_csv(R + 'ml_runs.csv')


def fmt(v, sig=2):
    if v is None or not np.isfinite(v):
        return '--'
    if v == 0:
        return '0'
    e = int(np.floor(np.log10(abs(v)))); m = v / 10**e
    if 1 <= e <= 2:
        return f'{v:.0f}'
    if -2 <= e <= 0:
        return f'{v:#.{sig}g}'
    return f'{m:.1f}\\cdot10^{{{e}}}'


def M(v):
    s = fmt(v)
    return '--' if s == '--' else f'${s}$'


def pw(v):
    e = int(round(np.log10(v)))
    return f'10^{{{e}}}'


def get(problem, param, event, method, h, col='D'):
    s = fx[(fx.problem == problem) & np.isclose(fx.param, param) & (fx.event == event) & (fx.method == method)
           & np.isclose(fx.h, h)]
    return float(s[col].iloc[0]) if len(s) else np.nan


def heun(problem, param, event):
    s = hp[(hp.problem == problem) & np.isclose(hp.param, param) & (hp.event == event)]
    return float(s.pred_heun.iloc[0]) if len(s) else np.nan


def bdf_scalar(case):
    s = idx[(idx.case == case) & (idx.method == 'BDF') & np.isclose(idx.tol, 1e-4)]
    if not len(s):
        return np.nan, np.nan, False
    r = s.iloc[0]
    return float(r.D), int(r.steps), bool(r.ok) and np.isfinite(r.D)


rows = []   # (label, param, h, event, steps, EE, predEE, IE, Heun, predHeun, RK4, bdfD, bdfsteps, bdfok)
for d in (1e-2, 1e-4, 1e-6):
    D, n, ok = bdf_scalar(f'flame_q2_d{d:.0e}')
    rows.append(("$y'=y^2-y^3$", rf'$\delta={pw(d)}$', 0.25, '0.5', 'flame', d, D, n, ok))
for x0 in (1e-1, 1e-3, 1e-5, 1e-7):
    D, n, ok = bdf_scalar(f'logistic_x{x0:.0e}')
    rows.append(('logistic', rf'$x_0={pw(x0)}$', 0.25, '0.5', 'logistic', x0, D, n, ok))
for xe in ('0.1', '1.0'):
    for e in (1e-2, 1e-8):
        D, n, ok = bdf_scalar(f'ghost_e{e:.0e}') if xe == '1.0' else (np.nan, np.nan, True)
        rows.append((rf'ghost, $x_e={float(xe):g}$', rf'$\varepsilon={pw(e)}$', 0.25, xe, 'ghost', e, D, n, ok))
for I0 in (1e-2, 1e-4, 1e-6, 1e-8):
    s = sir[np.isclose(sir.I0, I0) & (sir.method == 'BDF') & np.isclose(sir.tol, 1e-4)].iloc[0]
    ev = str(fx[(fx.problem == 'sir')].event.iloc[0])
    rows.append(('SIR', rf'$I_0={pw(I0)}$', 0.1, ev, 'sir', I0, float(s.D), int(s.steps),
                 bool(s.ok) and np.isfinite(s.D)))
for Dl in (1e-1, 1e-2):
    s = mlr[np.isclose(mlr.Delta, Dl) & (mlr.method == 'BDF') & np.isclose(mlr.tol, 1e-4) & (mlr.variant == 'standard')].iloc[0]
    rows.append(('Morris--Lecar', rf'$\Delta={pw(Dl)}$', 0.05, 'spike 10', 'ml', Dl, float(s.D10),
                 int(s.steps), np.isfinite(s.D10)))

L = [r'\begin{tabular}{llrcccccccc}', r'\toprule',
     r'& & & \multicolumn{2}{c}{explicit Euler} & implicit & \multicolumn{2}{c}{Heun} & RK4 & \multicolumn{2}{c}{BDF, $\tol=10^{-4}$}\\',
     r'\cmidrule(lr){4-5}\cmidrule(lr){6-6}\cmidrule(lr){7-8}\cmidrule(lr){9-9}\cmidrule(lr){10-11}',
     r'problem & & steps & $D$ & theory & $D$ & $D$ & theory & $D$ & $D$ & steps\\', r'\midrule']
last = None
summ = []
for lab, par, h, ev, prob, p, bD, bn, bok in rows:
    n = get(prob, p, ev, 'EE', h, 'steps')
    ee, pe = get(prob, p, ev, 'EE', h), get(prob, p, ev, 'EE', h, 'pred_euler')
    ie = get(prob, p, ev, 'IE', h)
    he, ph = get(prob, p, ev, 'Heun', h), (heun(prob, p, ev) if prob in ('flame', 'logistic', 'ghost') else np.nan)
    rk = get(prob, p, ev, 'RK4', h)
    if lab != last and last is not None:
        L.append(r'\addlinespace[2pt]')
    first = lab if lab != last else ''
    last = lab
    bcol = (f'${fmt(bD)}$' if bok else 'no event') if np.isfinite(bD) or not bok else '--'
    bsteps = str(bn) if np.isfinite(bn) else '--'
    rks = r'$\lesssim10^{-7}$' if prob == 'ml' and abs(rk) < 1e-7 else M(rk)
    L.append(f'{first} & {par} & {int(n)} & {M(ee)} & {M(pe)} & {M(ie)} & {M(he)} & {M(ph)} & '
             f'{rks} & {bcol} & {bsteps}\\\\')
    summ.append(f'{prob:8s} {p:.0e} ev={ev:8s} h={h}: n={int(n)} EE={ee:+.4e} pred={pe:+.4e} ratio={ee/pe if pe else np.nan:.4f} '
                f'IE/-EE={(-ie/ee) if np.isfinite(ie) else np.nan:.4f} Heun={he:+.4e} predHeun={ph:+.4e} '
                f'ratioH={he/ph if np.isfinite(ph) else np.nan:.4f} RK4={rk:+.3e} BDF1e-4: D={bD} steps={bn} ok={bok}')
L += [r'\bottomrule', r'\end{tabular}']
open('../tables/table_fixed.tex', 'w').write('\n'.join(L) + '\n')

# ---------------- summary numbers ----------------
out = ['Fixed-step experiment: summary numbers', '']
out += summ
out.append('')
# h-dependence (SIR, ML): ratio D(h)/D(h/2)
for prob, pars, hs, ev in (('sir', (1e-2, 1e-4, 1e-6, 1e-8), (0.1, 0.05), None), ('ml', (1e-1, 1e-2), (0.05, 0.025), 'spike 10')):
    for p in pars:
        for m in ('EE', 'Heun', 'RK4'):
            s = fx[(fx.problem == prob) & np.isclose(fx.param, p) & (fx.method == m)]
            if ev:
                s = s[s.event == ev]
            a = float(s[np.isclose(s.h, hs[0])].D.iloc[0]); b = float(s[np.isclose(s.h, hs[1])].D.iloc[0])
            out.append(f'h-halving {prob} {p:.0e} {m:4s}: D(h)={a:+.4e} D(h/2)={b:+.4e} ratio={a/b:.3f}')
out.append('')
for p in (1e-1, 1e-2):
    for h in (0.05, 0.025):
        for m in ('EE', 'Heun'):
            s = fx[(fx.problem == 'ml') & np.isclose(fx.param, p) & (fx.method == m) & np.isclose(fx.h, h)]
            d1 = float(s[s.event == 'spike 1'].D.iloc[0]); d10 = float(s[s.event == 'spike 10'].D.iloc[0])
            out.append(f'ML {p:.0e} h={h} {m}: D1={d1:+.4e} D10={d10:+.4e} D10/D1={d10/d1:.3f}')
out.append('')
# ratios of EE to prediction over all rows with nonzero prediction
ee = fx[(fx.method == 'EE') & np.isfinite(fx.pred_euler) & (fx.pred_euler != 0)]
r = ee.D / ee.pred_euler
for prob in ee.problem.unique():
    rr = r[ee.problem == prob]
    out.append(f'EE/pred {prob:8s}: n={len(rr)} min={rr.min():.4f} max={rr.max():.4f}')
ie = fx[(fx.method == 'IE')]
for prob in ie.problem.unique():
    s = ie[ie.problem == prob]
    rr = []
    for _, q in s.iterrows():
        e = fx[(fx.problem == prob) & np.isclose(fx.param, q.param) & (fx.event == q.event) & (fx.method == 'EE')].D.iloc[0]
        rr.append(-q.D / e)
    out.append(f'IE/(-EE) {prob:8s}: min={min(rr):.4f} max={max(rr):.4f}')
out.append('')
# eta for fixed-step EE (SciPy error scale of the adaptive runs, AbsTol/RelTol = 1e-2)
for case, prob, p in (('flame_q2_d1e-06', 'flame', 1e-6), ('logistic_x1e-07', 'logistic', 1e-7)):
    th = float(idc[idc.case == case].Theta_e.iloc[0])
    out.append(f'eta EE {case}: D={get(prob, p, "0.5", "EE", 0.25):.4f} Theta_e={th} eta={get(prob, p, "0.5", "EE", 0.25)/th:.3f}')
th = float(sirc[np.isclose(sirc.I0, 1e-8)].Theta_e.iloc[0])
out.append(f'eta EE sir 1e-08 h=0.1: Theta_e={th:.3f} eta={get("sir", 1e-8, fx[fx.problem=="sir"].event.iloc[0], "EE", 0.1)/th:.4f}')
th = float(mlr[np.isclose(mlr.Delta, 1e-1)].Theta_e.iloc[0])
out.append(f'eta EE ml 1e-01 spike10 h=0.05: Theta_e={th:.4f} eta={get("ml", 1e-1, "spike 10", "EE", 0.05)/th:.4f}')
open(R + 'summary_fixed.txt', 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
print(open('../tables/table_fixed.tex').read())
