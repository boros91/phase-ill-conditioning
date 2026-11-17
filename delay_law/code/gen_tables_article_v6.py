"""Tables and summary numbers added in article v6.
Inputs : ../results/delay_law_fit.csv, ideal_controller.csv, delay_law_lofo.csv, event_error_check.csv,
         delay_law_systems_runs.csv, h2_review.csv
Outputs: ../tables/table_v6_law.tex, table_v6_events.tex, table_v6_h2.tex, ../results/summary_article_v6.txt
"""
import numpy as np
import pandas as pd

R = '../results/'
F = pd.read_csv(R + 'delay_law_fit.csv')
IC = pd.read_csv(R + 'ideal_controller.csv')
LO = pd.read_csv(R + 'delay_law_lofo.csv')
EV = pd.read_csv(R + 'event_error_check.csv'); EV = EV[EV.ok == True].copy()
SY = pd.read_csv(R + 'delay_law_systems_runs.csv')
H2 = pd.read_csv(R + 'h2_review.csv')


def e(v, sig=2):
    if v is None or not np.isfinite(v):
        return '--'
    k = int(np.floor(np.log10(abs(v)))); m = v / 10**k
    if -2 <= k <= 0:
        return f'{v:#.{sig}g}'
    if 1 <= k <= 3:
        return f'{v:.0f}'
    return f'{m:.1f}\\cdot10^{{{k}}}'


# ---------------- Table: law, nominal exponents ----------------
nominal = {'BDF': (r'$1/2$--$5/6$', 'BDF 1--5'), 'LSODA': (r'$1/2$--$12/13$', 'Adams 1--12/BDF 1--5'),
           'RK45': ('1', '5(4) extrap.'), 'DOP853': ('1', '8(5,3) extrap.'),
           'Radau': ('5/4', '5, estimate 3'),
           'ode15s_BDF': (r'$1/2$--$5/6$', 'BDF 1--5'), 'ode15s': (r'$1/2$--$5/6$', 'NDF 1--5'),
           'ode23t': ('2/3', 'trapezoid'), 'ode23tb': ('2/3', 'TR-BDF2'), 'ode23s': ('2/3', 'Rosenbrock 2(3)'),
           'ode113': ('1', 'Adams 1--13 extrap.'), 'ode23': ('1', '3(2) extrap.'),
           'ode45': ('1', '5(4) extrap.')}
order = [('SciPy', s) for s in ('BDF', 'LSODA', 'RK45', 'DOP853', 'Radau')] + \
        [('MATLAB', s) for s in ('ode15s_BDF', 'ode15s', 'ode23t', 'ode23tb', 'ode23s', 'ode113', 'ode23', 'ode45')]
L = [r'\begin{tabular}{lllrcccccc}', r'\toprule',
     r'& & & & \multicolumn{2}{c}{exponent $\alpha$} & \multicolumn{3}{c}{spread of $\log_{10}$ residuals}\\',
     r'\cmidrule(lr){5-6}\cmidrule(lr){7-9}',
     r'& solver & method & runs & nominal & fitted & $c\,\tol\,\Pi$ & $K\tol^\alpha\Pi$ & $K\tol^\alpha\Lambda_\alpha$\\', r'\midrule']
last = None
for env, s in order:
    r = F[(F.env == env) & (F.solver == s)].iloc[0]
    nm, desc = nominal[s]
    L.append(f'{env if env != last else ""} & \\code{{{s.replace("_", " ")}}} & {desc} & {int(r.n)} & {nm} & {r.alpha_M2:.2f} & '
             f'{r.sd_M0:.2f} & {r.sd_M1:.2f} & {r.sd_M2:.2f}\\\\')
    last = env
L.append(r'\midrule')
for env in ('SciPy', 'MATLAB'):
    o = F[F.env == env]; w = o.n / o.n.sum()
    L.append(f'{env}, weighted mean & & & {int(o.n.sum())} & & & {np.sum(w*o.sd_M0):.2f} & {np.sum(w*o.sd_M1):.2f} & {np.sum(w*o.sd_M2):.2f}\\\\')
L += [r'\bottomrule', r'\end{tabular}']
open('../tables/table_v6_law.tex', 'w').write('\n'.join(L) + '\n')

# ---------------- Table: event-error definitions ----------------
EV['rel_end_root'] = np.abs(EV.D_end - EV.D_root) / np.abs(EV.D_root)
EV['rel_hat_end'] = np.abs(EV.D_hat - EV.D_end) / np.abs(EV.D_end)
EV['cancel'] = np.abs(EV.D_end) / EV.A
L = [r'\begin{tabular}{lrccccc}', r'\toprule',
     r'& & \multicolumn{2}{c}{$|D_{\rm end}-D_{\rm root}|/|D_{\rm root}|$} & \multicolumn{2}{c}{$|\hat D-D_{\rm end}|/|D_{\rm end}|$} & $|D_{\rm end}|/A$\\',
     r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-7}',
     r'problem & runs & median & $\le1\%$ & median & 90\% & median (10\%--90\%)\\', r'\midrule']
for p, lab in (('sir', 'SIR'), ('ml', 'Morris--Lecar')):
    G = EV[EV.problem == p]
    L.append(f'{lab} & {len(G)} & ${e(G.rel_end_root.median())}$ & {100*np.mean(G.rel_end_root<=0.01):.0f}\\% & '
             f'${e(G.rel_hat_end.median())}$ & ${e(G.rel_hat_end.quantile(0.9))}$ & '
             f'{G.cancel.median():.2f} ({G.cancel.quantile(0.1):.2f}--{G.cancel.quantile(0.9):.2f})\\\\')
L += [r'\bottomrule', r'\end{tabular}']
open('../tables/table_v6_events.tex', 'w').write('\n'.join(L) + '\n')

# ---------------- Table: H2 ----------------
tols = {'defaults, t_end=1 s': ('$10^{-3}$', '$10^{-6}$'), 'rtol 1e-4, scalar atol': ('$10^{-4}$', '$10^{-6}$'),
        'rtol 1e-4, species atol': ('$10^{-4}$', '$10^{-6}$, $10^{-14}$'),
        'Cantera defaults': ('$10^{-9}$', '$10^{-15}$'), 'Cantera rtol 1e-4': ('$10^{-4}$', '$10^{-6}$'),
        'Cantera rtol 1e-3': ('$10^{-3}$', '$10^{-6}$')}
L = [r'\begin{tabular}{llllrcccc}', r'\toprule',
     r'code & & $\RelTol$ & $\AbsTol$ & steps & $D/t_e$ & last $t$ (s) & last $T$ (K) & $\min_kY_k$\\', r'\midrule']
for r in H2[H2.code != 'RK45'].itertuples():
    rt, at = tols[r.group]
    note = 'defaults' if r.group in ('defaults, t_end=1 s', 'Cantera defaults') else ''
    code = 'ReactorNet' if r.code == 'ReactorNet' else r.code
    if r.ignited:
        L.append(f'{code} & {note} & {rt} & {at} & {r.steps} & ${e(r.rel_delay)}$ & & & \\\\')
    else:
        L.append(f'{code} & {note} & {rt} & {at} & {r.steps} & none & ${e(r.t_final)}$ & {r.T_final:.1f} & ${e(r.minY)}$\\\\')
L += [r'\bottomrule', r'\end{tabular}']
open('../tables/table_v6_h2.tex', 'w').write('\n'.join(L) + '\n')

# ---------------- summary ----------------
out = ['Numbers added in article v6', '']
for (pair, tol), G in IC.groupby(['pair', 'tol']):
    out.append(f'ideal controller {pair} tol={tol:.0e}: measured/pred {G.ratio_meas_pred.min():.4f}..{G.ratio_meas_pred.max():.4f} (n={len(G)})')
out.append('')
for fam, G in LO.groupby('held_out'):
    out.append(f'LOFO {fam}: n={len(G)} M0 median|res| {G.res_M0.abs().median():.2f} in1dec {np.mean(G.res_M0.abs()<1):.2f}; '
               f'M2 median|res| {G.res_M2.abs().median():.2f} in1dec {np.mean(G.res_M2.abs()<1):.2f}')
out.append(f'LOFO all: n={len(LO)} M0 median|res| {LO.res_M0.abs().median():.2f} in1dec {np.mean(LO.res_M0.abs()<1):.2f}; '
           f'M2 median|res| {LO.res_M2.abs().median():.2f} in1dec {np.mean(LO.res_M2.abs()<1):.2f}')
out.append('')
for p, G in EV.groupby('problem'):
    out.append(f'events {p}: n={len(G)} end/root median {G.rel_end_root.median():.2e} within1% {np.mean(G.rel_end_root<=0.01):.2f} '
               f'max {G.rel_end_root.max():.2e}; hat/end median {G.rel_hat_end.median():.2e} q90 {G.rel_hat_end.quantile(.9):.2e} '
               f'max {G.rel_hat_end.max():.2e}; sign agree {np.mean(np.sign(G.D_hat)==np.sign(G.D_end)):.2f}; '
               f'|D|/A median {G.cancel.median():.2f} q10 {G.cancel.quantile(.1):.2f} q90 {G.cancel.quantile(.9):.2f}')
    for s, H in G.groupby('solver'):
        out.append(f'   {s}: |D|/A median {H.cancel.median():.2f}')
    big = G.loc[G.rel_end_root.idxmax()]
    out.append(f'   largest end/root difference: {big.solver} tol={big.tol:.0e} par={big.par:.0e}: D_end={big.D_end:+.3e} D_root={big.D_root:+.3e}')
S = SY[SY.env == 'SciPy']
M = EV.merge(S[['problem', 'par', 'solver', 'tol', 'pred_M2', 'Theta_e']], on=['problem', 'par', 'solver', 'tol'])
for p, G in M.groupby('problem'):
    la = np.log10(G.A / G.Theta_e / G.pred_M2); ld = np.log10(np.abs(G.D_end) / G.Theta_e / G.pred_M2)
    out.append(f'law vs sums {p}: n={len(G)} median log10(A/pred) {la.median():+.2f}, median log10(|D|/pred) {ld.median():+.2f}')
out.append('')
for r in H2.itertuples():
    out.append(f'H2 {r.group} {r.code}: ignited={r.ignited} D/t_e={r.rel_delay:+.3e} steps={r.steps} t_final={r.t_final:.3e} '
               f'T_final={r.T_final:.2f} minY={r.minY:.3e}')
open(R + 'summary_article_v6.txt', 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
for t in ('table_v6_law', 'table_v6_events', 'table_v6_h2'):
    print(open(f'../tables/{t}.tex').read())
