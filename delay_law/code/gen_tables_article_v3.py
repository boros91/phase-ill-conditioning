"""Tables and summary numbers for article v3.
Inputs : ../results/delay_law_fit.csv, delay_law_fit_runs.csv, delay_law_systems_runs.csv, delay_law_slopes.csv,
         new_checks_A.csv, new_checks_B.csv, new_checks_C.csv, new_checks_D.csv
Outputs: ../tables/table_v3_law.tex, ../tables/table_v3_systems.tex, ../tables/table_v3_shift.tex,
         ../results/summary_article_v3.txt
"""
import numpy as np
import pandas as pd

R = '../results/'
F = pd.read_csv(R + 'delay_law_fit.csv')
FR = pd.read_csv(R + 'delay_law_fit_runs.csv'); FR = FR[FR.res_M0.notna()]
SY = pd.read_csv(R + 'delay_law_systems_runs.csv')
SL = pd.read_csv(R + 'delay_law_slopes.csv')
A = pd.read_csv(R + 'new_checks_A.csv'); B = pd.read_csv(R + 'new_checks_B.csv')
C = pd.read_csv(R + 'new_checks_C.csv'); D = pd.read_csv(R + 'new_checks_D.csv')


def e(v, sig=2):
    if not np.isfinite(v):
        return '--'
    k = int(np.floor(np.log10(abs(v)))); m = v / 10**k
    if -2 <= k <= 0:
        return f'{v:#.{sig}g}'
    if 1 <= k <= 3:
        return f'{v:.0f}'
    return f'{m:.1f}\\cdot10^{{{k}}}'


# theoretical exponent alpha = q/(phat+1): q order of the propagated solution, phat order of the error estimate
theory = {'ode45': '1', 'ode23': '1', 'RK45': '1', 'DOP853': '1', 'Radau': '5/4',
          'ode23s': '2/3', 'ode23t': '2/3', 'ode23tb': '2/3',
          'ode15s': r'$\le 5/6$', 'ode15s_BDF': r'$\le 5/6$', 'BDF': r'$\le 5/6$', 'LSODA': r'$\le 5/6$',
          'ode113': r'$\le 13/14$'}
order = [('SciPy', s) for s in ('BDF', 'LSODA', 'RK45', 'DOP853', 'Radau')] + \
        [('MATLAB', s) for s in ('ode15s_BDF', 'ode15s', 'ode23t', 'ode23tb', 'ode23s', 'ode113', 'ode23', 'ode45')]
L = [r'\begin{tabular}{llrccccc}', r'\toprule',
     r'& & & \multicolumn{2}{c}{exponent $\alpha$} & \multicolumn{3}{c}{spread of $\log_{10}$ residuals}\\',
     r'\cmidrule(lr){4-5}\cmidrule(lr){6-8}',
     r'& solver & runs & fitted & $q/(\hat p+1)$ & $c\,\tol\,\Pi$ & $K\tol^\alpha\Pi$ & $K\tol^\alpha\Lambda_\alpha$\\', r'\midrule']
last = None
for env, s in order:
    r = F[(F.env == env) & (F.solver == s)].iloc[0]
    L.append(f'{env if env != last else ""} & \\code{{{s.replace("_", " ")}}} & {int(r.n)} & {r.alpha_M2:.2f} & {theory[s]} & '
             f'{r.sd_M0:.2f} & {r.sd_M1:.2f} & {r.sd_M2:.2f}\\\\')
    last = env
for env in ('SciPy', 'MATLAB'):
    o = F[F.env == env]; w = o.n / o.n.sum()
    L.append(r'\midrule' if env == 'SciPy' else '')
    L.append(f'{env}, weighted mean & & {int(o.n.sum())} & & & {np.sum(w*o.sd_M0):.2f} & {np.sum(w*o.sd_M1):.2f} & {np.sum(w*o.sd_M2):.2f}\\\\')
L += [r'\bottomrule', r'\end{tabular}']
open('../tables/table_v3_law.tex', 'w').write('\n'.join(l for l in L if l) + '\n')

# systems, out of sample
L = [r'\begin{tabular}{llrcccc}', r'\toprule',
     r'& & & \multicolumn{2}{c}{$c\,\tol\,\Pi$} & \multicolumn{2}{c}{$K\tol^\alpha\Lambda_\alpha$}\\',
     r'\cmidrule(lr){4-5}\cmidrule(lr){6-7}',
     r'& problem & runs & mean & within 1 decade & mean & within 1 decade\\', r'\midrule']
for (env, prob), G in SY.groupby(['env', 'problem']):
    L.append(f'{env} & {"SIR" if prob=="sir" else "Morris--Lecar"} & {len(G)} & ${G.res_M0.mean():+.2f}$ & {100*np.mean(G.res_M0.abs()<1):.0f}\\% & '
             f'${G.res_M2.mean():+.2f}$ & {100*np.mean(G.res_M2.abs()<1):.0f}\\%\\\\')
L.append(r'\midrule')
L.append(f'all & & {len(SY)} & ${SY.res_M0.mean():+.2f}$ & {100*np.mean(SY.res_M0.abs()<1):.0f}\\% & '
         f'${SY.res_M2.mean():+.2f}$ & {100*np.mean(SY.res_M2.abs()<1):.0f}\\%\\\\')
L += [r'\bottomrule', r'\end{tabular}']
open('../tables/table_v3_systems.tex', 'w').write('\n'.join(L) + '\n')

# shifted bottleneck
L = [r'\begin{tabular}{rcccccc}', r'\toprule',
     r'centre $a$ & $\Pi$ & $\Theta(T_e)$ & solver & steps & $D/t_*$ & $\eta$\\', r'\midrule']
for a, G in B.groupby('a'):
    for i, (_, r) in enumerate(G.iterrows()):
        L.append(f'{(f"{a:g}" if i == 0 else "")} & {("$"+e(r.Pi)+"$") if i == 0 else ""} & {("$"+e(r.Theta_e)+"$") if i == 0 else ""} & '
                 f'{r.solver} & {int(r.steps)} & ${e(r.D_over_t)}$ & ${e(r.eta)}$\\\\')
L += [r'\bottomrule', r'\end{tabular}']
open('../tables/table_v3_shift.tex', 'w').write('\n'.join(L) + '\n')

# ---------------- summary ----------------
out = ['Numbers quoted in article v3', '']
out.append(f'scalar runs used in the fit: {len(FR)} (SciPy {(FR.env=="SciPy").sum()}, MATLAB {(FR.env=="MATLAB").sum()})')
for env in ('SciPy', 'MATLAB'):
    o = F[F.env == env]; w = o.n / o.n.sum()
    out.append(f'{env}: weighted sd M0 {np.sum(w*o.sd_M0):.3f} M1 {np.sum(w*o.sd_M1):.3f} M2 {np.sum(w*o.sd_M2):.3f} M3 {np.sum(w*o.sd_M3):.3f}')
out.append(f'M2 better than M0 for {(F.sd_M2 < F.sd_M0).sum()} of {len(F)} solvers; exceptions: {list(F[F.sd_M2 >= F.sd_M0].solver)}')
out.append(f'fitted alpha range: {F.alpha_M2.min():.2f}..{F.alpha_M2.max():.2f}')
for s in ('ode23s', 'ode23t', 'ode23tb', 'ode15s', 'ode15s_BDF', 'BDF', 'LSODA', 'ode113', 'ode23', 'ode45', 'RK45', 'DOP853', 'Radau'):
    r = F[F.solver == s].iloc[0]; out.append(f'  {s}: alpha {r.alpha_M2:.2f}, K {r.K_M2:.3g}, sd M0 {r.sd_M0:.2f}, sd M2 {r.sd_M2:.2f}')
out.append('')
out.append(f'slopes: groups {len(SL)}, median |err| Pi {SL.err_Pi.abs().median():.3f}, Lambda {SL.err_Lambda.abs().median():.3f}; '
           f'|err|<0.1 Pi {np.mean(SL.err_Pi.abs()<0.1):.2f}, Lambda {np.mean(SL.err_Lambda.abs()<0.1):.2f}')
for p, G in SL.groupby('problem'):
    out.append(f'  {p}: groups {len(G)} median slope eta {G.slope_eta.median():+.2f}, Pi {G.slope_Pi.median():+.2f}, '
               f'Lambda {G.slope_Lambda.median():+.2f}; median|err| Pi {G.err_Pi.abs().median():.2f} Lambda {G.err_Lambda.abs().median():.2f}')
out.append('')
out.append(f'systems out of sample: n {len(SY)}; M0 median|res| {SY.res_M0.abs().median():.2f} within 1 dec {np.mean(SY.res_M0.abs()<1):.2f}; '
           f'M2 median|res| {SY.res_M2.abs().median():.2f} within 1 dec {np.mean(SY.res_M2.abs()<1):.2f}')
for (env, prob), G in SY.groupby(['env', 'problem']):
    out.append(f'  {env} {prob}: n {len(G)} M0 mean {G.res_M0.mean():+.2f} M2 mean {G.res_M2.mean():+.2f}')
out.append('')
Aok = A[A.ok]
out.append(f'pure relative control: runs {len(A)}, ok {len(Aok)}; t* {Aok.tstar.min():.3g}..{Aok.tstar.max():.3g}')
for m, G in Aok.groupby('solver'):
    r = np.log10(np.abs(G.D_over_t) / G.pred_abs)
    fl = G[G.problem.str.startswith('flame')]
    out.append(f'  {m}: |D|/t* {np.abs(G.D_over_t).min():.2e}..{np.abs(G.D_over_t).max():.2e}; flame only {np.abs(fl.D_over_t).min():.2e}..{np.abs(fl.D_over_t).max():.2e}; '
               f'log10(meas/pred) {r.min():+.2f}..{r.max():+.2f} mean {r.mean():+.2f}; within factor 3: {np.mean(np.abs(r)<np.log10(3)):.2f}')
out.append('')
for _, r in B.iterrows():
    out.append(f'shift a={r.a:g} {r.solver}: D/t* {r.D_over_t:+.3e} eta {r.eta:.3e} Pi {r.Pi:.4g} Theta_e {r.Theta_e:.4g} pred {r.pred_eta:.3e} steps {r.steps}')
out.append('')
for _, r in C.iterrows():
    out.append(f'fixed N=2000: delta {r.delta:.0e} {r.method}: h {r.h:g} D {r.D:+.4e} D*delta^2 {r.D_delta2:.3e}')
out.append('')
for _, r in D.iterrows():
    out.append(f'coupled eps {r.eps:.0e}: eps*dtau/dx0 {r.eps*r.dtau_dx0_adj:.4f} (fd {r.eps*r.dtau_dx0_fd:.4f}); eps*dtau/dz0 {r.eps*r.dtau_dz0_adj:.4f} '
               f'(fd {r.eps*r.dtau_dz0_fd:.4f}); eps^2*dtau/dz0 {r.eps**2*r.dtau_dz0_adj:.4f}; Pi {r.Pi:.4g} Pi_* {r.Pi_star:.4g}')
open(R + 'summary_article_v3.txt', 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
