"""Figure for article v3: (a) delay against the one-step model c*tol*Pi, (b) against the integrated law
K*tol^alpha*Lambda_alpha (both fitted per solver on the scalar runs), (c) pure relative control: |D|/t_*
against the passage time t_*.  -> ../figs/fig_v3_delay_law.pdf
Inputs: ../results/delay_law_fit_runs.csv, ../results/delay_law_fit.csv, ../results/new_checks_A.csv"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'legend.fontsize': 6.5,
                     'font.family': 'serif', 'mathtext.fontset': 'cm'})
COL = {'flame2': '#2a78d6', 'flame3': '#eb6834', 'ghost': '#1baf7a', 'logistic': '#eda100'}
MK = {'flame2': 'o', 'flame3': 's', 'ghost': '^', 'logistic': 'D'}
LAB = {'flame2': r"$y'=y^2(1-y)$", 'flame3': r"$y'=y^3(1-y)$", 'ghost': r"$x'=\varepsilon+x^2$",
       'logistic': 'logistic'}
R = pd.read_csv('../results/delay_law_fit_runs.csv'); R = R[R.res_M0.notna()]
F = pd.read_csv('../results/delay_law_fit.csv').set_index(['env', 'solver'])
pred0, pred2 = [], []
for _, r in R.iterrows():
    f = F.loc[(r.env, r.solver)]
    pred2.append(f.K_M2 * r.tol**f.alpha_M2 * r.Lambda)
g = R.groupby(['env', 'solver'])
c = g.apply(lambda G: 10**np.mean(np.log10(G.eta / (G.tol * G.Pi))))
R['pred0'] = [c.loc[(r.env, r.solver)] * r.tol * r.Pi for _, r in R.iterrows()]
R['pred2'] = pred2

fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.6))
for ax, col, ttl in ((axs[0], 'pred0', r'(a) one-step model $c\,\mathrm{tol}\,\Pi$'),
                     (axs[1], 'pred2', r'(b) integrated law $K\,\mathrm{tol}^{\alpha}\Lambda_\alpha$')):
    for p in COL:
        a = R[R.problem == p]
        ax.loglog(a[col], a.eta, MK[p], ms=2.6, mfc='none', mew=0.6, color=COL[p], label=LAB[p])
    xx = np.array([1e-14, 1e6])
    ax.loglog(xx, xx, '-', color='#0b0b0b', lw=0.8)
    for k in (10, 0.1):
        ax.loglog(xx, k * xx, ':', color='#52514e', lw=0.6)
    ax.set_xlim(1e-12, 1e5); ax.set_ylim(1e-12, 1e5)
    ax.set_xlabel('prediction'); ax.set_title(ttl)
    ax.grid(alpha=0.25, lw=0.4)
    sd = np.log10(R.eta / R[col]).std()
    ax.text(0.97, 0.04, f'sd of log$_{{10}}$ ratio: {sd:.2f}', transform=ax.transAxes, ha='right', fontsize=6.5)
axs[0].set_ylabel(r'$\eta=|D|/\Theta(T_e)$')
axs[0].legend(loc='upper left', frameon=False)

ax = axs[2]
A = pd.read_csv('../results/new_checks_A.csv'); A = A[A.ok]
for m, mk, cc in (('BDF', 'v', '#2a78d6'), ('RK45', 'o', '#eb6834'), ('Radau', 's', '#1baf7a')):
    a = A[A.solver == m]
    ax.loglog(a.tstar, np.abs(a.D_over_t), mk, ms=3.5, mfc='none', mew=0.8, color=cc, label=m)
ax.set_xlabel(r'passage time $t_*$'); ax.set_ylabel(r'$|D|/t_*$')
ax.set_title('(c) relative control only')
ax.set_ylim(1e-10, 1e-3); ax.grid(alpha=0.25, lw=0.4)
ax.legend(loc='upper right', frameon=False, ncol=3)
fig.tight_layout(w_pad=0.6)
fig.savefig('../figs/fig_v3_delay_law.pdf')
fig.savefig('../figs/fig_v3_delay_law.png', dpi=150)
print('sd', np.log10(R.eta / R.pred0).std(), np.log10(R.eta / R.pred2).std(), len(R))
