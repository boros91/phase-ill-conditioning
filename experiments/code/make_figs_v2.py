"""Figure for v2: (a) MATLAB R2023b runs of Paper A, delay at the event against tol*Pi;
(b) timing sensitivity along the orbit for the two applications.  -> ../figs/fig_s2_matlab_apps.pdf"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'legend.fontsize': 6.3,
                     'font.family': 'serif', 'mathtext.fontset': 'cm'})
R = '../results/'
d = pd.read_csv(R + 'matlab_pi_flame.csv'); d = d[np.isfinite(d.eta) & (d.completed == 1) & (d.eta > 0)]
s = pd.read_csv(R + 'matlab_pi_sir.csv'); s = s[np.isfinite(s.eta) & (s.eta > 0)]
fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.0))
ax = axs[0]
sty = {'ode15s_BDF': ('v', 'C3'), 'ode15s': ('^', 'C2'), 'ode23t': ('<', 'C5'), 'ode23tb': ('>', 'C6'),
       'ode23s': ('s', 'C1'), 'ode113': ('D', 'C8'), 'ode23': ('o', 'C0'), 'ode45': ('x', 'C4')}
for sv, (mk, c) in sty.items():
    a = d[d.solver == sv]
    ax.loglog(a.tolPi, a.eta, mk, ms=3, mfc='none', color=c, label=sv.replace('_', ' '))
    b = s[s.solver == sv]
    if len(b):
        ax.loglog(b.tolPi, b.eta, mk, ms=4, color=c)
xx = np.array([1e-12, 1e6])
ax.loglog(xx, xx, 'k-', lw=0.8); ax.loglog(xx, 10 * xx, 'k:', lw=0.6); ax.loglog(xx, 0.1 * xx, 'k:', lw=0.6)
ax.set_xlim(1e-12, 1e6); ax.set_ylim(1e-14, 1e6)
ax.set_xlabel(r'$\mathrm{RelTol}\cdot\Pi$'); ax.set_ylabel(r'$\eta=|D|/\Theta(T_e)$')
ax.set_title('(a) MATLAB R2023b, runs of Paper A\n(open: flame $q=2,3$; filled: SIR)')
ax.legend(loc='upper left', frameon=False, ncol=2); ax.grid(alpha=0.3)

ax = axs[1]
h = pd.read_csv(R + 'h2_theta.csv'); hc = pd.read_csv(R + 'h2_cases.csv')
for ratio, ls in ((1e-2, '-'), (1e-6, '--'), (1e-10, ':')):
    a = h[h.ratio == ratio]
    Te = hc.Te.iloc[0]
    ax.semilogy(a.t / Te, a.Theta / a.Theta_e, color='C3', ls=ls, lw=1.1,
                label=rf'H$_2$/air, $\mathrm{{AbsTol}}/\mathrm{{RelTol}}=10^{{{int(np.log10(ratio))}}}$')
m = pd.read_csv(R + 'ml_theta.csv'); mc = pd.read_csv(R + 'ml_cases.csv')
for Dl, ls in ((1e-1, ':'), (1e-2, '--'), (1e-3, '-')):
    a = m[m.Delta == Dl]; P = mc[mc.Delta == Dl].period.iloc[0]
    ax.semilogy(a.t / P, a.Theta / a.Theta_e, color='C0', ls=ls, lw=1.1,
                label=rf'Morris–Lecar, $\Delta=10^{{{int(np.log10(Dl))}}}$')
ax.set_xlabel(r'$t/T_e$ (H$_2$: ignition; Morris–Lecar: one period)')
ax.set_ylabel(r'$\Theta(t)/\Theta(T_e)$')
ax.set_title('(b) timing sensitivity in the two applications')
ax.set_xlim(0, 1); ax.set_ylim(0.1, 1e9)
ax.legend(loc='upper right', frameon=False); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig('../figs/fig_s2_matlab_apps.pdf'); plt.close(fig)
print('ok')
