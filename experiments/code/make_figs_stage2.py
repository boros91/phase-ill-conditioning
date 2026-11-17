"""Figures of the stage-2 note.
fig_s2_theta.pdf    : Theta / Theta_e along the orbit for the four mechanisms
fig_s2_index.pdf    : (a) measured relative error at the event vs tol*Pi,
                      (b) SIR: a posteriori estimate D_hat vs exact delay D
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import phase_common as pc

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'legend.fontsize': 6.5,
                     'font.family': 'serif', 'mathtext.fontset': 'cm'})
R = '../results/'

# ---------- Figure 1 ----------
fig, ax = plt.subplots(1, 1, figsize=(6.4, 2.9))
items = [(pc.flame_q(2, 1e-4), 1e-2, 'C0', '-', r"I: $y'=y^2(1-y)$, $\delta=10^{-4}$"),
         (pc.flame_q(2, 1e-4), 0.0, 'C0', ':', r"I: same, relative control only"),
         (pc.flame_q(3, 1e-3), 1e-2, 'C1', '-', r"I: $y'=y^3(1-y)$, $\delta=10^{-3}$"),
         (pc.ghost(1e-6), 1e-2, 'C2', '-', r"III: $x'=\varepsilon+x^2$, $\varepsilon=10^{-6}$"),
         (pc.logistic(1e-7), 1e-2, 'C3', '-', r"II: $x'=x(1-x)$, $x_0=10^{-7}$"),
         (pc.logistic(1e-1), 1e-2, 'C7', '-', r"none: $x'=x(1-x)$, $x_0=10^{-1}$")]
for p, ratio, col, ls, lab in items:
    x = p.grid
    t = np.array([p.T(v) for v in x]) / p.T(p.xe)
    th = p.Theta(x, ratio) / p.Theta(np.array([p.xe]), ratio)[0]
    ax.semilogy(t, th, color=col, ls=ls, lw=1.1, label=lab)
s = pd.read_csv(R + 'sir_theta.csv')
c = pd.read_csv(R + 'sir_cases.csv')
for I0, col in ((1e-6, 'C4'),):
    ss = s[s.I0 == I0]; cc = c[c.I0 == I0].iloc[0]
    ax.semilogy(ss.t / cc.Te, ss.Theta / cc.Theta_e, color=col, lw=1.1,
                label=r"II: SIR, $I_0=10^{-6}$ (adjoint)")
ax.set_xlabel(r'$t/T_e$ (fraction of the time to the event)')
ax.set_ylabel(r'$\Theta(t)/\Theta(T_e)$')
ax.set_xlim(0, 1.0); ax.set_ylim(0.3, 1e7)
ax.grid(alpha=0.3); ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), ncol=1, frameon=False)
fig.tight_layout(); fig.savefig('../figs/fig_s2_theta.pdf'); plt.close(fig)

# ---------- Figure 2 ----------
d = pd.read_csv(R + 'index_scalar.csv'); d = d[d.ok]
r = pd.read_csv(R + 'sir_runs.csv'); r = r[r.ok]
mk = {'RK45': 'o', 'DOP853': 's', 'LSODA': '^', 'BDF': 'v', 'Radau': 'D'}
cl = {'RK45': 'C0', 'DOP853': 'C1', 'LSODA': 'C2', 'BDF': 'C3', 'Radau': 'C4'}
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.9))
ax = axs[0]
for m in mk:
    a = d[d.method == m]
    ax.loglog(a.tolPi, a.eta, mk[m], ms=3, mfc='none', color=cl[m], label=m)
    b_ = r[r.method == m]
    ax.loglog(b_.tolPi, b_.eta, mk[m], ms=3.5, color=cl[m], alpha=0.8)
xx = np.array([1e-9, 1e4])
ax.loglog(xx, xx, 'k-', lw=0.8); ax.loglog(xx, 10 * xx, 'k:', lw=0.6); ax.loglog(xx, 0.1 * xx, 'k:', lw=0.6)
ax.set_xlim(1e-9, 1e4); ax.set_ylim(1e-13, 1e4)
ax.set_xlabel(r'$\mathrm{tol}\cdot\Pi$'); ax.set_ylabel(r'$\eta=|D|/\Theta(T_e)$')
ax.set_title('(a) relative error at the event vs. tol$\\cdot\\Pi$\n(open: scalar problems, filled: SIR)')
ax.legend(loc='upper left', frameon=False); ax.grid(alpha=0.3)
ax = axs[1]
rr = r[np.isfinite(r.Dhat)]
for m in mk:
    a = rr[rr.method == m]
    ax.loglog(np.abs(a.D), np.abs(a.Dhat), mk[m], ms=3.5, color=cl[m], label=m)
xx = np.array([1e-11, 1e2]); ax.loglog(xx, xx, 'k-', lw=0.8)
ax.set_xlabel(r'exact delay $|D|$'); ax.set_ylabel(r'estimate $|\hat D|=|\sum_k\lambda_k\cdot\ell_k|$')
ax.set_title('(b) SIR: a posteriori estimate of the delay')
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig('../figs/fig_s2_index.pdf'); plt.close(fig)
print('ok')
