"""Figures for article v11 (SIAM format), drawn at the printed width of the SIAM text block (5.125 in),
so that the lettering is printed at 7-8 pt. Same data and content as the earlier figures:
  ../figs/fig_v11_theta.pdf       (as fig_s2_theta.pdf, stage-2 make_figs_stage2.py, Figure 1)
  ../figs/fig_v11_index.pdf       (as fig_s2_index.pdf, stage-2 make_figs_stage2.py, Figure 2)
  ../figs/fig_v11_delay_law.pdf   (as fig_v3_delay_law.pdf, make_fig_delay_law.py)
  ../figs/fig_v11_matlab_apps.pdf (as fig_s2_matlab_apps_v3.pdf, stage-2 make_figs_v3.py; legend text
                                   'runs of Paper A' replaced by 'companion-paper runs')
Inputs: ../../experiments/results/{sir_theta,sir_cases,index_scalar,sir_runs,matlab_pi_flame,matlab_pi_sir,
        matlab_phase_index,h2_theta,h2_cases,ml_theta,ml_cases}.csv; ../results/{delay_law_fit_runs,
        delay_law_fit,new_checks_A}.csv; phase_common.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import phase_common as pc

W = 5.12
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
                     'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                     'font.family': 'serif', 'mathtext.fontset': 'cm'})
S = '../../experiments/results/'
R = '../results/'

# ---------------- timing sensitivity along the orbit ----------------
fig, ax = plt.subplots(1, 1, figsize=(W, 3.3))
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
s = pd.read_csv(S + 'sir_theta.csv'); c = pd.read_csv(S + 'sir_cases.csv')
ss = s[s.I0 == 1e-6]; cc = c[c.I0 == 1e-6].iloc[0]
ax.semilogy(ss.t / cc.Te, ss.Theta / cc.Theta_e, color='C4', lw=1.1, label=r"II: SIR, $I_0=10^{-6}$ (adjoint)")
ax.set_xlabel(r'$t/T_e$ (fraction of the time to the event)')
ax.set_ylabel(r'$\Theta(t)/\Theta(T_e)$')
ax.set_xlim(0, 1.0); ax.set_ylim(0.3, 1e7); ax.grid(alpha=0.3)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
fig.tight_layout(); fig.savefig('../figs/fig_v11_theta.pdf'); plt.close(fig)

# ---------------- Pi against the measured delay; SIR a posteriori estimate ----------------
d = pd.read_csv(S + 'index_scalar.csv'); d = d[d.ok]
r = pd.read_csv(S + 'sir_runs.csv'); r = r[r.ok]
mk = {'RK45': 'o', 'DOP853': 's', 'LSODA': '^', 'BDF': 'v', 'Radau': 'D'}
cl = {'RK45': 'C0', 'DOP853': 'C1', 'LSODA': 'C2', 'BDF': 'C3', 'Radau': 'C4'}
fig, axs = plt.subplots(1, 2, figsize=(W, 2.7))
ax = axs[0]
for m in mk:
    a = d[d.method == m]
    ax.loglog(a.tolPi, a.eta, mk[m], ms=3, mfc='none', mew=0.7, color=cl[m], label=m)
    b_ = r[r.method == m]
    ax.loglog(b_.tolPi, b_.eta, mk[m], ms=3.2, color=cl[m], alpha=0.8)
xx = np.array([1e-9, 1e4])
ax.loglog(xx, xx, 'k-', lw=0.8); ax.loglog(xx, 10 * xx, 'k:', lw=0.6); ax.loglog(xx, 0.1 * xx, 'k:', lw=0.6)
ax.set_xlim(1e-9, 1e4); ax.set_ylim(1e-13, 1e4)
ax.set_xlabel(r'$\mathrm{tol}\cdot\Pi$'); ax.set_ylabel(r'$\eta=|D|/\Theta(T_e)$')
ax.set_title(r'(a) $\eta$ against tol$\cdot\Pi$')
ax.legend(loc='upper left', frameon=False, handletextpad=0.2, borderaxespad=0.2); ax.grid(alpha=0.3)
ax = axs[1]
rr = r[np.isfinite(r.Dhat)]
for m in mk:
    a = rr[rr.method == m]
    ax.loglog(np.abs(a.D), np.abs(a.Dhat), mk[m], ms=3.2, color=cl[m], label=m)
xx = np.array([1e-11, 1e2]); ax.loglog(xx, xx, 'k-', lw=0.8)
ax.set_xlabel(r'exact delay $|D|$'); ax.set_ylabel(r'estimate $|\hat D|$')
ax.set_title('(b) SIR: a posteriori estimate')
ax.grid(alpha=0.3)
fig.tight_layout(w_pad=1.0); fig.savefig('../figs/fig_v11_index.pdf'); plt.close(fig)

# ---------------- integrated law ----------------
COL = {'flame2': '#2a78d6', 'flame3': '#eb6834', 'ghost': '#1baf7a', 'logistic': '#eda100'}
MK = {'flame2': 'o', 'flame3': 's', 'ghost': '^', 'logistic': 'D'}
LAB = {'flame2': r"$y'=y^2(1-y)$", 'flame3': r"$y'=y^3(1-y)$", 'ghost': r"$x'=\varepsilon+x^2$",
       'logistic': 'logistic'}
D = pd.read_csv(R + 'delay_law_fit_runs.csv'); D = D[D.res_M0.notna()]
F = pd.read_csv(R + 'delay_law_fit.csv').set_index(['env', 'solver'])
D['pred2'] = [F.loc[(q.env, q.solver)].K_M2 * q.tol**F.loc[(q.env, q.solver)].alpha_M2 * q.Lambda
              for _, q in D.iterrows()]
g = D.groupby(['env', 'solver'])
cM = g.apply(lambda G: 10**np.mean(np.log10(G.eta / (G.tol * G.Pi))))
D['pred0'] = [cM.loc[(q.env, q.solver)] * q.tol * q.Pi for _, q in D.iterrows()]
fig = plt.figure(figsize=(W, 4.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0])
axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
for ax, col, ttl in ((axs[0], 'pred0', r'(a) one-step model $c\,\mathrm{tol}\,\Pi$'),
                     (axs[1], 'pred2', r'(b) integrated law $K\,\mathrm{tol}^{\alpha}\Lambda_\alpha$')):
    for p in COL:
        a = D[D.problem == p]
        ax.loglog(a[col], a.eta, MK[p], ms=2.6, mfc='none', mew=0.6, color=COL[p], label=LAB[p])
    xx = np.array([1e-14, 1e6])
    ax.loglog(xx, xx, '-', color='#0b0b0b', lw=0.8)
    for k in (10, 0.1):
        ax.loglog(xx, k * xx, ':', color='#52514e', lw=0.6)
    ax.set_xlim(1e-12, 1e5); ax.set_ylim(1e-12, 1e5)
    ax.set_xticks([1e-10, 1e-5, 1]); ax.set_yticks([1e-10, 1e-5, 1])
    ax.set_xlabel('prediction'); ax.set_title(ttl); ax.grid(alpha=0.25, lw=0.4)
    sd = np.log10(D.eta / D[col]).std()
    ax.text(0.97, 0.04, f'sd of log$_{{10}}$ ratio: {sd:.2f}', transform=ax.transAxes, ha='right', fontsize=7)
axs[0].set_ylabel(r'$\eta=|D|/\Theta(T_e)$')
axs[0].legend(loc='upper left', frameon=False, handletextpad=0.2, borderaxespad=0.2)
ax = axs[2]
A = pd.read_csv(R + 'new_checks_A.csv'); A = A[A.ok]
for m, mkk, cc in (('BDF', 'v', '#2a78d6'), ('RK45', 'o', '#eb6834'), ('Radau', 's', '#1baf7a')):
    a = A[A.solver == m]
    ax.loglog(a.tstar, np.abs(a.D_over_t), mkk, ms=3.5, mfc='none', mew=0.8, color=cc, label=m)
ax.set_xlabel(r'passage time $t_*$'); ax.set_ylabel(r'$|D|/t_*$')
ax.set_title('(c) relative control only')
ax.set_ylim(1e-10, 1e-3); ax.grid(alpha=0.25, lw=0.4)
ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), frameon=False)
fig.tight_layout(h_pad=0.8, w_pad=1.0); fig.savefig('../figs/fig_v11_delay_law.pdf'); plt.close(fig)

# ---------------- MATLAB runs and the applications (supplement) ----------------
dm = pd.read_csv(S + 'matlab_pi_flame.csv'); dm = dm[np.isfinite(dm.eta) & (dm.completed == 1) & (dm.eta > 0)]
sm = pd.read_csv(S + 'matlab_pi_sir.csv'); sm = sm[np.isfinite(sm.eta) & (sm.eta > 0)]
new = pd.read_csv(S + 'matlab_phase_index.csv'); new = new[np.isfinite(new.eta) & (new.eta > 0)]
fig, axs = plt.subplots(2, 1, figsize=(W, 5.6))
ax = axs[0]
sty = {'ode15s_BDF': ('v', 'C3'), 'ode15s': ('^', 'C2'), 'ode23t': ('<', 'C5'), 'ode23tb': ('>', 'C6'),
       'ode23s': ('s', 'C1'), 'ode113': ('D', 'C8'), 'ode23': ('o', 'C0'), 'ode45': ('x', 'C4')}
for sv, (mkk, cc) in sty.items():
    a = dm[dm.solver == sv]
    ax.loglog(a.tolPi, a.eta, mkk, ms=3, mfc='none', color=cc, label=sv.replace('_', ' '))
    b = sm[sm.solver == sv]
    if len(b):
        ax.loglog(b.tolPi, b.eta, mkk, ms=3.5, color=cc)
    nn = new[new.solver == sv]
    ax.loglog(nn.tolPi, nn.eta, mkk, ms=3, color=cc, alpha=0.6)
xx = np.array([1e-12, 1e6])
ax.loglog(xx, xx, 'k-', lw=0.8); ax.loglog(xx, 10 * xx, 'k:', lw=0.6); ax.loglog(xx, 0.1 * xx, 'k:', lw=0.6)
ax.set_xlim(1e-12, 1e6); ax.set_ylim(1e-14, 1e6)
ax.set_xlabel(r'$\mathrm{RelTol}\cdot\Pi$'); ax.set_ylabel(r'$\eta=|D|/\Theta(T_e)$')
ax.set_title('(a) MATLAB R2023b, eight solvers\n(open: flame, companion-paper runs; filled: other problems)')
ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), frameon=False); ax.grid(alpha=0.3)
ax = axs[1]
h = pd.read_csv(S + 'h2_theta.csv'); hc = pd.read_csv(S + 'h2_cases.csv')
for ratio, ls in ((1e-2, '-'), (1e-6, '--'), (1e-10, ':')):
    a = h[h.ratio == ratio]; Te = hc.Te.iloc[0]
    ax.semilogy(a.t / Te, a.Theta / a.Theta_e, color='C3', ls=ls, lw=1.1,
                label=rf'H$_2$/air, $\mathrm{{AbsTol}}/\mathrm{{RelTol}}=10^{{{int(np.log10(ratio))}}}$')
m = pd.read_csv(S + 'ml_theta.csv'); mc = pd.read_csv(S + 'ml_cases.csv')
for Dl, ls in ((1e-1, ':'), (1e-2, '--'), (1e-3, '-')):
    a = m[m.Delta == Dl]; P = mc[mc.Delta == Dl].period.iloc[0]
    ax.semilogy(a.t / P, a.Theta / a.Theta_e, color='C0', ls=ls, lw=1.1,
                label=rf'Morris–Lecar, $\Delta=10^{{{int(np.log10(Dl))}}}$')
ax.set_xlabel(r'$t/T_e$ (H$_2$: ignition; Morris–Lecar: one period)')
ax.set_ylabel(r'$\Theta(t)/\Theta(T_e)$')
ax.set_title('(b) timing sensitivity in the two applications')
ax.set_xlim(0, 1); ax.set_ylim(0.1, 1e9)
ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), frameon=False); ax.grid(alpha=0.3)
fig.tight_layout(h_pad=1.2); fig.savefig('../figs/fig_v11_matlab_apps.pdf'); plt.close(fig)
print('ok')
