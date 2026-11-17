"""Figure 12 of the paper (APNUM v23; also Fig. 14 of the demo note): the phase error on three other autonomous problems
(MATLAB R2023b runs of matlab/run_other_problems.m, analysed by code/exp_other_problems.py).
(a) SIR, one infected in 1e7: I(t) with RelTol = 1e-4, AbsTol = 1e-6, and the exact solution.
(b) May-Leonard heteroclinic cycle: |D| at the successive switches, unknowns x (solid) and ln x (dashed).
(c) Theta neuron, I = 1e-6: |D| at the successive spikes, RelTol = 1e-4, AbsTol = 1e-6 (solid)
    and with the time-control tolerances RelTol = 100u, AbsTol = 1e-3 * 2I (dashed).
Writes figs/fig12_other.pdf/.png."""
import os
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, '..', 'results')
OUT = os.path.join(HERE, '..', 'figs')
st = pd.read_csv(os.path.join(RES, 'matlab', 'other_problems_stats.csv'))
COL = {'ode45': (0.00, 0.45, 0.74), 'ode23': (0.49, 0.18, 0.56), 'ode23s': (0.85, 0.33, 0.10),
       'ode23t': (0.40, 0.40, 0.40), 'ode15s': (0.47, 0.67, 0.19)}
MK = {'ode45': 'o', 'ode23': 'd', 'ode23s': 's', 'ode23t': 'v', 'ode15s': '^'}
plt.rcParams.update({'font.size': 9, 'axes.linewidth': 0.6, 'axes.grid': True, 'grid.color': '#e3e3e3',
                     'font.family': 'sans-serif', 'pdf.fonttype': 42, 'legend.frameon': True})
fig, ax = plt.subplots(1, 3, figsize=(13, 4.0))

# (a) SIR
be, ga, I0 = 0.3, 0.1, 1e-7
ref = solve_ivp(lambda t, u: [-be * np.exp(u[1]), be * np.exp(u[0]) - ga], (0, 300),
                [np.log(1 - I0), np.log(I0)], method='DOP853', rtol=1e-13, atol=1e-13, dense_output=True)
tt = np.linspace(0, 300, 3001)
ax[0].plot(tt, np.exp(ref.sol(tt)[1]), 'k-', lw=2.2, label='exact (peak at t = 84.4)')
tr = pd.read_csv(os.path.join(RES, 'matlab', 'other_sir.csv'))
for s in ['ode45', 'ode23', 'ode23s', 'ode23t', 'ode15s']:
    r = st[(st.example == 'sir') & (st.param == I0) & (st.variant == 'std') & (st.solver == s)].iloc[0]
    d = tr[tr.run == r.run].sort_values('k')
    ax[0].plot(d.t, d.y2, MK[s] + '-', color=COL[s], ms=3, lw=0.9, label=s)
ax[0].set_xlim(0, 300); ax[0].set_xlabel('t (days)'); ax[0].set_ylabel('infected fraction I')
ax[0].legend(loc='upper center', bbox_to_anchor=(0.6, 1.0), fontsize=7.5)
ax[0].set_title('(a) SIR, one infected in $10^7$\nRelTol $=10^{-4}$, AbsTol $=10^{-6}$', fontsize=10)

# (b) May-Leonard
sw = pd.read_csv(os.path.join(RES, 'other_ml_switches.csv'))
for s in ['ode45', 'ode23', 'ode23s', 'ode15s']:
    for v, ls, mfc in [('std', '-', None), ('log', '--', 'white')]:
        g = sw[(sw.solver == s) & (sw.variant == v)].sort_values('switch')
        g = g[np.isfinite(g.D)]
        ax[1].semilogy(g.switch, np.abs(g.D), MK[s] + ls, color=COL[s], ms=4, lw=1, mfc=mfc or COL[s],
                       label=f'{s}, unknown {"x" if v == "std" else "ln x"}')
ax[1].set_xlabel('switch number'); ax[1].set_ylabel('|D| at the switch')
ax[1].set_ylim(1e-6, 1e3); ax[1].legend(loc='upper left', fontsize=6.5, ncol=2)
ax[1].set_title('(b) May-Leonard heteroclinic cycle\nRelTol $=10^{-4}$, AbsTol $=10^{-6}$', fontsize=10)

# (c) theta neuron, I = 1e-6
sp = pd.read_csv(os.path.join(RES, 'other_theta_spikes.csv'))
for s in ['ode45', 'ode23', 'ode23s', 'ode15s']:
    for v, ls, mfc in [('std', '-', None), ('time', '--', 'white')]:
        g = sp[(sp.I == 1e-6) & (sp.solver == s) & (sp.variant == v)].sort_values('spike')
        ax[2].semilogy(g.spike, np.abs(g.D), MK[s] + ls, color=COL[s], ms=4, lw=1, mfc=mfc or COL[s],
                       label=f'{s}, {"standard" if v == "std" else "time control"}')
ax[2].axhline(np.pi / 1e-3, color='k', lw=0.7, ls=':')
ax[2].text(1.1, np.pi / 1e-3 * 1.3, 'one period', fontsize=7.5)
ax[2].set_xlabel('spike number'); ax[2].set_ylabel('|D| at the spike')
ax[2].set_ylim(1e-5, 1e5); ax[2].legend(loc='center right', fontsize=6.5)
ax[2].set_title('(c) theta neuron near SNIC, $I=10^{-6}$\n(period $\\pi/\\sqrt{I}=3142$)', fontsize=10)
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(OUT, 'fig12_other.' + ext), dpi=200)
print('wrote fig12_other')
