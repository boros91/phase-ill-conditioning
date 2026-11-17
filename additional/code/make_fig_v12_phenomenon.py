"""Figure 1 of v12: the phenomenon.
(a) flame problem y' = y^2 - y^3, y(0) = 1e-4: exact solution and SciPy solutions (RK45, BDF, LSODA,
    rtol 1e-4, atol 1e-6) near the front;
(b) the same computed points plotted against t - D, D the delay of each run at y = 1/2;
(c) SIR, I0 = 1e-8: infected fraction, reference and SciPy runs with the same tolerances.
Output: ../figs/fig_v12_phenomenon.pdf and ../results/fig_v12_phenomenon.txt (the numbers quoted).
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import wrightomega

W = 5.12
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'legend.fontsize': 6.5,
                     'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                     'font.family': 'serif', 'mathtext.fontset': 'cm'})
R, A = 1e-4, 1e-6
out = []
delta = 1e-4
lam = 1 / delta - 1
exact = lambda t: 1.0 / (1.0 + np.real(wrightomega(lam + np.log(lam) - t)))
Tflame = lambda y: 1 / delta - 1 / y + np.log(y * (1 - delta) / (delta * (1 - y)))
c = Tflame(0.5)
sty = {'RK45': ('C0', 'o'), 'BDF': ('C3', 'v'), 'LSODA': ('C2', '^')}

fig, axs = plt.subplots(1, 3, figsize=(W, 2.05))
tt = np.linspace(c - 260, c + 60, 2000)
for ax in axs[:2]:
    ax.plot(tt - c, exact(tt), 'k-', lw=1.0, label='exact')
for m, (col, mk) in sty.items():
    kw = dict(jac=lambda t, y: [[2 * y[0] - 3 * y[0]**2]]) if m != 'RK45' else {}
    s = solve_ivp(lambda t, y: [y[0]**2 - y[0]**3], (0, 2 / delta), [delta], method=m, rtol=R, atol=A, **kw)
    k = int(np.argmax(s.y[0] >= 0.5))
    D = s.t[k] - Tflame(s.y[0][k])
    err = np.abs(s.y[0] - exact(s.t)).max()
    out.append(f"flame {m}: steps {s.t.size - 1}, delay at the first step past y=1/2: {D:+.4g}, max error {err:.3g}")
    sel = (s.t > c - 300) & (s.t < c + 80)
    axs[0].plot(s.t[sel] - c, s.y[0][sel], mk + '-', color=col, ms=2.6, lw=0.6, mfc='none', mew=0.6, label=m)
    axs[1].plot(s.t[sel] - D - c, s.y[0][sel], mk, color=col, ms=2.6, mfc='none', mew=0.6, label=m)
axs[0].set_xlim(-260, 60); axs[1].set_xlim(-40, 40)
axs[0].set_xlabel(r'$t-T_e$'); axs[1].set_xlabel(r'$t-D-T_e$')
axs[0].set_ylabel(r'$y$')
axs[0].set_title('(a) flame problem', loc='left'); axs[1].set_title('(b) shifted by the delay', loc='left')
axs[1].legend(loc='lower right', frameon=False, handlelength=1.5, borderaxespad=0.2)

b, g = 0.3, 0.1
I0 = 1e-8
sir = lambda t, u: [-b * u[0] * u[1], b * u[0] * u[1] - g * u[1]]
sirJ = lambda t, u: [[-b * u[1], -b * u[0]], [b * u[1], b * u[0] - g]]
lr = lambda t, u: [-b * np.exp(u[1]), b * np.exp(u[0]) - g]
ref = solve_ivp(lr, (0, 220), [np.log(1 - I0), np.log(I0)], method='DOP853', rtol=1e-13, atol=1e-13, dense_output=True)
tg = np.linspace(0, 220, 2201)
Iref = np.exp(ref.sol(tg)[1])
tpk = tg[np.argmax(Iref)]
from scipy.optimize import brentq
Te_sir = brentq(lambda t: np.exp(ref.sol(t)[0]) - g / b, 50, 150, xtol=1e-12)
out.append(f"SIR I0=1e-8: reference event time T_e = {Te_sir:.6f}")
axs[2].plot(tg, Iref, 'k-', lw=1.0, label='reference')
out.append(f"SIR I0=1e-8: reference peak of I at t = {tpk:.2f}")
for m, (col, mk) in sty.items():
    kw = dict(jac=sirJ) if m != 'RK45' else {}
    s = solve_ivp(sir, (0, 220), [1 - I0, I0], method=m, rtol=R, atol=A, dense_output=True, **kw)
    Im = s.sol(tg)[1]
    ev = lambda t, u: u[0] - g / b
    ev.terminal = True
    se = solve_ivp(sir, (0, 220), [1 - I0, I0], method=m, rtol=R, atol=A, events=ev, **kw)
    out.append(f"SIR {m}: event S = gamma/beta " + (f"at t = {se.t_events[0][0]:.4f}, delay {se.t_events[0][0] - Te_sir:+.3f} days"
                                                   if se.t_events[0].size else "not reached on [0, 220]"))
    out.append(f"SIR {m}: steps {s.t.size - 1}, max I = {Im.max():.3g} at t = {tg[np.argmax(Im)]:.1f}, min I = {s.y[1].min():.2g}")
    axs[2].plot(tg, Im, '-' if m != 'RK45' else '--', color=col, lw=1.4 if m == 'BDF' else 0.9, label=m)
axs[2].set_xlabel(r'$t$ (days)'); axs[2].set_ylabel(r'$I$')
axs[2].set_title(r'(c) SIR, $I_0=10^{-8}$', loc='left'); axs[2].set_xlim(40, 160)
for ax in axs:
    ax.grid(alpha=0.3)
fig.tight_layout(pad=0.4, w_pad=0.6)
fig.savefig('../figs/fig_v12_phenomenon.pdf'); plt.close(fig)
open('../results/fig_v12_phenomenon.txt', 'w').write("\n".join(out) + "\n")
print("\n".join(out))
