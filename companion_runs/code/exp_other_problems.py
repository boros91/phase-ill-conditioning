"""The phase error on three other autonomous problems: analysis of the MATLAB runs of
matlab/run_other_problems.m (results/matlab/other_*.csv).

1. SIR (beta = 0.3, gamma = 0.1, I(0) = 1e-7, 1e-9).  S decreases monotonically, so every computed
   point has an exact time on the orbit, T(S) = int_S^S0 ds / (beta s I(s)),
   I(s) = S0 + I0 - s + rho ln(s/S0), rho = gamma/beta (quadrature).  D_k = t_k - T(S_k);
   D_peak = D at the first computed point with S_k <= rho (the epidemic peak).  Max error in I against
   a reference solution (DOP853 in ln S, ln I, rtol = atol = 1e-13), checked against the quadrature.
2. May-Leonard (alpha = 0.8, beta = 1.3, x0 = (0.6, 0.3, 0.1)).  Reference: DOP853 in u = ln x,
   rtol = atol = 1e-13 (checked against rtol = atol = 1e-11).  At the j-th switch the incoming
   component x_i rises through 1/2; D_j = t_k - T_ref,j(x_i,k) at the first computed point with
   x_i >= 1/2, T_ref,j the reference time at which x_i reaches that value on the same rise.
3. Theta neuron, I = 1e-4, 1e-6, theta(0) = -pi, not wrapped.  Exact time to reach theta:
   T(theta) = n P + P/2 + arctan(tan(r/2)/sqrt(I))/sqrt(I), theta = r + 2 pi n, r in [-pi, pi),
   P = pi/sqrt(I).  D at each spike (first computed point with theta >= (2k+1) pi), k = 0..9.

Outputs: results/other_sir.csv, results/other_ml.csv, results/other_theta.csv (one row per run),
results/other_ml_switches.csv, results/other_theta_spikes.csv, results/other_reference.txt,
results/other_profiles.npz (D_k along the trajectories, for the figure).
"""
import os
import numpy as np, pandas as pd
from scipy.integrate import quad, solve_ivp

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, '..', 'results')
st = pd.read_csv(os.path.join(RES, 'matlab', 'other_problems_stats.csv'))
log = open(os.path.join(RES, 'other_reference.txt'), 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s); log.write(s + '\n')
prof = {}

# ------------------------------------------------------------------ 1. SIR
be, ga = 0.3, 0.1; rho = ga / be
tr = pd.read_csv(os.path.join(RES, 'matlab', 'other_sir.csv'))
rows = []
for I0 in sorted(st[st.example == 'sir'].param.unique()):
    S0 = 1 - I0
    c = 1 - rho / S0
    Iof = lambda x: I0 + x + rho * np.log1p(-x / S0)          # I as a function of x = S0 - S
    g = lambda x: 1.0 / (be * (S0 - x) * Iof(x))
    def T(S):
        X = S0 - S
        if X <= 0:
            return 0.0
        edges = [0.0]; e = I0 / c
        while e < X:
            edges.append(e); e *= 4
        edges.append(X)
        return sum(quad(g, a, b, epsabs=0, epsrel=1e-13, limit=200)[0] for a, b in zip(edges[:-1], edges[1:]))
    tpk = T(rho)
    fu = lambda t, u: [-be * np.exp(u[1]), be * np.exp(u[0]) - ga]
    ref = solve_ivp(fu, (0, 300), [np.log(S0), np.log(I0)], method='DOP853', rtol=1e-13, atol=1e-13,
                    dense_output=True)
    ev = solve_ivp(fu, (0, 300), [np.log(S0), np.log(I0)], method='DOP853', rtol=1e-13, atol=1e-13,
                   events=lambda t, u: np.exp(u[0]) - rho)
    say(f'SIR I0 = {I0:g}: peak time by quadrature {tpk:.10f}, by the reference ODE {ev.t_events[0][0]:.10f}')
    for _, r in st[(st.example == 'sir') & (st.param == I0)].iterrows():
        d = tr[tr.run == r.run].sort_values('k')
        t = d.t.values
        if r.form == 'log':
            S, I = np.exp(d.y1.values), np.exp(d.y2.values)
        else:
            S, I = d.y1.values, d.y2.values
        peak = bool(np.any(S <= rho))                      # did the computed epidemic reach its peak?
        j = int(np.argmax(S <= rho)) if peak else len(t) - 1
        D = np.array([tk - T(Sk) for tk, Sk in zip(t[:j + 1], S[:j + 1])])   # up to the peak
        Iex = np.exp(ref.sol(t)[1])
        rows.append(dict(run=r.run, I0=I0, variant=r.variant, solver=r.solver, RelTol=r.RelTol, AbsTol=r.AbsTol,
                         nsteps=r.nsteps, nfevals=r.nfevals, t_peak_exact=tpk, peak_reached=int(peak),
                         D_peak=D[j] if peak else np.nan, Imax=I.max(), t_Imax=t[np.argmax(I)], minI=I.min(),
                         maxerr_I=np.max(np.abs(I - Iex)), max_absD=np.max(np.abs(D)) if peak else np.nan))
        prof[f'sir_{I0:g}_{r.variant}_{r.solver}'] = np.c_[t[:j + 1], S[:j + 1], D]
        say(f'  {r.variant:4s} {r.solver:7s} steps {r.nsteps:6d}  peak {"yes" if peak else "NO "} D_peak {rows[-1]["D_peak"]:9.4f}'
            f'  max I {I.max():.3g} at t = {t[np.argmax(I)]:.1f}  min I {I.min():.2e}  max err I {rows[-1]["maxerr_I"]:.2e}')
pd.DataFrame(rows).to_csv(os.path.join(RES, 'other_sir.csv'), index=False)

# ------------------------------------------------------------------ 2. May-Leonard
a, b = 0.8, 1.3
M = np.array([[1, a, b], [b, 1, a], [a, b, 1]])
x0 = np.array([0.6, 0.3, 0.1]); TF = 1500
fu = lambda t, u: 1 - M @ np.exp(u)
ref = solve_ivp(fu, (0, TF), np.log(x0), method='DOP853', rtol=1e-13, atol=1e-13, dense_output=True)
ref2 = solve_ivp(fu, (0, TF), np.log(x0), method='DOP853', rtol=1e-11, atol=1e-11, dense_output=True)
tt = np.linspace(0, TF, 1500001)
U = ref.sol(tt).T; U2 = ref2.sol(tt).T

def rises(t, U):
    """Reference rises: list of (i, t_cross, segment index range) where u_i crosses ln(1/2) upward."""
    out = []
    h = np.log(0.5)
    for i in range(3):
        s = U[:, i] - h
        for k in np.where((s[:-1] < 0) & (s[1:] >= 0))[0]:
            tc = t[k] + (t[k + 1] - t[k]) * (-s[k]) / (s[k + 1] - s[k])
            out.append((tc, i, k))
    return sorted(out)

R = rises(tt, U); R2 = rises(tt, U2)
say('May-Leonard reference switch times:', ' '.join(f'{x[0]:.4f}' for x in R))
say('  rtol 1e-11 minus rtol 1e-13, largest difference:', max(abs(p[0] - q[0]) for p, q in zip(R, R2)))

def T_on_rise(i, k_cross, ulev):
    """Reference time at which u_i reaches ulev on the rise that crosses ln(1/2) at grid index k_cross."""
    lo = k_cross
    while lo > 0 and U[lo - 1, i] < U[lo, i]:
        lo -= 1
    hi = k_cross + 1
    while hi < len(tt) - 1 and U[hi + 1, i] > U[hi, i]:
        hi += 1
    seg_u, seg_t = U[lo:hi + 1, i], tt[lo:hi + 1]
    return np.interp(ulev, seg_u, seg_t)

tr = pd.read_csv(os.path.join(RES, 'matlab', 'other_ml.csv'))
rows, srows = [], []
for _, r in st[st.example == 'ml'].iterrows():
    d = tr[tr.run == r.run].sort_values('k')
    t = d.t.values
    Y = d[['y1', 'y2', 'y3']].values
    with np.errstate(divide='ignore', invalid='ignore'):
        Uk = Y if r.form == 'log' else np.log(Y)
    X = np.exp(Uk) if r.form == 'log' else Y
    Ds = []
    for j, (tc, i, kc) in enumerate(R):
        # first computed point of this rise past 1/2: after the previous switch, component i
        t_prev = R[j - 1][0] if j > 0 else -1
        cand = np.where((t > t_prev + 0.5 * (tc - t_prev) if j > 0 else t >= 0) & (X[:, i] >= 0.5))[0]
        if len(cand) == 0 or not np.all(np.isfinite(X[: cand[0] + 1])):
            Ds.append(np.nan); continue
        k = cand[0]
        Ds.append(t[k] - T_on_rise(i, kc, np.log(X[k, i])))
        srows.append(dict(run=r.run, variant=r.variant, solver=r.solver, switch=j + 1, t_ref=tc, D=Ds[-1]))
    Ds = np.array(Ds)
    nvalid = int(np.sum(np.isfinite(Ds)))
    rows.append(dict(run=r.run, variant=r.variant, solver=r.solver, RelTol=r.RelTol, AbsTol=r.AbsTol,
                     nsteps=r.nsteps, nfevals=r.nfevals, min_x=np.nanmin(X), n_switch_ref=len(R),
                     n_switch_found=nvalid, D_last=Ds[nvalid - 1] if nvalid else np.nan,
                     max_absD=np.nanmax(np.abs(Ds)) if nvalid else np.nan))
    prof[f'ml_{r.variant}_{r.solver}'] = np.c_[[x[0] for x in R], Ds]
    say(f'  ML {r.variant:4s} {r.solver:7s} steps {r.nsteps:6d} min x {np.nanmin(X):.2e}  D per switch',
        ' '.join(f'{x:.3g}' for x in Ds))
pd.DataFrame(rows).to_csv(os.path.join(RES, 'other_ml.csv'), index=False)
pd.DataFrame(srows).to_csv(os.path.join(RES, 'other_ml_switches.csv'), index=False)

# ------------------------------------------------------------------ 3. theta neuron
tr = pd.read_csv(os.path.join(RES, 'matlab', 'other_theta.csv'))
rows, srows = [], []
for _, r in st[st.example == 'theta'].iterrows():
    Iext = r.param; P = np.pi / np.sqrt(Iext)
    def Tex(th):
        n = np.floor((th + np.pi) / (2 * np.pi)); rr = th - 2 * np.pi * n
        return n * P + P / 2 + np.arctan(np.tan(rr / 2) / np.sqrt(Iext)) / np.sqrt(Iext)
    d = tr[tr.run == r.run].sort_values('k')
    t, th = d.t.values, d.y1.values
    D = t - Tex(th)
    Ds = []
    for k in range(10):
        idx = np.where(th >= (2 * k + 1) * np.pi)[0]
        Ds.append(D[idx[0]] if len(idx) else np.nan)
        if len(idx):
            srows.append(dict(run=r.run, I=Iext, variant=r.variant, solver=r.solver, spike=k + 1,
                              t_exact=(k + 1) * P, D=D[idx[0]]))
    Ds = np.array(Ds)
    nsp = int(np.sum(np.isfinite(Ds)))
    rows.append(dict(run=r.run, I=Iext, period=P, variant=r.variant, solver=r.solver, RelTol=r.RelTol,
                     AbsTol=r.AbsTol, nsteps=r.nsteps, nfevals=r.nfevals, spikes_found=nsp,
                     spikes_exact=10, D_spike1=Ds[0], D_last=Ds[nsp - 1] if nsp else np.nan,
                     max_absD=np.nanmax(np.abs(D))))
    prof[f'theta_{Iext:g}_{r.variant}_{r.solver}'] = np.c_[np.arange(1, 11), Ds]
    say(f'  theta I={Iext:g} {r.variant:4s} {r.solver:7s} steps {r.nsteps:7d} spikes {nsp:2d}  D at spikes',
        ' '.join(f'{x:.3g}' for x in Ds))
pd.DataFrame(rows).to_csv(os.path.join(RES, 'other_theta.csv'), index=False)
pd.DataFrame(srows).to_csv(os.path.join(RES, 'other_theta_spikes.csv'), index=False)
np.savez(os.path.join(RES, 'other_profiles.npz'), **prof)
log.close()
