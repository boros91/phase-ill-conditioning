"""H2/air ignition (Cantera h2o2.yaml, 1 atm, T0 = 1000 K, stoichiometric, adiabatic,
constant pressure).  Event: T = T0 + 400 K.
 - reference solution and its accuracy (BDF and Radau, rtol 1e-12)
 - adjoint along the reference, Theta(t), Pi and Pi_* for several atol/rtol ratios
 - measured delays of the SciPy integrators for rtol = 1e-4, 1e-6, 1e-8 and
   atol/rtol = 1e-2 (benchmark ratio) and 1e-10 (small absolute tolerance)
 - orbit test (rtol and rtol/10) for BDF and LSODA
Outputs: ../results/h2_cases.csv, h2_runs.csv, h2_theta.csv, h2_orbit.csv
"""
import csv, time
import numpy as np
from scipy.integrate import solve_ivp, RK45, LSODA, BDF, Radau
from scipy.optimize import brentq
import h2_common as h

METHODS = {'RK45': RK45, 'LSODA': LSODA, 'BDF': BDF, 'Radau': Radau}
Te_level = h.T0 + 400.0
names = ['T'] + h.gas.species_names

# ---------------- reference ----------------
refs = {}
for m in ('BDF', 'Radau'):
    t0 = time.time()
    s = solve_ivp(h.rhs, [0, 1e-3], h.x0, method=m, rtol=1e-12, atol=1e-24, dense_output=True,
                  jac=lambda t, x: h.jac_fd(t, x))
    i = np.argmax(s.y[0] >= Te_level)
    tE = brentq(lambda t: s.sol(t)[0] - Te_level, s.t[i - 1], s.t[i], xtol=1e-18)
    refs[m] = (s, tE)
    print(f'reference {m}: t_e = {tE:.12e}  steps {s.t.size}  ({time.time()-t0:.1f}s)', flush=True)
ref, Te = refs['Radau']
ref_err = abs(refs['BDF'][1] - Te)
mono = bool(np.all(np.diff(ref.sol(np.linspace(0, Te, 20001))[0]) > -1e-9))
print(f'reference event time difference BDF-Radau: {ref_err:.2e}; T monotone up to t_e: {mono}')


def t_ref_of_T(Tval):
    """time at which the reference reaches the temperature Tval on the ignition upstroke"""
    lo = Te * 0.5
    hi = 1e-3
    if ref.sol(lo)[0] > Tval:
        lo = 0.0
    return brentq(lambda t: ref.sol(t)[0] - Tval, lo, hi, xtol=1e-18)


# ---------------- adjoint ----------------
xe = ref.sol(Te); fe = h.rhs(Te, xe)
lamT = np.zeros(len(xe)); lamT[0] = 1.0 / fe[0]
# Jacobian along the reference: central differences at the reference steps, linear in t between them
t0 = time.time()
tj = ref.t[ref.t <= Te * (1 + 1e-12)]
tj = np.unique(np.concatenate([tj, [Te]]))
Jg = np.empty((len(tj), len(xe), len(xe)))
for i, ti in enumerate(tj):
    xi = ref.sol(ti)
    for j in range(len(xi)):
        hj = 1e-6 * max(abs(xi[j]), 1e-6 if j else 1.0)
        xp = xi.copy(); xm = xi.copy(); xp[j] += hj; xm[j] -= hj
        Jg[i, :, j] = (h.rhs(ti, xp) - h.rhs(ti, xm)) / (2 * hj)
print(f'Jacobian at {len(tj)} reference points ({time.time()-t0:.1f}s)', flush=True)


def Jt(t):
    i = int(np.clip(np.searchsorted(tj, t) - 1, 0, len(tj) - 2))
    w = (t - tj[i]) / (tj[i + 1] - tj[i])
    return (1 - w) * Jg[i] + w * Jg[i + 1]


# integrate mu = lambda * f_T(T_e) (so that mu(T_e) = e_T), backward in time
adj = lambda tau, mu: Jt(Te - tau).T @ mu
adjJ = lambda tau, mu: Jt(Te - tau).T
t0 = time.time()
sa = solve_ivp(adj, [0, Te], lamT * fe[0], method='BDF', rtol=1e-8, atol=1e-10, dense_output=True, jac=adjJ)
print(f'adjoint: {sa.t.size} steps ({time.time()-t0:.1f}s)', flush=True)
_sol = sa.sol
sa.sol = lambda tau: _sol(tau) / fe[0]
tg = np.unique(np.concatenate([np.linspace(0, Te, 4001), Te * (1 - np.geomspace(1e-8, 1, 1000)),
                               Te * np.geomspace(1e-12, 1, 1000)]))
LAM = np.array([sa.sol(Te - t) for t in tg]).T
X = ref.sol(tg)
check = np.array([LAM[:, k] @ h.rhs(tg[k], X[:, k]) for k in range(0, len(tg), 500)])
print('lambda.f along the orbit (should be 1):', check.min(), check.max())

cases, rows_theta = [], []
for ratio in (1e-2, 1e-6, 1e-10, 0.0):
    S = np.abs(X) + ratio
    th = (np.abs(LAM) * S).sum(0)
    the = (abs(lamT[0]) * (abs(xe[0]) + ratio))
    F = np.array([h.rhs(t, X[:, k]) for k, t in enumerate(tg)]).T
    with np.errstate(divide='ignore', invalid='ignore'):
        mu = np.nanmax(np.where(S > 0, np.abs(F) / S, 0), axis=0)
    Pi = th.max() / the
    # the adjoint-free bound needs s_i > 0 for every component (not true for ratio 0 and Y_i = 0)
    Pistar = (1 / mu).max() / the if ratio > 0 else np.nan
    k = np.argmax(th)
    # component that dominates Theta at the maximum
    dom = names[int(np.argmax(np.abs(LAM[:, k]) * S[:, k]))]
    cases.append(dict(ratio=ratio, Te=Te, Theta_e=the, Pi=Pi, Pi_star=Pistar, t_argmax=tg[k],
                      dominant=dom, ref_err=ref_err))
    print(f'atol/rtol={ratio:.0e}: Theta_e={the:.3e}  Pi={Pi:.3e}  Pi_*={Pistar:.3e}  '
          f'argmax t={tg[k]:.3e} ({dom})', flush=True)
    for k2 in range(0, len(tg), 10):
        rows_theta.append(dict(ratio=ratio, t=tg[k2], Theta=th[k2], Theta_e=the))

# ---------------- solver runs ----------------
rows = []
for ratio in (1e-2, 1e-6, 1e-10):
    Pi = [c['Pi'] for c in cases if c['ratio'] == ratio][0]
    the = [c['Theta_e'] for c in cases if c['ratio'] == ratio][0]
    for m, cls in METHODS.items():
        for tol in (1e-4, 1e-6, 1e-8):
            if m == 'RK45' and tol < 1e-6:
                continue
            kw = dict(rtol=tol, atol=tol * ratio)
            if m != 'RK45':
                kw['jac'] = lambda t, x: h.jac_fd(t, x)
            t0 = time.time()
            s = cls(h.rhs, 0.0, h.x0.copy(), 1e-3, **kw)
            n = 0; ok = True; outcome = 'event'
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                try:
                    while s.status == 'running':
                        s.step(); n += 1
                        if s.y[0] >= Te_level or n > 400000:
                            break
                        if not np.all(np.isfinite(s.y)):
                            outcome = 'error: non-finite state'; ok = False; break
                except Exception as e:
                    ok = False; outcome = 'error: ' + str(e).strip().splitlines()[-2 if 'Cantera' in str(e) else -1][:60]
            if ok and s.y[0] < Te_level:
                ok = False
                outcome = (f'no ignition up to t = {s.t:.1e} (T = {s.y[0]:.1f} K, min Y = {s.y[1:].min():.1e})'
                           if s.status == 'finished' else f'solver status {s.status}')
            D = s.t - t_ref_of_T(s.y[0]) if ok else np.nan
            rows.append(dict(ratio=ratio, method=m, tol=tol, Pi=Pi, tolPi=tol * Pi, Theta_e=the,
                             D=D, eta=abs(D) / the if ok else np.nan, steps=n, nfev=s.nfev, ok=ok,
                             rel_delay=D / Te if ok else np.nan, outcome=outcome))
            print(f'  ratio={ratio:.0e} {m:6s} tol={tol:.0e}: D={D:+.3e} s  D/t_e={D/Te:+.2e}  '
                  f'eta={abs(D)/the:.2e}  tol*Pi={tol*Pi:.2e}  steps={n}  {outcome}  ({time.time()-t0:.1f}s)', flush=True)

# ---------------- default tolerances of SciPy (rtol 1e-3, atol 1e-6) ----------------
for m, cls in METHODS.items():
    kw = dict(rtol=1e-3, atol=1e-6)
    if m != 'RK45':
        kw['jac'] = lambda t, x: h.jac_fd(t, x)
    s = cls(h.rhs, 0.0, h.x0.copy(), 1e-3, **kw)
    n = 0; ok = True; outcome = 'event'
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            while s.status == 'running':
                s.step(); n += 1
                if s.y[0] >= Te_level or n > 400000:
                    break
        except Exception as e:
            ok = False; outcome = 'error: ' + str(e)[:60]
    if ok and s.y[0] < Te_level:
        ok = False
        outcome = (f'no ignition up to t = {s.t:.1e} (T = {s.y[0]:.1f} K, min Y = {s.y[1:].min():.1e})'
                   if s.status == 'finished' else f'solver status {s.status}: {s.message}')
    D = s.t - t_ref_of_T(s.y[0]) if ok else np.nan
    Pi3 = [c['Pi'] for c in cases if c['ratio'] == 1e-2][0]
    rows.append(dict(ratio=1e-3, method=m, tol=1e-3, Pi=np.nan, tolPi=np.nan, Theta_e=np.nan, D=D,
                     eta=np.nan, steps=n, nfev=s.nfev, ok=ok, rel_delay=D / Te if ok else np.nan,
                     outcome='defaults: ' + outcome))
    print(f'  SciPy defaults {m:6s}: D/t_e={D/Te:+.2e}  steps={n}  {outcome}', flush=True)

# ---------------- orbit test ----------------
import importlib.util
spec = importlib.util.spec_from_file_location('ot', 'exp_orbit_test.py')
src = open('exp_orbit_test.py').read()
ns = {}
exec(src[src.index('def orbit_track'):src.index('def orbit_project')], {'np': np}, ns)
orbit_track = ns['orbit_track']
orows = []
span = (0.0, 2 * Te)
tg2 = np.linspace(*span, 20001)
for m in ('BDF', 'LSODA'):
    for tol in (1e-4, 1e-6, 1e-8):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sols = [solve_ivp(h.rhs, span, h.x0, method=m, rtol=tt, atol=tt * 1e-2, dense_output=True,
                              jac=lambda t, x: h.jac_fd(t, x)) for tt in (tol, tol / 10)]
        if not all(z.success for z in sols):
            orows.append(dict(method=m, tol=tol, Etime=np.nan, Eorbit=np.nan, ratio=np.nan, D12=np.nan,
                              note='solver failure'))
            print(f'  orbit test {m} {tol:.0e}: solver failure', flush=True); continue
        U1, U2 = sols[0].sol(tg2).T, sols[1].sol(tg2).T
        sc = np.maximum(np.abs(U2).max(0), 1e-300)
        P, Q = U1 / sc, U2 / sc
        Etime = np.abs(P - Q).max()
        dist, sig = orbit_track(P, Q, tg2)
        ign = [bool((U[:, 0] >= Te_level).any()) for U in (U1, U2)]
        k = int(np.argmax(U1[:, 0] >= Te_level))
        D12 = sig[k] - tg2[k] if ign[0] else np.inf
        note = 'both ignite' if all(ign) else f'ignition: run1 {ign[0]}, run2 {ign[1]}'
        orows.append(dict(method=m, tol=tol, Etime=Etime, Eorbit=dist.max(), ratio=dist.max() / Etime,
                          D12=D12, note=note))
        print(f'  orbit test {m} {tol:.0e}: Etime={Etime:.2e} Eorbit={dist.max():.2e} D12={D12:+.3e} {note}', flush=True)

for name, rr in (('h2_cases', cases), ('h2_runs', rows), ('h2_theta', rows_theta), ('h2_orbit', orows)):
    with open(f'../results/{name}.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rr[0])); w.writeheader(); w.writerows(rr)
