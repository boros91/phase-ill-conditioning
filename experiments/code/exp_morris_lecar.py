"""Morris-Lecar neuron near the SNIC bifurcation, I = I_c + Delta.
Events: upward crossings of V = 10 mV (spikes).  Start on the orbit at a spike.
 - reference (DOP853 and Radau, tight) and its accuracy
 - adjoint over one period before a spike (the phase response curve), Theta, Pi, Pi_*
 - delays of the 1st and 10th spike for the SciPy integrators, rtol = 1e-4,1e-6,1e-8, atol = rtol/100
 - remedy: time as dependent variable, rate rho = max_i |f_i|/s_i, absolute tolerance on t
Outputs: ../results/ml_cases.csv, ml_runs.csv, ml_theta.csv
"""
import csv, time
import numpy as np
from scipy.integrate import solve_ivp, RK45, DOP853, LSODA, BDF, Radau
from scipy.optimize import brentq
import ml_common as ml

METHODS = {'RK45': RK45, 'DOP853': DOP853, 'LSODA': LSODA, 'BDF': BDF, 'Radau': Radau}
VE, RATIO, NSP = 10.0, 1e-2, 10
Ic, _ = ml.snic_current()
cases, runs, throws = [], [], []

for Delta in (1e-1, 1e-2, 1e-3):
    I = Ic + Delta
    f, J = ml.make(I)
    # a point on the limit cycle at a spike: integrate past the transient
    pre = solve_ivp(f, [0, 40000], [-30.0, 0.1], method='DOP853', rtol=1e-13, atol=1e-15, dense_output=True)
    V = pre.y[0]; up = np.where((V[:-1] < VE) & (V[1:] >= VE))[0]
    k = up[-2]
    ts = brentq(lambda t: pre.sol(t)[0] - VE, pre.t[k], pre.t[k + 1], xtol=1e-14)
    x0 = pre.sol(ts)
    # reference: spike times over NSP periods, two integrators
    span_end = None
    spike_ref = {}
    for m in ('DOP853', 'Radau'):
        kw = dict(rtol=1e-13, atol=1e-15) if m == 'DOP853' else dict(rtol=1e-12, atol=1e-14, jac=J)
        s = solve_ivp(f, [0, 1e6], x0, method=m, dense_output=True,
                      events=[(lambda t, x: x[0] - VE)], **kw) if False else None
        # integrate to a generous end, then locate crossings on the dense output
        P_est = np.pi / np.sqrt(Delta) * 60
        s = solve_ivp(f, [0, (NSP + 0.5) * P_est * 1.2], x0, method=m, dense_output=True, **kw)
        V = s.y[0]; up = np.where((V[:-1] < VE) & (V[1:] >= VE))[0]
        tsp = [brentq(lambda t: s.sol(t)[0] - VE, s.t[i], s.t[i + 1], xtol=1e-13) for i in up]
        tsp = [t for t in tsp if t > 0.3 * P_est]          # the start is itself a spike
        spike_ref[m] = (s, np.array(tsp))
    ref, tspk = spike_ref['DOP853']
    ref_err = np.abs(spike_ref['Radau'][1][:NSP] - tspk[:NSP]).max()
    P = tspk[0]
    tspk = tspk[:NSP]
    print(f'Delta={Delta:.0e}: period {P:.6f}, spike times ref err {ref_err:.2e}', flush=True)

    # adjoint over one period ending at the first spike
    xe = ref.sol(P); fe = f(P, xe)
    lamT = np.array([1.0 / fe[0], 0.0])
    sa = solve_ivp(lambda tau, lam: J(P - tau, ref.sol(P - tau)).T @ lam, [0, P], lamT,
                   method='DOP853', rtol=1e-10, atol=1e-14, dense_output=True)
    tg = np.linspace(0, P, 20001)
    LAM = np.array([sa.sol(P - t) for t in tg]).T
    X = ref.sol(tg)
    F = np.array([f(t, X[:, i]) for i, t in enumerate(tg)]).T
    chk = (LAM * F).sum(0)
    S = np.abs(X) + RATIO
    th = (np.abs(LAM) * S).sum(0)
    the = th[-1]
    mu = (np.abs(F) / S).max(0)
    Pi, Pistar = th.max() / the, (1 / mu).max() / the
    Srel = np.abs(X)
    Pi_rel = (np.abs(LAM) * Srel).sum(0).max() / (np.abs(LAM[:, -1]) * Srel[:, -1]).sum()
    cases.append(dict(Delta=Delta, I=I, period=P, ref_err=ref_err, Theta_e=the, Pi=Pi, Pi_star=Pistar,
                      Pi_relative=Pi_rel, V_argmax=X[0, np.argmax(th)], lambda_f_min=chk.min(),
                      lambda_f_max=chk.max()))
    print(f'   lambda.f in [{chk.min():.6f},{chk.max():.6f}]  Theta_e={the:.3e}  Pi={Pi:.3e}  '
          f'Pi_*={Pistar:.3e}  Pi_rel={Pi_rel:.3e}  V at max={X[0, np.argmax(th)]:.2f}', flush=True)
    for i in range(0, len(tg), 50):
        throws.append(dict(Delta=Delta, t=tg[i], V=X[0, i], w=X[1, i], Theta=th[i], Theta_e=the,
                           lamV=LAM[0, i], lamw=LAM[1, i]))

    def tref_level(n, Vval):
        """time at which the reference reaches V=Vval on the upstroke of spike n (1-based)"""
        tn = tspk[n - 1]
        lo, hi = tn - 0.2 * P, tn + 0.05 * P
        # restrict to the monotone upstroke around tn
        g = lambda t: ref.sol(t)[0] - Vval
        a = tn
        while g(a) > 0 and a > lo:
            a -= 0.05
        b = tn
        while g(b) < 0 and b < hi:
            b += 0.05
        return brentq(g, a, b, xtol=1e-13)

    def run(method, tol, timectl=False):
        kw = dict(rtol=tol, atol=tol * RATIO)
        if method in ('LSODA', 'BDF', 'Radau'):
            kw['jac'] = J
        if not timectl:
            s = METHODS[method](f, 0.0, x0.copy(), 1e9, **kw)
            get = lambda: (s.t, s.y[0])
        else:
            # time as dependent variable: dx/dtau = f/rho, dt/dtau = 1/rho
            def g(tau, z):
                x = z[:2]; fx = f(0, x); rho = np.max(np.abs(fx) / (np.abs(x) + RATIO))
                return np.concatenate([fx / rho, [1.0 / rho]])
            s = METHODS[method](g, 0.0, np.concatenate([x0, [0.0]]), 1e9,
                                rtol=np.array([tol, tol, 1e-13]),
                                atol=np.array([tol * RATIO, tol * RATIO, tol * the]))
            get = lambda: (s.y[2], s.y[0])
        D = []; n = 0; below = False; steps = 0
        while s.status == 'running' and len(D) < NSP and steps < 2_000_000:
            s.step(); steps += 1
            t, V = get()
            if V < VE - 1.0:
                below = True
            if below and V >= VE:
                below = False; n += 1
                D.append(t - tref_level(n, V))
        return np.array(D), steps, s.nfev

    for m in METHODS:
        for tol in (1e-4, 1e-6, 1e-8):
            t0 = time.time()
            D, steps, nfev = run(m, tol)
            ok = len(D) == NSP
            r = dict(Delta=Delta, method=m, tol=tol, variant='standard', Pi=Pi, tolPi=tol * Pi, Theta_e=the,
                     D1=D[0] if len(D) else np.nan, D10=D[-1] if ok else np.nan, nspikes=len(D),
                     eta1=abs(D[0]) / the if len(D) else np.nan, steps=steps, nfev=nfev,
                     D10_over_D1=D[-1] / D[0] if ok and D[0] != 0 else np.nan)
            runs.append(r)
            print(f'   {m:6s} {tol:.0e}: D1={r["D1"]:+.3e}  D10={r["D10"]:+.3e}  D10/D1={r["D10_over_D1"]:.2f}'
                  f'  eta1={r["eta1"]:.2e}  tolPi={tol*Pi:.2e}  steps={steps}  ({time.time()-t0:.1f}s)', flush=True)
    for m in ('RK45', 'LSODA'):
        for tol in (1e-4, 1e-6):
            D, steps, nfev = run(m, tol, timectl=True)
            ok = len(D) == NSP
            runs.append(dict(Delta=Delta, method=m, tol=tol, variant='time control', Pi=Pi, tolPi=tol * Pi,
                             Theta_e=the, D1=D[0] if len(D) else np.nan, D10=D[-1] if ok else np.nan,
                             nspikes=len(D), eta1=abs(D[0]) / the if len(D) else np.nan, steps=steps,
                             nfev=nfev, D10_over_D1=D[-1] / D[0] if ok and D[0] != 0 else np.nan))
            print(f'   time control {m:6s} {tol:.0e}: D1={runs[-1]["D1"]:+.3e}  D10={runs[-1]["D10"]:+.3e}'
                  f'  steps={steps}', flush=True)

for name, rr in (('ml_cases', cases), ('ml_runs', runs), ('ml_theta', throws)):
    with open(f'../results/{name}.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rr[0])); w.writeheader(); w.writerows(rr)
