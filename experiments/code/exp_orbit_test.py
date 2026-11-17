"""Black-box detection test without exact solution or adjoint.

Solve with tol and with tol/10 (atol = tol/100 in both).  Compare the two
numerical solutions u1, u2
  * at equal times:      E_time  = max_t |u1(t) - u2(t)|_sc
  * as orbits:           E_orbit = max_t dist_sc(u1(t), orbit of u2)
(|.|_sc: max norm after dividing each component by its maximum over the run).
Mapping each point u1(t) to the time s(t) at which u2 passes the nearest point
of its orbit gives the relative phase profile D12(t) = s(t) - t.
E_orbit << E_time (and E_time >> tol) means that the error is a phase error.
Output: ../results/orbit_test.csv
"""
import csv
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree
import phase_common as pc

RATIO = 1e-2
b, g = 0.3, 0.1


def sir(t, x):
    S, I = x
    return [-b * S * I, b * S * I - g * I]


def sirJ(t, x):
    return [[-b * x[1], -b * x[0]], [b * x[1], b * x[0] - g]]


def scal(p):
    return (lambda t, y: [p.f(y[0])]), (lambda t, y: [[p.fp(y[0])]])


cases = []
# (problem, time window, level at which the relative phase is read, margin):
# the orbit of the second run is sampled up to window end + margin, so that the
# points of the first run near the end of the window are covered by it
for p, span, lev, marg in ((pc.flame_q(2, 1e-4), (0.0, 2e4), 0.5, 0.0),
                           (pc.flame_q(3, 1e-2), (0.0, 2e4), 0.5, 0.0),
                           (pc.ghost(1e-6), (0.0, pc.ghost(1e-6).T(1.0) - 8.0), 0.05, 7.5),
                           (pc.logistic(1e-1), (0.0, 40.0), 0.5, 0.0),
                           (pc.logistic(1e-7), (0.0, 40.0), 0.5, 0.0)):
    f, J = scal(p)
    cases.append((p.name, f, J, [p.x0], span, lev, marg))
for I0 in (1e-6, 1e-8):
    cases.append((f'sir_I{I0:.0e}', sir, sirJ, [1 - I0, I0], (0.0, 300.0), None, 0.0))


# control: a chaotic system, where differences between runs grow transversally to the orbit
def lorenz(t, u):
    x, y, z = u
    return [10.0 * (y - x), x * (28.0 - z) - y, x * y - 8.0 / 3.0 * z]


def lorenzJ(t, u):
    x, y, z = u
    return [[-10.0, 10.0, 0.0], [28.0 - z, -1.0, -x], [y, x, -8.0 / 3.0]]


cases.append(('lorenz', lorenz, lorenzJ, [1.0, 1.0, 20.0], (0.0, 25.0), 'lorenz', 0.0))
import sys
only = sys.argv[1:]
if only:
    cases = [c for c in cases if c[0] in only]


def orbit_track(P, Q, tq, W=300):
    """Follow the polygon Q continuously: for each point of P (in time order)
    search the nearest point of Q only on the W segments before and after the
    previous match.  A phase error gives a continuous, small-distance match;
    differences transverse to the orbit (or a jump to another part of a
    recurrent orbit) give large distances."""
    nseg = len(Q) - 1
    A, B = Q[:-1], Q[1:]
    AB = B - A
    L2 = np.maximum((AB * AB).sum(1), 1e-300)
    j = 0
    best_d = np.empty(len(P)); best_t = np.empty(len(P))
    for i in range(len(P)):
        lo, hi = max(j - W, 0), min(j + W, nseg - 1)
        a_, ab = A[lo:hi + 1], AB[lo:hi + 1]
        lam = np.clip(((P[i] - a_) * ab).sum(1) / L2[lo:hi + 1], 0, 1)
        X = a_ + lam[:, None] * ab
        dd = np.abs(P[i] - X).max(1)
        k = int(np.argmin(dd))
        j = lo + k
        best_d[i] = dd[k]
        best_t[i] = tq[j] + lam[k] * (tq[j + 1] - tq[j])
    return best_d, best_t


def orbit_project(P, Q, tq):
    """For each point of P find the nearest point on the polygon Q (times tq)."""
    tree = cKDTree(Q)
    _, j = tree.query(P)
    best_d = np.full(len(P), np.inf); best_t = np.zeros(len(P))
    for off in (-1, 0):
        a = np.clip(j + off, 0, len(Q) - 2)
        A, B = Q[a], Q[a + 1]
        AB = B - A
        L2 = np.maximum((AB * AB).sum(1), 1e-300)
        lam = np.clip(((P - A) * AB).sum(1) / L2, 0, 1)
        X = A + lam[:, None] * AB
        dd = np.abs(P - X).max(1)
        tt = tq[a] + lam * (tq[a + 1] - tq[a])
        upd = dd < best_d
        best_d[upd], best_t[upd] = dd[upd], tt[upd]
    return best_d, best_t


rows = []
for name, f, J, x0, span, xe, marg in cases:
    tg = np.linspace(span[0], span[1], 50001)
    tq = np.linspace(span[0], span[1] + marg, 50001 + int(50000 * marg / (span[1] - span[0])))
    for m in ('RK45', 'LSODA', 'BDF', 'Radau'):
        for tol in (1e-4, 1e-6):
            sols = []
            for tt in (tol, tol / 10):
                kw = dict(rtol=tt, atol=tt * RATIO, dense_output=True)
                if m != 'RK45':
                    kw['jac'] = J
                sols.append(solve_ivp(f, (span[0], span[1] + marg), x0, method=m, **kw))
            if not all(s.success for s in sols):
                print(name, m, tol, 'solver failure'); continue
            U1, U2 = sols[0].sol(tg).T, sols[1].sol(tg).T
            U2q = sols[1].sol(tq).T
            sc = np.maximum(np.abs(U2).max(0), 1e-300)
            P, Q = U1 / sc, U2 / sc
            Etime = np.abs(P - Q).max()
            dist, s = orbit_track(P, U2q / sc, tq)
            Eorbit = dist.max()
            D12 = s - tg
            # relative phase at the end of the motion: where u1 is closest to its mid-range
            if xe == 'lorenz':
                event1, k = True, len(tg) - 1
            elif xe is not None:
                event1 = bool((U1[:, 0] >= xe).any())
                k = int(np.argmax(U1[:, 0] >= xe))   # run 1 reaches the event level
            else:
                event1 = U1[:, 1].max() > 0.1 * U2[:, 1].max()
                k = np.argmax(U1[:, 1])          # SIR: peak of I
            Dend = D12[k]
            # where the relative phase is created: first time |D12| reaches half of |Dend|
            idx = np.where(np.abs(D12[:k + 1]) >= 0.5 * abs(Dend))[0]
            t50 = tg[idx[0]] / tg[k] if (len(idx) and abs(Dend) > 0) else np.nan
            if not event1:
                Dend, t50 = np.inf, np.nan
            verdict = ('phase error' if (Eorbit < 0.1 * Etime and Etime > 10 * tol)
                       else ('error at tolerance level' if Etime <= 10 * tol else 'not phase-dominated'))
            rows.append(dict(case=name, method=m, tol=tol, Etime=Etime, Eorbit=Eorbit,
                             ratio=Eorbit / Etime, D12_event=Dend, t50_frac=t50, verdict=verdict))
            print(f"{name:18s} {m:6s} {tol:6.0e}  Etime={Etime:9.2e} Eorbit={Eorbit:9.2e} "
                  f"ratio={Eorbit/Etime:8.1e}  D12={Dend:+10.3e}  t50={t50:6.3f}  {verdict}", flush=True)

import os
if only and os.path.exists('../results/orbit_test.csv'):
    with open('../results/orbit_test.csv') as fh:
        old = [r for r in csv.DictReader(fh) if r['case'] not in only]
    rows = old + rows
with open('../results/orbit_test.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
