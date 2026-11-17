"""Phase space error control (Higham, Humphries, Wain 2000; Humphries, Vigneswaran 2004) on the test
problems of the paper.

An embedded Runge-Kutta code with the standard control of SciPy (coefficients and error estimate of
scipy's RK23 and RK45, weighted RMS norm, scale atol + rtol*max(|y_n|, |y_{n+1}|)) is augmented by one
of the constraints
  arc     ||le_est||            <= phi ||y_{n+1} - y_n||                         [HHW00, (3.1)]
  ps      ||y_{n+1} - y_n - h fbar|| <= phi h ||fbar||, fbar = (1-theta) f_n + theta f_{n+1}
                                                                                 [HHW00 (theta = 1/2), HV04]
(max norm), with the step-size selection (46)-(48) of [HV04]: h_new = min(h_standard, h_constraint, ...),
h_constraint = (chi phi / R)^(1/qt) h, chi = 0.9.
The delay is measured as in the paper: D = t_n - T(c_n) at the first accepted step past the event.

Output: ../results/v16_pscontrol.csv, ../results/v16_pscontrol.txt, ../tables/table_v16_pscontrol.tex
"""
import csv, os, sys, time
import numpy as np
from scipy.integrate._ivp.rk import RK23, RK45
from scipy.integrate._ivp.common import select_initial_step

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get("PHASECOND", os.path.join(HERE, "..", "..", "phasecond")))
sys.path.insert(0, os.path.join(os.environ.get("PHASECOND", os.path.join(HERE, "..", "..", "phasecond")), "examples"))
import phase_common as pc
import sir_common as sc
import ml_common as mc
import problems as PR

PAIRS = {"RK23": RK23, "RK45": RK45}
CHI = 0.9


def integrate(f, y0, crossed, pair, rtol, atol, mode="std", phi=0.1, theta=0.5, t_stop=np.inf, max_steps=2_000_000):
    """Returns t, Y (accepted steps), nfev, nrej (standard), nrej_c (constraint), ok."""
    cls = PAIRS[pair]
    A, B, C, E = cls.A, cls.B, cls.C, cls.E
    s = cls.n_stages
    q = cls.error_estimator_order
    qt = q if mode == "arc" else (2 if theta == 0.5 else 1)
    y = np.array(y0, float); n = y.size
    fy = f(y); nfev = 1
    h = select_initial_step(lambda t, x: f(x), 0.0, y, t_stop if np.isfinite(t_stop) else 1e12, np.inf,
                            fy, 1, cls.order, rtol, atol)
    t = 0.0
    T = [t]; Y = [y.copy()]
    nrej = nrej_c = 0
    K = np.empty((s + 1, n))
    for _ in range(max_steps):
        K[0] = fy
        for i in range(1, s):
            K[i] = f(y + h * (K[:i].T @ A[i, :i]))
        ynew = y + h * (K[:s].T @ B)
        fnew = f(ynew); K[s] = fnew; nfev += s
        err = h * (K.T @ E)
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(ynew))
        errn = np.sqrt(np.mean((err / scale) ** 2))
        fac_s = 10.0 if errn == 0 else 0.9 * errn ** (-1.0 / (q + 1))
        if not np.all(np.isfinite(ynew)):
            errn, fac_s = np.inf, 0.2
        R = 0.0; fac_c = np.inf
        if mode == "arc":
            R = np.abs(err).max() / max(np.abs(ynew - y).max(), 1e-300)
        elif mode == "ps":
            fb = (1 - theta) * fy + theta * fnew
            R = np.abs(ynew - y - h * fb).max() / max(h * np.abs(fb).max(), 1e-300)
        if mode != "std":
            fac_c = np.inf if R == 0 else (CHI * phi / R) ** (1.0 / qt)
        ok_s = errn <= 1.0; ok_c = (mode == "std") or (R <= phi)
        if ok_s and ok_c:
            t += h; y = ynew; fy = fnew
            T.append(t); Y.append(y.copy())
            if crossed(y) or t >= t_stop:
                break
            h *= min(10.0, max(0.2, min(fac_s, fac_c)))
        else:
            nrej += (not ok_s); nrej_c += (ok_s and not ok_c)
            h *= max(0.2, min(fac_s if not ok_s else 1.0, fac_c if not ok_c else 1.0, 1.0) * (1.0 if (not ok_s) else 1.0))
            if h < 1e-300:
                break
    Y = np.array(Y).T
    return np.array(T), Y, nfev, nrej, nrej_c, bool(crossed(Y[:, -1]))


# ---------------------------------------------------------------- problems
def scalar_cases():
    out = []
    for d in (1e-2, 1e-4, 1e-6):
        out.append(("flame", "I", d, pc.flame_q(2, d)))
    for e in (1e-2, 1e-4, 1e-6, 1e-8):
        out.append(("ghost", "III", e, pc.ghost(e)))
    for x0 in (1e-1, 1e-3, 1e-5, 1e-7):
        out.append(("logistic", "II", x0, pc.logistic(x0)))
    return out


MODES = [("std", None, None), ("arc", 0.1, None), ("arc", 0.01, None), ("ps", 0.1, 0.5), ("ps", 0.1, 1.0),
         ("ps", 0.03, 1.0), ("ps", 0.01, 1.0)]
rows = []
log = []


def say(s):
    print(s, flush=True); log.append(s)


def record(prob, cls, par, pair, tol, mode, phi, theta, ok, D, Te, Theta_e, steps, nfev, nrej, nrej_c, note=""):
    rows.append(dict(problem=prob, cls=cls, par=par, pair=pair, tol=tol, mode=mode, phi=phi, theta=theta, event=ok,
                     D=D, Te=Te, Theta_e=Theta_e, eta=abs(D) / Theta_e if ok else np.nan,
                     rel=abs(D) / Te if ok else np.nan, steps=steps, nfev=nfev, nrej=nrej, nrej_c=nrej_c, note=note))


t0 = time.time()
for name, cls, par, p in scalar_cases():
    Te = p.T(p.xe)
    for tol in (1e-4, 1e-6):
        ratio = 1e-2
        Theta_e = p.Theta(np.array([p.xe]), ratio)[0]
        for pair in PAIRS:
            for mode, phi, theta in MODES:
                T, Y, nfev, nrej, nrej_c, ok = integrate(lambda y: np.array([p.f(y[0])]), [p.x0], lambda y: y[0] >= p.xe,
                                                         pair, tol, tol * ratio, mode, phi or 0, theta or 0.5,
                                                         t_stop=20 * Te)
                D = T[-1] - p.T(Y[0, -1]) if ok else np.nan
                record(name, cls, par, pair, tol, mode, phi, theta, ok, D, Te, Theta_e, T.size - 1, nfev, nrej, nrej_c)
    say(f"{name} {par:g} done ({time.time()-t0:.0f}s)")

for I0 in (1e-2, 1e-4, 1e-6, 1e-8):
    S0, Tfun = sc.make_T(I0); Te = Tfun(sc.Se)
    A_ = sc.adjoint_on_grid(I0, npts=4001)
    for tol in (1e-4, 1e-6):
        Theta_e = sc.Pi(A_, 1e-2)[1]
        for pair in PAIRS:
            for mode, phi, theta in MODES:
                T, Y, nfev, nrej, nrej_c, ok = integrate(lambda u: sc.rhs(0, u), [S0, I0], lambda u: u[0] <= sc.Se,
                                                         pair, tol, tol * 1e-2, mode, phi or 0, theta or 0.5, t_stop=20 * Te)
                D = T[-1] - Tfun(Y[0, -1]) if ok else np.nan
                record("SIR", "II", I0, pair, tol, mode, phi, theta, ok, D, Te, Theta_e, T.size - 1, nfev, nrej, nrej_c,
                       note=f"minI={Y[1].min():.2g}")
    say(f"SIR {I0:g} done ({time.time()-t0:.0f}s)")

for Delta in (1e-1, 1e-2):
    f = PR.morris_lecar(Delta); x0 = PR.ml_cycle_point(f)
    Rf = mc.reference(Delta, x0, nsp=2)
    Te = Rf["tsp"][0]; Theta_e = mc.Pi_from(Rf, 1e-2)[1]
    for tol in (1e-4, 1e-6):
        for pair in PAIRS:
            for mode, phi, theta in MODES:
                state = {"armed": False}
                def crossed(y):
                    if y[0] < 0:
                        state["armed"] = True
                    return state["armed"] and y[0] >= 10.0
                T, Y, nfev, nrej, nrej_c, ok = integrate(f, x0, crossed, pair, tol, tol * 1e-2, mode, phi or 0, theta or 0.5,
                                                         t_stop=5 * Te)
                D = T[-1] - Rf["tref_level"](1, Y[0, -1]) if ok else np.nan
                record("Morris-Lecar", "III", Delta, pair, tol, mode, phi, theta, ok, D, Te, Theta_e, T.size - 1, nfev, nrej, nrej_c)
    say(f"Morris-Lecar {Delta:g} done ({time.time()-t0:.0f}s)")

R_ = os.path.join(HERE, "..", "results")
with open(os.path.join(R_, "v16_pscontrol.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

# ---------------------------------------------------------------- summary
import pandas as pd
d = pd.DataFrame(rows)
d["m"] = d["mode"] + d.phi.map(lambda v: "" if pd.isna(v) else f" phi={v:g}") + d.theta.map(lambda v: "" if pd.isna(v) else f" theta={v:g}")
say(f"\n{len(d)} runs; without event: {int((~d.event).sum())}")
say(d[~d.event][["problem", "par", "pair", "tol", "m", "steps", "note"]].to_string())
for prob in ("flame", "logistic", "SIR", "ghost", "Morris-Lecar"):
    say(f"\n== {prob}: median over pairs and tolerances of |D|/T_e [largest] ; median eta ; median steps")
    g = d[d.problem == prob]
    for par in sorted(g.par.unique(), reverse=True):
        line = f"  par={par:8.0e}"
        for m in ["std", "arc phi=0.1", "arc phi=0.01", "ps phi=0.1 theta=0.5", "ps phi=0.1 theta=1", "ps phi=0.03 theta=1", "ps phi=0.01 theta=1"]:
            x = g[(g.par == par) & (g.m == m)]
            ev = x[x.event]
            line += f" | {m.replace(' phi=', ' ').replace(' theta=', '/')}: " + (
                f"{ev.rel.median():.1e} [{ev.rel.max():.1e}] eta {ev.eta.median():.1e} st {int(x.steps.median())}"
                + (f" LOST {len(x)-len(ev)}" if len(ev) < len(x) else "") if len(ev) else f"LOST {len(x)}")
        say(line)
say("\n== per pair, tol = 1e-4: eta (steps)")
for pair in PAIRS:
    g = d[(d.pair == pair) & (d.tol == 1e-4)]
    for (prob, par), x in g.groupby(["problem", "par"], sort=False):
        say(f"  {pair} {prob:12s} {par:7.0e} " + " | ".join(
            f"{r.m.replace(' phi=', ' ').replace(' theta=', '/')}: {r.eta:.2e} ({r.steps})" for r in x.itertuples()))
say("\n== relative delay |D|/T_e with ps theta=1 against phi (median over problems with the constraint active: flame 1e-6, logistic 1e-7, SIR 1e-8)")
for pair in PAIRS:
    for tol in (1e-4, 1e-6):
        line = f"  {pair} tol={tol:g}: "
        for phi in (0.1, 0.03, 0.01):
            x = d[(d.pair == pair) & (d.tol == tol) & (d["mode"] == "ps") & (d.theta == 1.0) & (d.phi == phi) &
                  (((d.problem == "flame") & (d.par == 1e-6)) | ((d.problem == "logistic") & (d.par == 1e-7)) | ((d.problem == "SIR") & (d.par == 1e-8)))]
            line += f"phi={phi:g}: " + ", ".join(f"{v:.2e}" for v in x.rel) + f" (steps {', '.join(str(v) for v in x.steps)});  "
        say(line)
from scipy.integrate import solve_ivp
pp = pc.flame_q(2, 1e-4)
for m in ("RK23", "RK45"):
    ss = solve_ivp(lambda t, y: [pp.f(y[0])], (0, 2e4), [1e-4], method=m, rtol=1e-4, atol=1e-6)
    k = int(np.argmax(ss.y[0] >= 0.5))
    mine = d[(d.problem == "flame") & (d.par == 1e-4) & (d.pair == m) & (d.tol == 1e-4) & (d["mode"] == "std")].iloc[0]
    say(f"check against scipy {m} (flame 1e-4, tol 1e-4): scipy D = {ss.t[k] - pp.T(ss.y[0][k]):+.5g} after {k} steps; this code D = {mine.D:+.5g} after {mine.steps} steps")
open(os.path.join(R_, "v16_pscontrol.txt"), "w").write("\n".join(log) + "\n")
