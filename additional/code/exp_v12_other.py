"""Indicators for the two examples taken over from the companion paper (theta neuron and
May-Leonard cycle), computed with phasecond.  MATLAB error scale, RelTol 1e-4, AbsTol 1e-6
(the benchmark tolerance of those runs).  Output: ../results/v12_other.txt"""
import os, sys
import numpy as np
sys.path.insert(0, os.environ.get("PHASECOND", os.path.join(os.path.dirname(__file__), "..", "..", "phasecond")))
from phasecond import scalar_report, phase_report

out = []
def say(s):
    print(s, flush=True); out.append(s)

R, A = 1e-4, 1e-6
# theta neuron: theta' = 1 - cos(theta) + (1 + cos(theta)) I, theta(0) = -pi, k-th spike at theta = (2k-1) pi
for I in (1e-4, 1e-6):
    f = lambda th: 1 - np.cos(th) + (1 + np.cos(th)) * I
    for k in (1, 2, 10):
        r = scalar_report(f, -np.pi, (2 * k - 1) * np.pi, R, A, scale="matlab")
        say(f"theta neuron I={I:g} spike {k}: T_e={r.T_e:.6g} (k*pi/sqrt(I)={k*np.pi/np.sqrt(I):.6g}) Theta_e={r.Theta_e:.4g} "
            f"Pi={r.Pi:.4g} at theta={r.x_max:.4g} rtolPi={r.rtolPi:.3g} Pi_rel={r.Pi_rel:.4g}")

# May-Leonard: x_i' = x_i (1 - x_i - a x_{i+1} - b x_{i+2}), a = 0.8, b = 1.3
a, b = 0.8, 1.3
def ml(x):
    return np.array([x[i] * (1 - x[i] - a * x[(i + 1) % 3] - b * x[(i + 2) % 3]) for i in range(3)])
def mljac(x):
    J = np.zeros((3, 3))
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        J[i, i] = 1 - 2 * x[i] - a * x[j] - b * x[k]; J[i, j] = -a * x[i]; J[i, k] = -b * x[i]
    return J
x0 = [0.6, 0.3, 0.1]
# switch n: the incoming component passes 1/2 upwards; find which component rises at each switch
from scipy.integrate import solve_ivp
lr = lambda t, u: np.array([1 - np.exp(u[i]) - a * np.exp(u[(i + 1) % 3]) - b * np.exp(u[(i + 2) % 3]) for i in range(3)])
ref = solve_ivp(lr, (0, 1300), np.log(x0), method="DOP853", rtol=1e-13, atol=1e-13, dense_output=True)
tt = np.linspace(0, 1300, 2600001); X = np.exp(ref.sol(tt))
sw = []
for i in range(3):
    up = np.where((X[i, :-1] < 0.5) & (X[i, 1:] >= 0.5))[0]
    sw += [(tt[j], i) for j in up]
sw.sort()
say("May-Leonard switches (reference in log variables): " + ", ".join(f"{t:.1f}(x{i+1})" for t, i in sw))
rows = []
for n in range(1, 13):
    if n > len(sw):
        break
    ts, i = sw[n - 1]
    occ = 1 + sum(1 for t, j in sw[:n - 1] if j == i)
    try:
        r = phase_report(ml, x0, lambda x, i=i: x[i] - 0.5, ts * 1.02 + 1, R, A, scale="matlab", jac=mljac,
                         direction=1, occurrence=occ, ref_rtol=1e-12, ref_atol=1e-300, names=["x1", "x2", "x3"])
        rows.append((n, r.T_e, r.X.min(), r.Pi, r.Pi_star, r.Pi_rel, r.Pi_log, r.t_max, int(np.argmax(r.share_max)), r.share_max.max()))
        say(f"May-Leonard switch {n}: T_e={r.T_e:.6g} Theta_e={r.Theta_e:.4g} Pi={r.Pi:.4g} rtolPi={r.rtolPi:.3g} "
            f"Pi_*={r.Pi_star:.4g} Pi_rel={r.Pi_rel:.4g} Pi_log={r.Pi_log:.4g} min x on the way={r.X.min():.3g} "
            f"at max: t={r.t_max:.5g}, shares={np.round(r.share_max,3)} check={r.adjoint_check:.1e}")
    except Exception as e:
        say(f"May-Leonard switch {n}: failed ({str(e)[:80]})")
def sci(x):
    if 0.01 <= abs(x) < 1000:
        return f"${x:.3g}$"
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\cdot10^{{{int(e)}}}$"
with open(os.path.join(os.path.dirname(__file__), "..", "tables", "table_v12_mayleonard.tex"), "w") as fh:
    fh.write("\\begin{tabular}{rrcccccl}\n\\toprule\n$n$ & $T_e$ & smallest $x_i$ on the way & $\\Pi$ & $\\Pi_*$ & $\\Pi_{\\rm rel}$ & $\\Pi_{\\ln}$ & $\\Theta$ largest at\\\\\n\\midrule\n")
    for n, Te, xm, Pi, Ps, Pr, Pl, tm, ic, sh in rows:
        fh.write(f"{n} & {Te:.1f} & {sci(xm)} & {sci(Pi)} & {sci(Ps)} & {sci(Pr)} & {sci(Pl)} & $t={tm:.1f}$, {sh:.0%} from $x_{ic+1}$\\\\\n".replace("%", "\\%"))
    fh.write("\\bottomrule\n\\end{tabular}\n")
open(os.path.join(os.path.dirname(__file__), "..", "results", "v12_other.txt"), "w").write("\n".join(out) + "\n")
