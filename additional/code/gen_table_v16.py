"""Table of the phase space error control runs (v16): ../results/v16_pscontrol.csv -> ../tables/table_v16_pscontrol.tex
and the numbers quoted in the text -> ../results/v16_numbers.txt"""
import os
import numpy as np
import pandas as pd
H = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(H, "..", "results", "v16_pscontrol.csv"))
d = d[d.tol == 1e-4]


def sci(x):
    if not np.isfinite(x):
        return "--"
    if 0.01 <= abs(x) < 1000:
        return f"${x:.2g}$" if abs(x) < 100 else f"${x:.0f}$"
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\cdot10^{{{int(e)}}}$"


def get(pair, prob, par, mode, phi=None, theta=None):
    x = d[(d.pair == pair) & (d.problem == prob) & (d.par == par) & (d["mode"] == mode)]
    if phi is not None:
        x = x[x.phi == phi]
    if theta is not None:
        x = x[x.theta == theta]
    assert len(x) == 1, (pair, prob, par, mode, phi, theta, len(x))
    return x.iloc[0]


rows = [("flame, $\\delta=10^{-2}$", "flame", 1e-2), ("flame, $\\delta=10^{-4}$", "flame", 1e-4), ("flame, $\\delta=10^{-6}$", "flame", 1e-6),
        ("logistic, $x_0=10^{-5}$", "logistic", 1e-5), ("logistic, $x_0=10^{-7}$", "logistic", 1e-7),
        ("SIR, $I_0=10^{-6}$", "SIR", 1e-6), ("SIR, $I_0=10^{-8}$", "SIR", 1e-8),
        ("ghost, $\\varepsilon=10^{-6}$", "ghost", 1e-6), ("ghost, $\\varepsilon=10^{-8}$", "ghost", 1e-8),
        ("Morris--Lecar, $\\Delta=10^{-2}$", "Morris-Lecar", 1e-2)]
with open(os.path.join(H, "..", "tables", "table_v16_pscontrol.tex"), "w") as fh:
    fh.write("\\begin{tabular}{llrcccrcr}\n\\toprule\n"
             "& & \\multicolumn{2}{c}{standard} & arc length & trapezoidal & \\multicolumn{3}{c}{$\\theta=1$}\\\\\n"
             "\\cmidrule(lr){3-4}\\cmidrule(lr){7-9}\n"
             "pair & problem & steps & $\\eta$ & $\\eta$, $\\varphi=0.01$ & $\\eta$, $\\varphi=0.1$ & steps & $\\eta$, $\\varphi=0.1$ & $\\eta$, $\\varphi=0.01$\\\\\n\\midrule\n")
    for pair, lab in (("RK23", "3(2)"), ("RK45", "5(4)")):
        for i, (name, prob, par) in enumerate(rows):
            s = get(pair, prob, par, "std"); a = get(pair, prob, par, "arc", 0.01)
            t = get(pair, prob, par, "ps", 0.1, 0.5); p1 = get(pair, prob, par, "ps", 0.1, 1.0); p2 = get(pair, prob, par, "ps", 0.01, 1.0)
            fh.write(f"{lab if i == 0 else ''} & {name} & {s.steps} & {sci(s.eta)} & {sci(a.eta)} & {sci(t.eta)} & {p1.steps} & {sci(p1.eta)} & {sci(p2.eta)}\\\\\n")
        if pair == "RK23":
            fh.write("\\midrule\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

out = []
full = pd.read_csv(os.path.join(H, "..", "results", "v16_pscontrol.csv"))
out.append(f"runs: {len(full)}, without event: {int((~full.event).sum())}")
for pair in ("RK23", "RK45"):
    for phi in (0.1, 0.03, 0.01):
        x = full[(full.pair == pair) & (full["mode"] == "ps") & (full.theta == 1.0) & (full.phi == phi) &
                 (((full.problem == "flame") & (full.par == 1e-6)) | ((full.problem == "logistic") & (full.par == 1e-7)) | ((full.problem == "SIR") & (full.par == 1e-8)))]
        out.append(f"{pair} theta=1 phi={phi:g}: |D|/T_e over flame 1e-6, logistic 1e-7, SIR 1e-8 and both tolerances: "
                   f"min {x.rel.min():.2e} max {x.rel.max():.2e} median {x.rel.median():.2e}; steps {x.steps.min()}..{x.steps.max()}")
    x = d[(d.pair == pair)]
    st = x[x["mode"] == "std"].set_index(["problem", "par"]).steps
    p1 = x[(x["mode"] == "ps") & (x.theta == 1.0) & (x.phi == 0.1)].set_index(["problem", "par"]).steps
    r = (p1 / st)
    out.append(f"{pair} tol 1e-4: steps theta=1, phi=0.1 over standard: {r.min():.2f}..{r.max():.2f} (median {r.median():.2f})")
    for mode, phi, th, nm in (("arc", 0.1, None, "arc 0.1"), ("arc", 0.01, None, "arc 0.01"), ("ps", 0.1, 0.5, "trapezoidal 0.1"), ("ps", 0.1, 1.0, "theta=1 0.1")):
        y = x[(x["mode"] == mode) & (x.phi == phi)]
        if th is not None:
            y = y[y.theta == th]
        e0 = x[x["mode"] == "std"].set_index(["problem", "par"]).eta; e1 = y.set_index(["problem", "par"]).eta
        ch = (e1 / e0)
        out.append(f"{pair} tol 1e-4 {nm}: eta/eta_standard: unchanged (within 1%) in {int(((ch > 0.99) & (ch < 1.01)).sum())} of {len(ch)} problems; "
                   f"smallest ratio {ch.min():.2e}; largest eta {e1.max():.3g}")
open(os.path.join(H, "..", "results", "v16_numbers.txt"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
