"""Run accounting of v12 (Table SM2 and the middle band of Sect. 7.2).
Inputs (paths relative to the repository root):
  experiments/results/index_scalar.csv, sir_runs.csv, matlab_phase_index.csv, matlab_pi_flame.csv,
  matlab_pi_sir.csv; delay_law/results/delay_law_fit_runs.csv, delay_law_systems_runs.csv
Usage: python gen_accounting_v12.py [<repository root>]   -> prints the numbers; writes accounting_v12.txt"""
import sys, os
import numpy as np
import pandas as pd

root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "..")
S = os.path.join(root, "experiments", "results"); R = os.path.join(root, "delay_law", "results")
out = []
def say(s):
    print(s); out.append(s)

def bands(df, name, col="tolPi"):
    for nm, d in (("r*Pi < 1e-3", df[df[col] < 1e-3]), ("1e-3 <= r*Pi < 1", df[(df[col] >= 1e-3) & (df[col] < 1)]),
                  ("r*Pi >= 1", df[df[col] >= 1])):
        e = d.eta.dropna()
        say(f"{name:22s} {nm:18s} runs {len(d):4d}, with event {len(e):4d}, max eta {e.max():.3g}, median {e.median():.3g}, "
            f"fraction eta > 0.1: {(e > 0.1).mean():.2f}")

a = pd.read_csv(os.path.join(S, "index_scalar.csv")); s = pd.read_csv(os.path.join(S, "sir_runs.csv"))
say(f"SciPy scalar: {len(a)} runs, {int((~a.ok).sum())} without event; SIR: {len(s)} runs, {int((~s.ok).sum())} without event")
bands(pd.concat([a[["tolPi", "eta"]], s[["tolPi", "eta"]]]), "SciPy, all")
mf = pd.read_csv(os.path.join(S, "matlab_pi_flame.csv"))
say(f"MATLAB flame: {len(mf)} runs, {int((mf.completed == 1).sum())} completed")
bands(mf[mf.completed == 1], "MATLAB, flame")
ms = pd.read_csv(os.path.join(S, "matlab_pi_sir.csv"))
say(f"MATLAB SIR (first set): {len(ms)} runs, {int(ms.D.isna().sum())} without event")
m = pd.read_csv(os.path.join(S, "matlab_phase_index.csv")); m.loc[m.event == 0, "eta"] = np.nan
say(f"MATLAB second set: {len(m)} runs, {int((m.event == 0).sum())} without event; by problem {m.groupby('problem').size().to_dict()}")
bands(m, "MATLAB, second set")
f = pd.read_csv(os.path.join(R, "delay_law_fit_runs.csv")); inc = f[~f.res_M2.isna()]
say(f"delay-law fit: {len(f)} scalar runs with event ({f.groupby('env').size().to_dict()}), {len(inc)} used "
    f"({inc.groupby('env').size().to_dict()}); smallest eta used {inc.eta.min():.3g}, largest eta left out {f[f.res_M2.isna()].eta.max():.3g}")
y = pd.read_csv(os.path.join(R, "delay_law_systems_runs.csv"))
say(f"systems: {len(y)} runs: {y.groupby(['env', 'problem']).size().to_dict()}")
open(os.path.join(os.path.dirname(__file__), "..", "results", "accounting_v12.txt"), "w").write("\n".join(out) + "\n")
