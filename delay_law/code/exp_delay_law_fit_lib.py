"""Problem table and alpha grid shared with exp_delay_law_fit.py."""
import numpy as np
import delay_law_common as L

ALPHAS = np.round(np.arange(0.30, 1.501, 0.05), 2)


def problem(name, par):
    if name == 'flame2':
        f, fp = L.flame(2); return f, fp, par, 0.5, None
    if name == 'flame3':
        f, fp = L.flame(3); return f, fp, par, 0.5, None
    if name == 'ghost':
        f, fp = L.ghost(par); return f, fp, -1.0, 1.0, [0.0]
    if name == 'logistic':
        f, fp = L.logistic(); return f, fp, par, 0.5, None
    raise ValueError(name)
