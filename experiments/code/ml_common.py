"""Morris-Lecar neuron, class I parameter set (Rinzel & Ermentrout), near the
saddle-node on invariant circle (SNIC) bifurcation."""
import numpy as np
from scipy.optimize import brentq, minimize_scalar

C, gL, gCa, gK = 20.0, 2.0, 4.0, 8.0
VL, VCa, VK = -60.0, 120.0, -84.0
V1, V2, V3, V4, phi = -1.2, 18.0, 12.0, 17.4, 1.0 / 15.0

minf = lambda V: 0.5 * (1 + np.tanh((V - V1) / V2))
winf = lambda V: 0.5 * (1 + np.tanh((V - V3) / V4))
tauw = lambda V: 1.0 / np.cosh((V - V3) / (2 * V4))


def make(I):
    def f(t, x):
        V, w = x
        dV = (I - gL * (V - VL) - gCa * minf(V) * (V - VCa) - gK * w * (V - VK)) / C
        dw = phi * (winf(V) - w) / tauw(V)
        return np.array([dV, dw])

    def J(t, x, h=1e-7):
        f0 = f(t, x); out = np.empty((2, 2))
        for j in range(2):
            xp = np.array(x, float); dx = h * max(1.0, abs(xp[j])); xp[j] += dx
            out[:, j] = (f(t, xp) - f0) / dx
        return out
    return f, J


def Iss(V):
    return gL * (V - VL) + gCa * minf(V) * (V - VCa) + gK * winf(V) * (V - VK)


def snic_current():
    """I at the lower knee of the steady-state I-V curve (saddle-node of equilibria)."""
    r = minimize_scalar(lambda V: -Iss(V), bounds=(-40, 0), method='bounded', options=dict(xatol=1e-12))
    return Iss(r.x), r.x


def reference(Delta, x0, VE=10.0, nsp=10):
    """Reference spike times (DOP853, rtol 1e-13) from x0 (a point at a spike), the time
    function on each upstroke, and the adjoint over the period before the first spike."""
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq
    Ic, _ = snic_current()
    f, J = make(Ic + Delta)
    P_est = np.pi / np.sqrt(Delta) * 60
    ref = solve_ivp(f, [0, (nsp + 0.5) * P_est * 1.2], x0, method='DOP853', rtol=1e-13, atol=1e-15,
                    dense_output=True)
    V = ref.y[0]; up = np.where((V[:-1] < VE) & (V[1:] >= VE))[0]
    tsp = [brentq(lambda t: ref.sol(t)[0] - VE, ref.t[i], ref.t[i + 1], xtol=1e-13) for i in up]
    tsp = np.array([t for t in tsp if t > 0.3 * P_est])[:nsp]
    P = tsp[0]

    def tref_level(n, Vval):
        tn = tsp[n - 1]
        g = lambda t: ref.sol(t)[0] - Vval
        a = tn
        while g(a) > 0 and a > tn - 0.2 * P:
            a -= 0.05
        b = tn
        while g(b) < 0 and b < tn + 0.05 * P:
            b += 0.05
        return brentq(g, a, b, xtol=1e-13)

    xe = ref.sol(P); fe = f(P, xe)
    sa = solve_ivp(lambda tau, lam: J(P - tau, ref.sol(P - tau)).T @ lam, [0, P],
                   np.array([1.0 / fe[0], 0.0]), method='DOP853', rtol=1e-10, atol=1e-14, dense_output=True)
    tg = np.linspace(0, P, 20001)
    LAM = np.array([sa.sol(P - t) for t in tg]).T
    return dict(tsp=tsp, P=P, tref_level=tref_level, tg=tg, X=ref.sol(tg), LAM=LAM)


def Pi_from(R, ratio, conv='scipy'):
    S = np.abs(R['X']) + ratio if conv == 'scipy' else np.maximum(np.abs(R['X']), ratio)
    th = (np.abs(R['LAM']) * S).sum(0)
    return th.max() / th[-1], th[-1]
