"""Common definitions for the stage-2 study of phase ill-conditioning.

Scalar autonomous test problems x' = f(x) with an exact time function T(x),
a step-by-step driver for the SciPy integrators, and the timing sensitivity
Theta(x) = s(x)/|f(x)| with the SciPy error scale s(x) = |x| + atol/rtol.
"""
import numpy as np
from scipy.integrate import RK45, DOP853, LSODA, BDF, Radau

METHODS = {'RK45': RK45, 'DOP853': DOP853, 'LSODA': LSODA, 'BDF': BDF, 'Radau': Radau}


def theta_q(y, q):
    """Antiderivative of 1/(y^q (1-y)) on (0,1) (eq. (6) of flame_A_v21)."""
    return np.log(y / (1.0 - y)) - sum(1.0 / (j * y**j) for j in range(1, q))


class Scalar:
    """x' = f(x), x(0) = x0, event: first time x reaches xe (x increasing)."""

    def __init__(self, name, f, fp, x0, xe, T, grid, label):
        self.name, self.f, self.fp, self.x0, self.xe = name, f, fp, x0, xe
        self.T = T            # exact time to reach level x
        self.grid = grid      # points x in [x0, xe] for the index
        self.label = label

    def Theta(self, x, ratio):
        """Time shift of the event caused by a perturbation of unit size in the
        solver's scale s(x) = |x| + atol/rtol (ratio = atol/rtol)."""
        return (np.abs(x) + ratio) / np.abs(self.f(x))

    def Pi(self, ratio):
        th = self.Theta(self.grid, ratio)
        return th.max() / self.Theta(np.array([self.xe]), ratio)[0], self.grid[np.argmax(th)]


def flame_q(q, delta):
    f = lambda x: x**q * (1.0 - x)
    fp = lambda x: q * x**(q - 1) - (q + 1) * x**q
    T = lambda x: theta_q(x, q) - theta_q(delta, q)
    grid = np.unique(np.concatenate([np.geomspace(delta, 0.5, 4000), np.linspace(delta, 0.5, 4000)]))
    return Scalar(f'flame_q{q}_d{delta:.0e}', f, fp, delta, 0.5, T, grid,
                  rf"$y'=y^{q}(1-y)$, $\delta={delta:.0e}$")


def ghost(eps):
    se = np.sqrt(eps)
    f = lambda x: eps + x * x
    fp = lambda x: 2.0 * x
    T = lambda x: (np.arctan(x / se) + np.arctan(1.0 / se)) / se
    grid = np.unique(np.concatenate([np.linspace(-1, 1, 20001),
                                     np.sign(np.linspace(-1, 1, 2001)) * np.geomspace(1e-12, 1, 2001)[::1][:2001]]))
    grid = grid[(grid >= -1) & (grid <= 1)]
    return Scalar(f'ghost_e{eps:.0e}', f, fp, -1.0, 1.0, T, grid,
                  rf"$x'=\varepsilon+x^2$, $\varepsilon={eps:.0e}$")


def logistic(x0):
    f = lambda x: x * (1.0 - x)
    fp = lambda x: 1.0 - 2.0 * x
    T = lambda x: np.log(x / (1.0 - x)) - np.log(x0 / (1.0 - x0))
    grid = np.unique(np.concatenate([np.geomspace(x0, 0.5, 4000), np.linspace(x0, 0.5, 4000)]))
    return Scalar(f'logistic_x{x0:.0e}', f, fp, x0, 0.5, T, grid,
                  rf"$x'=x(1-x)$, $x_0={x0:.0e}$")


def run_to_level(prob, method, rtol, atol, t_bound=1e12, max_steps=2_000_000):
    """Integrate step by step until the first accepted step with x >= xe.
    Returns accepted (t_k, x_k), the exact delays D_k = t_k - T(x_k) and nfev."""
    cls = METHODS[method]
    fun = lambda t, y: np.array([prob.f(y[0])])
    kw = dict(rtol=rtol, atol=atol)
    if method in ('BDF', 'Radau', 'LSODA'):
        kw['jac'] = lambda t, y: np.array([[prob.fp(y[0])]])
    s = cls(fun, 0.0, np.array([prob.x0]), t_bound, **kw)
    ts, xs = [0.0], [prob.x0]
    n = 0
    while s.status == 'running' and n < max_steps:
        s.step()
        n += 1
        ts.append(s.t)
        xs.append(s.y[0])
        if s.y[0] >= prob.xe or not np.isfinite(s.y[0]):
            break
    ts, xs = np.array(ts), np.array(xs)
    ok = xs[-1] >= prob.xe and xs[-1] < 1.0 + 1e-12 if prob.name.startswith(('flame', 'logistic')) else xs[-1] >= prob.xe
    return ts, xs, s.nfev, n, bool(ok)
