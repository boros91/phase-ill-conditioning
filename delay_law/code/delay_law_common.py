"""Integrated delay law for scalar problems.

Under local error control with error scale s(x), a step at the state x accepts a local error of about
tol*s(x), i.e. a time defect tol*s/|f| = tol*Theta(x), and has a length of about
tol^(1/(p+1)) / |f'(x)| (the local rate of a scalar solution is |f'|).  Hence the time defect per unit
time is (tol*sigma)^alpha, alpha = p/(p+1), with

    sigma(x) = s(x) |f'(x)| / |f(x)| = Theta(x) |f'(x)|,

and the delay at the event is D ~ K tol^alpha int_0^Te sigma^alpha dt.  In units of the event's time
resolution, eta ~ K tol^alpha Lambda_alpha with

    Lambda_alpha = int_0^Te sigma(x(t))^alpha dt / Theta(T_e) = int_{x0}^{xe} sigma^alpha dx/|f| / Theta(x_e).

For s = |x|, sigma = |E| is the modulus of the elasticity E = x f'/f.
"""
import numpy as np
from scipy.integrate import quad


def scale(x, ratio, conv):
    return np.abs(x) + ratio if conv == 'scipy' else np.maximum(np.abs(x), ratio)


def Lambda(f, fp, x0, xe, alpha, ratio, conv='scipy', pts=None):
    """Lambda_alpha for x' = f(x) from x0 to the level xe (x increasing)."""
    sig = lambda x: scale(x, ratio, conv) * abs(fp(x)) / abs(f(x))
    g = lambda x: sig(x)**alpha / abs(f(x))
    # integrate in log-spaced pieces for x0 >= 0, else split at the given points
    if pts is None and x0 > 0:
        edges = np.geomspace(x0, xe, 60)
    else:
        edges = np.unique(np.concatenate([[x0, xe], [p for p in (pts or []) if x0 < p < xe]]))
        fine = []
        for a, b in zip(edges[:-1], edges[1:]):
            fine.append(np.linspace(a, b, 41))
        edges = np.unique(np.concatenate(fine))
    val = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        v, _ = quad(g, a, b, limit=200, epsabs=0, epsrel=1e-10)
        val += v
    Th_e = scale(xe, ratio, conv) / abs(f(xe))
    return val / Th_e


def phase_ratio(f, x0, xe, ratio, conv='scipy', pts=None):
    """t_* / Theta(x_e): the length of the passage in units of the event's time resolution."""
    return _tstar(f, x0, xe, pts) / (scale(xe, ratio, conv) / abs(f(xe)))


def _tstar(f, x0, xe, pts=None):
    edges = np.geomspace(x0, xe, 60) if (pts is None and x0 > 0) else np.linspace(x0, xe, 2001)
    return sum(quad(lambda x: 1 / abs(f(x)), a, b, limit=200, epsrel=1e-10)[0] for a, b in zip(edges[:-1], edges[1:]))


def flame(q):
    return (lambda x: x**q * (1 - x)), (lambda x: q * x**(q - 1) - (q + 1) * x**q)


def ghost(eps):
    return (lambda x: eps + x * x), (lambda x: 2 * x)


def logistic():
    return (lambda x: x * (1 - x)), (lambda x: 1 - 2 * x)
