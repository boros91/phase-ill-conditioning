"""Adiabatic constant-pressure H2/air ignition with the Cantera mechanism h2o2.yaml.
Unknowns x = (T, Y_1..Y_10).  Right-hand side written explicitly so that any
SciPy integrator, error scale and change of unknowns can be used."""
import numpy as np
import cantera as ct

gas = ct.Solution('h2o2.yaml')
P0 = ct.one_atm
T0 = 1000.0
gas.TPX = T0, P0, 'H2:2, O2:1, N2:3.76'
Y0 = gas.Y.copy()
x0 = np.concatenate([[T0], Y0])
W = gas.molecular_weights
N = gas.n_species


def rhs(t, x):
    """Right-hand side; a state Cantera cannot evaluate (non-positive T or density) returns NaN,
    which makes the integrators reject the step instead of stopping."""
    T = x[0]
    Y = x[1:]
    try:
        gas.set_unnormalized_mass_fractions(Y)
        gas.TP = T, P0
    except ct.CanteraError:
        return np.full(len(x), np.nan)
    rho = gas.density
    wdot = gas.net_production_rates
    dY = wdot * W / rho
    dT = -np.dot(gas.partial_molar_enthalpies, wdot) / (rho * gas.cp_mass)
    return np.concatenate([[dT], dY])


def jac_fd(t, x, f0=None):
    """Forward-difference Jacobian (used for the adjoint and by the implicit codes)."""
    if f0 is None:
        f0 = rhs(t, x)
    n = len(x)
    J = np.empty((n, n))
    for j in range(n):
        h = 1e-7 * max(abs(x[j]), 1e-12 if j else 1.0)
        if j:
            h = max(h, 1e-14)
        xp = x.copy(); xp[j] += h
        J[:, j] = (rhs(t, xp) - f0) / h
    return J
