"""SIR model S' = -b S I, I' = b S I - g I (b = 0.3, g = 0.1): exact time on the orbit,
reference solution and adjoint for the event S = g/b (peak of I).  Same formulas as exp_sir.py."""
import numpy as np
from scipy.integrate import quad, solve_ivp

b, g = 0.3, 0.1
c = g / b
Se = c


def make_T(I0):
    S0 = 1.0 - I0

    def Ifun(z):
        return I0 + z + c * np.log1p(-z / S0)

    def T(S):
        Z = S0 - S
        if Z <= 0:
            return 0.0
        U = np.log1p(Z / I0)
        h = lambda u: (I0 * np.exp(u)) / (b * (S0 - I0 * np.expm1(u)) * Ifun(I0 * np.expm1(u)))
        pts = np.linspace(0, U, 9)[1:-1]
        val, _ = quad(h, 0.0, U, points=pts, limit=1000, epsabs=1e-13, epsrel=1e-13)
        return val
    return S0, T


def rhs(t, x):
    S, I = x
    return np.array([-b * S * I, b * S * I - g * I])


def jac(t, x):
    S, I = x
    return np.array([[-b * I, -b * S], [b * I, b * S - g]])


def adjoint_on_grid(I0, npts=20001):
    """Reference orbit X(t), adjoint LAM(t) (lambda(T_e) = e_S / f_S) on a grid up to T_e."""
    S0, T = make_T(I0)
    Te = T(Se)
    lr = lambda t, u: np.array([-b * np.exp(u[1]), b * np.exp(u[0]) - g])
    sol = solve_ivp(lr, [0, Te], [np.log(S0), np.log(I0)], method='DOP853',
                    rtol=1e-13, atol=1e-13, dense_output=True)
    Xf = lambda t: np.exp(sol.sol(t))
    xe = Xf(Te); fe = rhs(Te, xe)
    lamT = np.array([1.0 / fe[0], 0.0])
    sa = solve_ivp(lambda t, lam: -jac(t, Xf(t)).T @ lam, [Te, 0.0], lamT, method='DOP853',
                   rtol=1e-11, atol=1e-30, dense_output=True)
    tg = np.unique(np.concatenate([np.linspace(0, Te, npts), Te * (1 - np.geomspace(1e-9, 1, 2000))]))
    return dict(T=T, Te=Te, S0=S0, tg=tg, X=Xf(tg), LAM=sa.sol(tg))


def Pi(A, ratio, conv='scipy', unknown='x'):
    """Phase condition number for error scale s = |u| + ratio (scipy) or max(|u|, ratio) (matlab),
    in the unknowns u = x or u = log x (then a perturbation du is dx = x du)."""
    X, LAM = A['X'], A['LAM']
    if unknown == 'x':
        U, L = X, LAM
    else:
        U, L = np.log(X), LAM * X
    S = np.abs(U) + ratio if conv == 'scipy' else np.maximum(np.abs(U), ratio)
    th = (np.abs(L) * S).sum(0)
    return th.max() / th[-1], th[-1]
