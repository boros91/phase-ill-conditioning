"""Check of Proposition (idealized controller) for scalar problems.

A method of order p is combined with an error estimate whose leading term is A(x) h^(m+1); the idealized
controller takes h(x) = (tol*s(x)/|A(x)|)^(1/(m+1)) exactly.  If the propagated local error has the leading
term B(x) h^(p+1), the delay at the event is predicted as
    D_pred = - tol^alpha * int_{x0}^{xe} (B(x)/f(x)) (s(x)/|A(x)|)^alpha dx / f(x),   alpha = p/(m+1).
Two pairs:
  Euler  : explicit Euler with its own leading term as estimate,  A = B = -f f'/2, m = p = 1, alpha = 1/2
  HeunEx : Heun propagated, Euler-Heun difference as estimate (local extrapolation),
           A = f f'/2, B = f'' f^2/12 - f'^2 f/6, m = 1, p = 2, alpha = 1
Problems: flame y' = y^2 (1-y), delta = 1e-2..1e-6, event 1/2; logistic x' = x(1-x), x0 = 1e-3..1e-7, event 0.4.
Error scales: s = |x| (relative) and s = |x| + 1e-2 (AbsTol/RelTol = 1e-2).
Output: ../results/ideal_controller.csv, ../results/ideal_controller.txt
"""
import numpy as np
import pandas as pd
from scipy.integrate import quad
import phase_common as pc

rows = []


def pieces(x0, xe, n=80):
    return np.geomspace(x0, xe, n)


for prob_name, pars, xe in (('flame', (1e-2, 1e-4, 1e-6), 0.5), ('logistic', (1e-3, 1e-5, 1e-7), 0.4)):
    for par in pars:
        P = pc.flame_q(2, par) if prob_name == 'flame' else pc.logistic(par)
        f, fp = P.f, P.fp
        fpp = (lambda x: 2 - 6 * x) if prob_name == 'flame' else (lambda x: -2.0 + 0 * x)
        for ratio in (0.0, 1e-2):
            s = lambda x: abs(x) + ratio
            for pair in ('Euler', 'HeunEx'):
                if pair == 'Euler':
                    A = lambda x: -f(x) * fp(x) / 2; B = A; m, p = 1, 1
                else:
                    A = lambda x: f(x) * fp(x) / 2; B = lambda x: fpp(x) * f(x)**2 / 12 - fp(x)**2 * f(x) / 6; m, p = 1, 2
                alpha = p / (m + 1)
                g = lambda x: -(B(x) / f(x)) * (s(x) / abs(A(x)))**alpha / f(x)
                e = pieces(P.x0, xe)
                I = sum(quad(g, a, b, epsrel=1e-11, limit=200)[0] for a, b in zip(e[:-1], e[1:]))
                for tol in (1e-4, 1e-6, 1e-8):
                    x, t, n = P.x0, 0.0, 0
                    while x < xe and n < 50_000_000:
                        h = (tol * s(x) / abs(A(x)))**(1 / (m + 1))
                        if pair == 'Euler':
                            xn = x + h * f(x)
                        else:
                            k1 = f(x); xn = x + h / 2 * (k1 + f(x + h * k1))
                        x = xn; t += h; n += 1
                    D = t - P.T(x)
                    pred = tol**alpha * I
                    rows.append(dict(problem=prob_name, par=par, ratio=ratio, pair=pair, alpha=alpha, tol=tol,
                                     steps=n, D=D, pred=pred, ratio_meas_pred=D / pred))
                    print(f'{prob_name:8s} {par:.0e} s-ratio={ratio:g} {pair:6s} tol={tol:.0e}: D={D:+.4e} '
                          f'pred={pred:+.4e} ratio={D/pred:.4f} steps={n}', flush=True)
R = pd.DataFrame(rows)
R.to_csv('../results/ideal_controller.csv', index=False)
txt = ['Idealized controller: measured delay / predicted leading term', '']
for (pair, tol), G in R.groupby(['pair', 'tol']):
    txt.append(f'{pair:6s} tol={tol:.0e}: ratio min {G.ratio_meas_pred.min():.4f} max {G.ratio_meas_pred.max():.4f} '
               f'(n={len(G)}); |D| from {G.D.abs().min():.2e} to {G.D.abs().max():.2e}; steps {G.steps.min()}..{G.steps.max()}')
open('../results/ideal_controller.txt', 'w').write('\n'.join(txt) + '\n')
print('\n'.join(txt))
