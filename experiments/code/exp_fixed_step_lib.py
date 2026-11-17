"""Fixed-step drivers for scalar problems (same code as in exp_fixed_step.py)."""


def step_scalar(meth, f, fp, x, h):
    if meth == 'EE':
        return x + h * f(x)
    if meth == 'Heun':
        k1 = f(x); k2 = f(x + h * k1); return x + h / 2 * (k1 + k2)
    if meth == 'RK4':
        k1 = f(x); k2 = f(x + h / 2 * k1); k3 = f(x + h / 2 * k2); k4 = f(x + h * k3)
        return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    if meth == 'IE':
        y = x + h * f(x)
        for _ in range(50):
            dy = (y - x - h * f(y)) / (1 - h * fp(y)); y -= dy
            if abs(dy) < 1e-15 * max(1.0, abs(y)):
                break
        return y


def run_scalar(p, meth, h, xe, nmax=5_000_000):
    x, t, n = p.x0, 0.0, 0
    while x < xe and n < nmax:
        x = step_scalar(meth, p.f, p.fp, x, h); t += h; n += 1
    return t - p.T(x), n
