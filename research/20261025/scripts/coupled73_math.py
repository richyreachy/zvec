from __future__ import annotations

import numpy as np


def reconstruct(r, s):
    norm = np.sum(r * r, axis=1)
    denom = np.sum(r * s, axis=1)
    assert np.all((norm == 0) | (denom > 0))
    alpha = np.divide(norm, denom, out=np.zeros_like(norm), where=norm > 0)
    return (alpha[:, None] * s, alpha)


def objective(r, s, Q, lam):
    h, _ = reconstruct(r, s)
    e = h - r
    projection = e @ Q.T
    projection -= projection.mean(axis=0)
    return float(np.sum(projection * projection) + lam * np.sum(e * e))


def flip_costs(r, s, Q, lam, i, max_alpha=np.inf):
    K, d = r.shape
    h, alpha = reconstruct(r, s)
    e = h - r
    ri = r[i]
    si = s[i]
    norm = ri @ ri
    if norm == 0:
        return np.full(d, np.inf)
    beta = 1 - 1 / K
    u = ri @ si
    qs = Q @ si
    qr = Q @ ri
    other = e.sum(axis=0) - e[i]
    As = beta * (Q.T @ qs) + lam * si
    diag = beta * np.sum(Q * Q, axis=0) + lam
    g = beta * (Q.T @ qr) + lam * ri + Q.T @ (Q @ other) / K
    B = beta * (qs @ qs) + lam * d
    C = si @ g
    unew = u - 2 * si * ri
    anew = np.divide(norm, unew, out=np.full(d, np.inf), where=unew > 0)
    Bnew = B - 4 * si * As + 4 * diag
    Cnew = C - 2 * si * g
    valid = (unew > 0) & (anew <= max_alpha)
    out = np.full(d, np.inf)
    out[valid] = (
        anew[valid] ** 2 * Bnew[valid]
        - 2 * anew[valid] * Cnew[valid]
        - (alpha[i] ** 2 * B - 2 * alpha[i] * C)
    )
    return out


def optimize(r, Q, lam, sweeps=2, per_edge_per_sweep=4, scale_cap=1.02):
    s = np.where(r > 0, 1.0, -1.0)
    _, a0 = reconstruct(r, s)
    history = [objective(r, s, Q, lam)]
    accepted = []
    for sweep in range(sweeps):
        for i in range(len(r)):
            for _ in range(per_edge_per_sweep):
                costs = flip_costs(r, s, Q, lam, i, a0[i] * scale_cap)
                k = int(np.argmin(costs))
                if not np.isfinite(costs[k]) or costs[k] >= -1e-13 * max(
                    1, history[-1]
                ):
                    break
                before = history[-1]
                trial = s.copy()
                trial[i, k] *= -1
                after = objective(r, trial, Q, lam)
                assert abs(after - before - costs[k]) < 1e-09 * max(
                    1, abs(before), abs(after)
                )
                if after >= before:
                    break
                s = trial
                history.append(after)
                accepted.append((sweep, i, k))
    return (s, history, accepted)
