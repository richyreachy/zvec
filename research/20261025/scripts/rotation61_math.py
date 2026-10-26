from __future__ import annotations

import logging

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
D = 768
PAIRS = []
for stride in [1, 16, 384]:
    left = np.concatenate([np.arange(i, i + stride) for i in range(0, D, 2 * stride)])
    PAIRS.append((left, left + stride))


def forward(x, theta, cache=False):
    z = x.copy()
    saved = []
    for angle, (left, right) in zip(theta, PAIRS, strict=True):
        c = np.cos(angle)
        s = np.sin(angle)
        a = z[:, left].copy()
        b = z[:, right].copy()
        aa = c * a - s * b
        bb = s * a + c * b
        z[:, left] = aa
        z[:, right] = bb
        if cache:
            saved.append((c, s, aa, bb))
    return (z, saved)


def objective(x, theta, gradient=False):
    z, saved = forward(x, theta, gradient)
    l1_norm = np.sum(np.abs(z), axis=1)
    losses = D / l1_norm**2 - 1
    if not gradient:
        return losses
    g = -2 * D / len(z) * np.sign(z) / l1_norm[:, None] ** 3
    result = np.zeros_like(theta)
    for layer in range(2, -1, -1):
        left, right = PAIRS[layer]
        c, s, aa, bb = saved[layer]
        ga = g[:, left].copy()
        gb = g[:, right].copy()
        result[layer] = np.sum(-ga * bb + gb * aa, axis=0)
        g[:, left] = ga * c + gb * s
        g[:, right] = -ga * s + gb * c
    return (float(np.mean(losses)), result)


def local_comparisons(residuals, gaps, theta):
    transformed = []
    for j in range(3):
        transformed.append(forward(residuals[:, j], theta)[0].astype(np.float64))
    a, b, q = transformed
    errors = []
    for r in [a, b]:
        norm = np.sum(r * r, axis=1)
        l1_norm = np.sum(np.abs(r), axis=1)
        amplitude = np.divide(norm, l1_norm, out=np.zeros_like(norm), where=l1_norm > 0)
        h = amplitude[:, None] * np.where(r > 0, 1.0, -1.0)
        errors.append(h - r)
    approximate = gaps - np.sum(q * (errors[0] - errors[1]), axis=1)
    valid = np.abs(gaps) > 1e-12
    flips = np.sign(approximate[valid]) != np.sign(gaps[valid])
    return (
        flips,
        float(np.mean((approximate[valid] - gaps[valid]) ** 2)),
        int(np.sum(~valid)),
    )
