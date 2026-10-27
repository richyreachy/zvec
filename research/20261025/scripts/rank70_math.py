from __future__ import annotations

import logging

import numpy as np
from rotation61_math import PAIRS, forward

logging.basicConfig(level=logging.INFO, format="%(message)s")


def backward(g, saved):
    result = np.zeros((3, 384), dtype=np.float64)
    for layer in range(2, -1, -1):
        left, right = PAIRS[layer]
        c, s, aa, bb = saved[layer]
        ga = g[:, left].copy()
        gb = g[:, right].copy()
        result[layer] = np.sum(-ga * bb + gb * aa, axis=0)
        g[:, left] = ga * c + gb * s
        g[:, right] = -ga * s + gb * c
    return result


def evaluate(
    residuals,
    queries,
    offsets,
    truth,
    theta,
    gradient=False,
    temperature=0.01,
    tolerance=1e-07,
):
    B, K, D = residuals.shape
    Q = queries.shape[1]
    zr, sr = forward(residuals.reshape(-1, D), theta, gradient)
    zq, sq = forward(queries.reshape(-1, D), theta, gradient)
    zr = zr.reshape(B, K, D)
    zq = zq.reshape(B, Q, D)
    sign = np.where(zr > 0, 1.0, -1.0)
    norm2 = np.sum(residuals * residuals, axis=2)
    l1 = np.sum(np.abs(zr), axis=2)
    amplitude = np.divide(norm2, l1, out=np.zeros_like(norm2), where=l1 > 0)
    h = amplitude[:, :, None] * sign
    dots = np.einsum("bqd,bkd->bqk", zq, sign, optimize=True)
    scores = offsets[:, None, :] - dots * amplitude[:, None, :]
    targets = truth <= np.min(truth, axis=2, keepdims=True) + tolerance
    weights = targets / np.sum(targets, axis=2, keepdims=True)
    logits = -scores / temperature
    shift = logits.max(axis=2, keepdims=True)
    exp = np.exp(logits - shift)
    prob = exp / exp.sum(axis=2, keepdims=True)
    logprob = logits - shift - np.log(exp.sum(axis=2, keepdims=True))
    loss = float(np.mean(-np.sum(weights * logprob, axis=2)))
    if not gradient:
        return (loss, scores)
    gd = (weights - prob) / (B * Q * temperature)
    gq = -np.einsum("bqk,bkd->bqd", gd, h, optimize=True)
    coefficient = np.divide(
        norm2, l1 * l1, out=np.zeros_like(norm2), where=l1 > 0
    ) * np.sum(gd * dots, axis=1)
    gr = coefficient[:, :, None] * sign
    grad = backward(gr.reshape(-1, D), sr) + backward(gq.reshape(-1, D), sq)
    return (loss, grad)
