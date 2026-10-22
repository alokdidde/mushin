"""Metrics over collected predictions (plain Python and NumPy, no model code).

Choice: accuracy against the hard target, NLL, multi-class Brier, top-label ECE, and the
risk-coverage curve when the least confident decisions are deferred (AURC: area under it).
Yes/no: Brier, ECE, AUROC and accuracy of P(success) per skill option.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

ECE_BINS = 15


def ece(confidence: Sequence[float], correct: Sequence[float], bins: int = ECE_BINS) -> float:
    confidence, correct = np.asarray(confidence, float), np.asarray(correct, float)
    if not len(confidence):
        return math.nan
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        inside = (confidence > low) & (confidence <= high) if low > 0 else confidence <= high
        if inside.any():
            total += inside.mean() * abs(confidence[inside].mean() - correct[inside].mean())
    return float(total)


def auroc(scores: Sequence[float], labels: Sequence[float]) -> float:
    scores, labels = np.asarray(scores, float), np.asarray(labels, float)
    positives, negatives = labels.sum(), (1 - labels).sum()
    if positives == 0 or negatives == 0:
        return math.nan
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    for value in np.unique(scores):  # average ranks of ties
        tied = scores == value
        ranks[tied] = ranks[tied].mean()
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2)
                 / (positives * negatives))


def risk_coverage(confidence: Sequence[float], correct: Sequence[float]) -> dict[str, float]:
    confidence, correct = np.asarray(confidence, float), np.asarray(correct, float)
    order = np.argsort(-confidence)
    risks = 1 - np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
    out = {"aurc": float(risks.mean()) if len(risks) else math.nan}
    for coverage in (0.5, 0.8, 1.0):
        k = max(1, int(round(coverage * len(order))))
        out[f"risk@{coverage:g}"] = float(risks[k - 1]) if len(risks) else math.nan
    return out


def choice_metrics(probs: Sequence[Sequence[float]], hard: Sequence[int],
                   soft: Sequence[Sequence[float]]) -> dict[str, float]:
    top = [int(np.argmax(p)) for p in probs]
    correct = [float(t == h) for t, h in zip(top, hard, strict=True)]
    confidence = [float(max(p)) for p in probs]
    nll = [-math.log(max(p[h], 1e-12)) for p, h in zip(probs, hard, strict=True)]
    brier = [float(sum((pi - (i == h)) ** 2 for i, pi in enumerate(p)))
             for p, h in zip(probs, hard, strict=True)]
    soft_nll = [-sum(q * math.log(max(pi, 1e-12)) for pi, q in zip(p, s, strict=True))
                for p, s in zip(probs, soft, strict=True)]
    return {
        "n": len(probs),
        "accuracy": float(np.mean(correct)) if correct else math.nan,
        "nll": float(np.mean(nll)) if nll else math.nan,
        "soft_nll": float(np.mean(soft_nll)) if soft_nll else math.nan,
        "brier": float(np.mean(brier)) if brier else math.nan,
        "ece": ece(confidence, correct),
        **risk_coverage(confidence, correct),
    }


def success_metrics(probs: Sequence[float], labels: Sequence[float]) -> dict[str, float]:
    p, y = np.asarray(probs, float), np.asarray(labels, float)
    if not len(p):
        return {"n": 0}
    return {
        "n": len(p),
        "accuracy": float(((p >= 0.5) == (y == 1)).mean()),
        "brier": float(((p - y) ** 2).mean()),
        "ece": ece(p, y),
        "auroc": auroc(p, y),
        "base_rate": float(y.mean()),
    }
