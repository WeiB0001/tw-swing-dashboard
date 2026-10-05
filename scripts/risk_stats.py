"""Loss disclosure and a date-block bootstrap for research return estimates."""
from __future__ import annotations

import math
import numpy as np
import config as C
import strategy


def losses(nets: list[float]) -> dict:
    a = np.asarray(nets, dtype=float)
    n = len(a)
    if not n:
        return {"losses": 0, "loss_rate": None, "loss_rate_upper": None,
                "tail_mean_5pct": None, "worst_net": None}
    count = int((a < -1e-9).sum())
    # Even zero historical losses leaves a positive uncertainty bound.
    upper = 100 - strategy.wilson_lower(n - count, n)
    tail = np.sort(a)[:max(1, math.ceil(.05 * n))]
    return {"losses": count, "loss_rate": round(100 * count / n, 2),
            "loss_rate_upper": round(upper, 2),
            "tail_mean_5pct": round(float(tail.mean()), 3),
            "worst_net": round(float(a.min()), 3)}


def expected_return_lower(signals: list[dict]) -> dict:
    """Resample whole dates in circular 10-date blocks, preserving stock clusters.

    This is an uncertainty estimate, not a bound on future loss. Serial dependence
    beyond the chosen block and changes in the market can invalidate it.
    """
    daily = {}
    for s in signals:
        if not s.get("date") or not s.get("exit", {}).get("closed", True):
            continue
        row = daily.setdefault(s["date"], [0., 0])
        row[0] += float(s["exit"]["net"])
        row[1] += 1
    n = len(daily)
    if n < C.MIN_CALIBRATION_DATES:
        return {"signal_dates": n, "ev_lower": None}
    values = np.asarray([daily[d] for d in sorted(daily)], dtype=float)
    block = min(C.EV_BOOTSTRAP_BLOCK_DAYS, n)
    rng = np.random.default_rng(20261005)
    starts = rng.integers(0, n, size=(C.EV_BOOTSTRAP_REPS, math.ceil(n / block)))
    ix = ((starts[:, :, None] + np.arange(block)) % n).reshape(C.EV_BOOTSTRAP_REPS, -1)[:, :n]
    sampled = values[ix].sum(axis=1)
    estimates = sampled[:, 0] / sampled[:, 1]
    return {"signal_dates": n, "ev_lower": round(float(np.quantile(estimates, .05)), 4)}


def rank_order_check(groups: list[dict]) -> dict:
    enough = len(groups) == 4 and all(g.get("samples", 0) >= C.MIN_SAMPLES_SCORE
                                     and g.get("expectancy") is not None for g in groups)
    values = [g["expectancy"] for g in groups] if enough else []
    monotonic = bool(enough and all(a >= b for a, b in zip(values, values[1:]))
                     and values[0] > values[-1])
    return {"available": bool(enough), "monotonic": monotonic,
            "front_minus_back": round(values[0] - values[-1], 3) if enough else None}
