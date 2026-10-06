"""Descriptive per-security outcomes; never a substitute for ranking calibration.

Input is the backtest's already cooled-down, completed signal sample. These
records cover all historical patterns, not today's specific setup. Keeping this
report separate prevents pooled rates from masquerading as stock-specific data.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import math

import config as C
import risk_stats
import strategy

VERSION = 2


def build(signals: list[dict]) -> dict:
    groups = defaultdict(list)
    for signal in signals:
        ex = signal.get("exit") or {}
        net = ex.get("net")
        if (signal.get("code") and signal.get("date") and ex.get("label_end")
                and ex.get("closed") and ex.get("entered", True)
                and isinstance(net, (int, float)) and math.isfinite(net)):
            groups[signal["code"]].append(signal)
    by_code = {}
    for code, rows in groups.items():
        nets = [s["exit"]["net"] for s in rows]
        n = len(nets)
        flags = [strategy.outcome_success(s["exit"]) for s in rows]
        successes = sum(flags)
        stats = {**risk_stats.losses(nets), **risk_stats.risk_reward(nets, flags)}
        # No losing observations means no observed conditional loss magnitude,
        # not a proven zero-size future loss.
        if not stats["losses"]:
            stats["avg_loss_magnitude"] = None
        by_code[code] = dict(stats, code=code, samples=n, successes=successes,
                            success_rate=round(100 * successes / n, 1),
                            late_exits=sum(s["exit"].get("days", 0) > C.EXIT_MAX_DAYS for s in rows),
                            expectancy=round(sum(nets) / n, 3),
                            signal_dates=len({s["date"] for s in rows}),
                            first_signal=min(s["date"] for s in rows),
                            last_signal=max(s["date"] for s in rows),
                            outcomes_through=max(s["exit"]["label_end"] for s in rows))
    return {"version": VERSION, "scope": "security_all_patterns_completed",
            "cooldown_days": C.SIGNAL_COOLDOWN_DAYS, "by_code": by_code}


def available(backtest: dict) -> bool:
    report = backtest.get("stock_history") or {}
    return (report.get("version") == VERSION
            and report.get("scope") == "security_all_patterns_completed"
            and report.get("cooldown_days") == C.SIGNAL_COOLDOWN_DAYS
            and isinstance(report.get("by_code"), dict))


def attach(rows: list[dict], backtest: dict | None) -> None:
    usable = bool(backtest and strategy.compatible(backtest)
                  and backtest.get("mode") == "live" and available(backtest))
    by_code = backtest["stock_history"]["by_code"] if usable else {}
    for row in rows:
        row["stock_history"] = None
        stats = by_code.get(row.get("code"))
        # Explicitly refuse a report with outcomes later than this quote date.
        if (stats and stats.get("code") == row.get("code") and stats.get("samples", 0) > 0
                and row.get("quote_date") and stats.get("outcomes_through")
                and stats["outcomes_through"] <= row["quote_date"]):
            row["stock_history"] = deepcopy(stats)
