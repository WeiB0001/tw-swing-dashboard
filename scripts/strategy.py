"""Shared target-return contract for ranking, validation and paper trading."""
from __future__ import annotations

import math
import config as C

VERSION = "net-target-v1"


def contract() -> dict:
    return {"version": VERSION, "target_net_pct": float(C.EXIT_MIN_PROFIT),
            "min_days": int(C.EXIT_MIN_DAYS), "max_days": int(C.EXIT_MAX_DAYS),
            "cost_pct": float(C.TOTAL_COST_PCT), "exit_rule": "daily_close"}


def compatible(data: dict | None) -> bool:
    return bool(data) and data.get("strategy") == contract()


def net_return(entry: float, exit_price: float, cost: float | None = None) -> float:
    return (exit_price / entry - 1) * 100 - (C.TOTAL_COST_PCT if cost is None else cost)


def target_met(net: float) -> bool:
    # Inclusive threshold with only floating-point rounding tolerance.
    return math.isfinite(net) and net >= C.EXIT_MIN_PROFIT - 1e-9


def exit_reason(net: float, days: int) -> str | None:
    if days >= C.EXIT_MIN_DAYS and target_met(net):
        return "淨利達標"
    return "持有到期" if days >= C.EXIT_MAX_DAYS else None


def outcome(df, entry_pos: int, cost: float | None = None) -> dict | None:
    """Require a fully matured cohort, including trades that hit the target early.

    Entry day counts as day 1. Closing-price fills are a simulation assumption;
    actual execution can differ. Never score an incomplete winning cohort.
    """
    if entry_pos < 0 or entry_pos >= len(df):
        return None
    end = entry_pos + C.EXIT_MAX_DAYS
    if end > len(df):
        return {"closed": False}
    window = df.iloc[entry_pos:end]
    entry = float(window["open"].iloc[0])
    values = [entry] + [float(x) for key in ("close", "low") for x in window[key]]
    if any(not math.isfinite(x) or x <= 0 for x in values):
        return None
    worst = 0.0
    for d, (date, row) in enumerate(window.iterrows(), 1):
        worst = min(worst, (float(row["low"]) / entry - 1) * 100)
        net = net_return(entry, float(row["close"]), cost)
        reason = exit_reason(net, d)
        if reason:
            return {"success": target_met(net), "days": d, "net": net,
                    "mdd": worst, "closed": True, "reason": reason,
                    "exit_date": str(date)[:10],
                    "label_end": str(window.index[-1])[:10]}
    return None


def wilson_lower(successes: int, samples: int, z: float = 1.6448536269514722) -> float:
    """One-sided 95% Wilson lower bound; ranking score, not a promised probability."""
    if samples <= 0:
        return 0.0
    p = successes / samples
    z2 = z * z
    return 100 * (p + z2 / (2 * samples) - z * math.sqrt(
        p * (1 - p) / samples + z2 / (4 * samples * samples))) / (1 + z2 / samples)


if __name__ == "__main__":
    import json
    from pathlib import Path
    import sys
    try:
        saved = json.loads((Path(__file__).resolve().parents[1] / C.BACKTEST_JSON).read_text())
        valid = compatible(saved) and saved.get("mode") == "live"
    except (OSError, ValueError):
        valid = False
    print("回測口徑相符" if valid else "回測需重算：目標、期間、成本或策略版本不同")
    sys.exit(0 if valid else 1)
