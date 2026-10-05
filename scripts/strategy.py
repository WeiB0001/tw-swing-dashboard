"""Shared target-return contract for ranking, validation and paper trading."""
from __future__ import annotations

import math
import config as C
import execution

VERSION = "executable-risk-v4"


def contract() -> dict:
    if not 1 <= C.EXIT_MIN_DAYS <= C.EXIT_MAX_DAYS:
        raise ValueError("持有天數設定不合法")
    if C.STOP_LOSS_NET_PCT is not None and C.STOP_LOSS_NET_PCT <= 0:
        raise ValueError("停損幅度須為正數；不啟用請設 None")
    if not math.isfinite(C.LOSS_AVERSION) or C.LOSS_AVERSION < 1:
        raise ValueError("虧損加權須為至少 1 的有限數值")
    return {"version": VERSION, "target_net_pct": float(C.EXIT_MIN_PROFIT),
            "min_days": int(C.EXIT_MIN_DAYS), "max_days": int(C.EXIT_MAX_DAYS),
            "cost_pct": float(C.TOTAL_COST_PCT), "exit_rule": "close_signal_next_open_day10_open",
            "fill_grace_days": C.EXIT_FILL_GRACE_DAYS,
            "execution": {"fee_pct": C.BROKER_FEE_PCT, "minimum_fee": C.MIN_BROKER_FEE_TWD,
                          "stock_tax_pct": C.STOCK_SELL_TAX_PCT, "etf_tax_pct": C.ETF_SELL_TAX_PCT,
                          "slippage_pct": C.SLIPPAGE_PCT, "reference_notional": C.REFERENCE_NOTIONAL_TWD},
            "risk_budget": {"per_trade_pct": C.RISK_PER_TRADE_PCT, "total_pct": C.MAX_TOTAL_RISK_PCT,
                            "position_pct": C.MAX_POSITION_PCT, "sector_pct": C.MAX_SECTOR_POSITION_PCT,
                            "gap_buffer_pct": C.RISK_GAP_BUFFER_PCT, "slots": C.PAPER_MAX_POSITIONS},
            "stop_loss_net_pct": C.STOP_LOSS_NET_PCT,
            "ranking": "target_gain_minus_weighted_loss", "loss_aversion": C.LOSS_AVERSION,
            "min_samples": C.MIN_SAMPLES_SCORE,
            "min_dates": C.MIN_CALIBRATION_DATES,
            "shrink_k": C.SHRINK_K, "bootstrap_block_days": C.EV_BOOTSTRAP_BLOCK_DAYS,
            "bootstrap_reps": C.EV_BOOTSTRAP_REPS}


def compatible(data: dict | None) -> bool:
    return bool(data) and data.get("strategy") == contract()


def net_return(entry: float, exit_price: float, cost: float | None = None) -> float:
    # Explicit legacy flat costs remain available only for fixed-horizon reports.
    return execution.net_return(entry, exit_price) if cost is None else (exit_price / entry - 1) * 100 - cost


def target_met(net: float) -> bool:
    # Inclusive threshold with only floating-point rounding tolerance.
    return math.isfinite(net) and net >= C.EXIT_MIN_PROFIT - 1e-9


def exit_reason(net: float, days: int) -> str | None:
    # A stop is a trigger, not a cap: record the actual closing loss, including gaps.
    if C.STOP_LOSS_NET_PCT is not None and net <= -C.STOP_LOSS_NET_PCT:
        return "風險停損"
    if days >= C.EXIT_MIN_DAYS and target_met(net):
        return "淨利達標"
    return "持有到期" if days >= C.EXIT_MAX_DAYS else None


def capital_policy() -> dict:
    if C.REQUIRE_NO_LOSS:
        return {"blocked": True, "require_no_loss": True,
                "reason": "零虧損要求：股票無法證明零風險，保持空手；僅提供研究排名"}
    return {"blocked": False, "require_no_loss": False,
            "reason": "採用風險報酬綜合排名；僅通過研究與樣本外驗證的標的可配置試算，仍可能虧損"}


def utility(net: float) -> float:
    """Target gains earn credit, sub-target gains earn zero, losses count fully.

    This is a preference score in percentage points, not a forecast return.
    """
    return net if target_met(net) else C.LOSS_AVERSION * min(net, 0.0)


def outcome(df, entry_pos: int, cost: float | None = None, code="") -> dict | None:
    """Compatibility entry point; costs now come from the versioned cash model.

    The old positional cost argument is deliberately not used for target trades.
    Backtests, published outcomes and paper books share execution.simulate.
    """
    return execution.simulate(df, entry_pos, code)


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
