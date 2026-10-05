"""Daily-bar execution model. Decisions use yesterday's close, fills today's open.

No price, volume or fill model is a guarantee of execution. A one-price bar is
conservatively treated as unavailable in the adverse direction. Unknown corporate
actions remain explicit audit flags; large losses are never silently removed.
"""
from __future__ import annotations

import math
import config as C


def tax_pct(code=""):
    return C.ETF_SELL_TAX_PCT if str(code).startswith("00") else C.STOCK_SELL_TAX_PCT


def fee(value):
    return max(C.MIN_BROKER_FEE_TWD, value * C.BROKER_FEE_PCT / 100) if value > 0 else 0.


def buy_cash(price, shares):
    value = price * (1 + C.SLIPPAGE_PCT / 200) * shares
    return value + fee(value)


def sell_cash(price, shares, code=""):
    value = price * (1 - C.SLIPPAGE_PCT / 200) * shares
    return value - fee(value) - value * tax_pct(code) / 100


def net_return(entry, exit_price, shares=None, code=""):
    shares = shares or max(1, int(C.REFERENCE_NOTIONAL_TWD / entry))
    paid = buy_cash(entry, shares)
    return (sell_cash(exit_price, shares, code) / paid - 1) * 100


def price_for_net(entry, target, shares=None, code=""):
    """Indicative trigger price including minimum fees, never a promised fill."""
    shares = shares or max(1, int(C.REFERENCE_NOTIONAL_TWD / entry))
    desired = buy_cash(entry, shares) * (1 + target / 100)
    lo, hi = 0., entry * max(2., 2 + target / 100)
    for _ in range(60):
        mid = (lo + hi) / 2
        if sell_cash(mid, shares, code) < desired:
            lo = mid
        else:
            hi = mid
    return hi


def tradable(bar, side, prev_close=None):
    try:
        px = float(bar["open"])
        if not math.isfinite(px) or px <= 0:
            return False
        vol = bar.get("volume")
        if vol is not None and (not math.isfinite(float(vol)) or float(vol) <= 0):
            return False
        high, low = float(bar["high"]), float(bar["low"])
        if not (math.isfinite(high) and math.isfinite(low) and 0 < low <= px <= high):
            return False
        if high == low and prev_close:
            if (side == "buy" and px >= prev_close) or (side == "sell" and px <= prev_close):
                return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def close_decision(entry, close, held, shares=None, code="", stop_pct=None):
    net = net_return(entry, close, shares, code)
    if stop_pct is not None and net <= -stop_pct:
        return "風險停損"
    # Earliest actual exit is day 2, based on day 1 close. Day 10 exit is
    # precommitted by day 9 close, rather than observing day 10's future close.
    if held + 1 >= C.EXIT_MIN_DAYS and net >= C.EXIT_MIN_PROFIT - 1e-9:
        return "淨利觸發"
    if held + 1 >= C.EXIT_MAX_DAYS:
        return "持有到期"
    return None


def make_entry_plan(reference, ma20=None):
    """Freeze the ceiling at signal time; the fill must include adverse slippage."""
    try:
        reference = float(reference)
        if not math.isfinite(reference) or reference <= 0:
            raise ValueError()
        ceiling = reference * (1 + C.ENTRY_MAX_PREMIUM_PCT / 100)
        moving = float(ma20) if ma20 is not None else None
        if moving is not None:
            if not math.isfinite(moving) or moving <= 0:
                raise ValueError()
            ceiling = min(ceiling, moving * (1 + C.ENTRY_MAX_MA20_BIAS_PCT / 100))
        return {"available": True, "reference_price": reference, "ma20": moving,
                "max_fill_price": ceiling,
                "max_open_price": ceiling / (1 + C.SLIPPAGE_PCT / 200),
                "max_premium_pct": C.ENTRY_MAX_PREMIUM_PCT,
                "max_ma20_bias_pct": C.ENTRY_MAX_MA20_BIAS_PCT}
    except (TypeError, ValueError):
        return {"available": False, "reason": "訊號參考價格不足，不進場"}


def history_entry_plan(df, signal_pos):
    if signal_pos < 0:
        return None  # Low-level execution fixtures can start at an already chosen entry.
    row = df.iloc[signal_pos]
    moving = row.get("ma20")
    if moving is None and signal_pos >= 19:
        moving = df["close"].iloc[signal_pos-19:signal_pos+1].mean(skipna=False)
    return make_entry_plan(row.get("close"), moving)


def check_entry_price(open_price, plan):
    if not plan or not plan.get("available"):
        return "訊號參考價格不足，不進場"
    try:
        fill = float(open_price) * (1 + C.SLIPPAGE_PCT / 200)
        if not math.isfinite(fill) or fill <= 0:
            return "進場價格無效"
        if fill > plan["max_fill_price"] + 1e-9:
            return "超過進場上限，不追價"
    except (KeyError, TypeError, ValueError):
        return "訊號參考價格不足，不進場"
    return None


def simulate(df, entry_pos, code="", stop_pct="default", require_mature=True,
             entry_plan=None, guard=True):
    if entry_pos < 0 or entry_pos >= len(df):
        return None
    horizon = C.EXIT_MAX_DAYS + C.EXIT_FILL_GRACE_DAYS
    if require_mature and entry_pos + horizon > len(df):
        return {"closed": False}
    stop = C.STOP_LOSS_NET_PCT if stop_pct == "default" else stop_pct
    window = df.iloc[entry_pos:entry_pos + horizon]
    first = window.iloc[0]
    previous = float(df["close"].iloc[entry_pos - 1]) if entry_pos else None
    if not tradable(first, "buy", previous):
        return {"closed": False, "entered": False, "reason": "進場無法成交"}
    entry = float(first["open"])
    frozen_plan = entry_plan if entry_plan is not None else history_entry_plan(df, entry_pos-1)
    if guard and (entry_pos > 0 or entry_plan is not None):
        rejected = check_entry_price(entry, frozen_plan)
        if rejected:
            return {"closed": False, "entered": False, "entry_skipped": True,
                    "reason": rejected, "entry_plan": frozen_plan}
    shares = max(1, int(C.REFERENCE_NOTIONAL_TWD / entry))
    pending, trigger_date, worst, delay = None, None, 0., 0
    for held, (date, bar) in enumerate(window.iterrows(), 1):
        if pending:
            if tradable(bar, "sell", previous):
                px = float(bar["open"])
                net = net_return(entry, px, shares, code)
                worst = min(worst, (px / entry - 1) * 100)
                return {"closed": True, "entered": True, "success": net >= C.EXIT_MIN_PROFIT - 1e-9,
                        "days": held, "net": net, "mdd": worst, "reason": pending,
                        "entry_date": str(window.index[0])[:10], "entry": entry,
                        "exit_price": px, "shares": shares, "trigger_date": trigger_date,
                        "exit_date": str(date)[:10], "delayed_days": delay,
                        "label_end": str(df.index[min(len(df)-1, entry_pos+horizon-1)])[:10]}
            delay += 1
        try:
            close, low = float(bar["close"]), float(bar["low"])
            if not (math.isfinite(close) and math.isfinite(low) and close > 0 and low > 0):
                return {"closed": False, "entered": True, "reason": "持有中行情缺漏"}
        except (KeyError, TypeError, ValueError):
            return {"closed": False, "entered": True, "reason": "持有中行情缺漏"}
        worst = min(worst, (low / entry - 1) * 100)
        if not pending:
            pending = close_decision(entry, close, held, shares, code, stop)
            if pending:
                trigger_date = str(date)[:10]
        previous = close
    return {"closed": False, "entered": True, "reason": "出場尚未成交" if pending else "持有中",
            "days": len(window), "mark_net": net_return(entry, previous, shares, code),
            "mdd": worst, "delayed_days": delay}


def audit(hist_map):
    issues = []
    invalid = 0
    for code, df in hist_map.items():
        if df is None or df.empty:
            continue
        previous = None
        for date, bar in df.iterrows():
            vals = [float(bar.get(k, float("nan"))) for k in ("open", "high", "low", "close")]
            o, h, l, c = vals
            if not all(math.isfinite(x) and x > 0 for x in vals) or not l <= min(o,c) <= max(o,c) <= h:
                invalid += 1
                issues.append({"code": code, "date": str(date)[:10], "reason": "OHLC 不完整或不一致"})
            elif previous and abs(o / previous - 1) > .25:
                issues.append({"code": code, "date": str(date)[:10], "reason": "開盤跳動逾 25%，待核對公司行動", "gap_pct": round((o/previous-1)*100,2)})
            elif abs(c / o - 1) > .25:
                issues.append({"code": code, "date": str(date)[:10], "reason": "日內價格變動逾 25%，待核對行情與公司行動"})
            previous = c if math.isfinite(c) and c > 0 else None
    return {"issues": issues[:100], "issue_count": len(issues), "invalid_bars": invalid,
            "corporate_actions_adjusted": False, "point_in_time_universe": False,
            "note": "保留有效行情中的大幅虧損；原始日線尚未整合完整公司行動及歷史成分股。異常待核對，不能據此認定策略有效。"}
