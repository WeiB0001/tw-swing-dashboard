"""Forward paper ledger for the same net-target exit used in backtests."""
from __future__ import annotations

from datetime import datetime
import pandas as pd
import config as C
import strategy


def update(pf, rows, trade_date, index_close, histories, published_at=None, capital_policy=None):
    from tracking import _blank_portfolio, _summarize
    if not strategy.compatible(pf):
        cancelled = [{**order, "cancel_reason": "策略版本更換，舊待成交單撤銷"}
                     for order in pf.get("pending", [])]
        pf = _blank_portfolio()
        pf["strategy"] = strategy.contract()
        pf["cancelled_pending"] = cancelled[-100:]
    policy = strategy.capital_policy()
    if not policy["blocked"]:
        if (capital_policy and strategy.compatible(capital_policy)
                and capital_policy.get("research_validated")):
            policy = dict(capital_policy)
        else:
            policy.update(blocked=True, reason="風險報酬綜合排名尚未通過樣本外驗證，暫不配置資金")
    pf["capital_policy"] = policy
    if policy["blocked"] and pf.get("pending"):
        pf.setdefault("cancelled_pending", []).extend(
            {**order, "cancel_reason": policy["reason"]} for order in pf["pending"])
        pf["cancelled_pending"] = pf["cancelled_pending"][-100:]
        pf["pending"] = []
    if pf.get("last_date") == trade_date:
        return pf, _summarize(pf, index_close)
    now = pd.Timestamp(published_at or datetime.now(C.TZ))
    if now.tzinfo is None:
        now = now.tz_localize(C.TZ)
    half_cost = C.TOTAL_COST_PCT / 200
    pending = []
    for order in pf["pending"]:
        if not order.get("entry_approved"):
            continue
        h = histories.get(order["code"])
        if h is None:
            pending.append(order)
            continue
        pub = pd.Timestamp(order["published_at"])
        bars = h[(h.index > pd.Timestamp(order["signal_date"]))
                 & (h.index <= pd.Timestamp(trade_date))]
        bars = bars[[pd.Timestamp(str(d)[:10] + " 09:00", tz=C.TZ) > pub for d in bars.index]]
        if bars.empty:
            pending.append(order)
            continue
        day, bar = next(bars.iterrows())
        px = float(bar["open"])
        if px <= 0:
            continue
        shares = int(min(order["budget"], pf["cash"]) / (px * (1 + half_cost)))
        if shares <= 0:
            continue
        pf["cash"] -= shares * px * (1 + half_cost)
        pf["positions"].append({"code": order["code"], "name": order["name"],
                                "shares": shares, "entry": px, "entry_date": str(day)[:10],
                                "held": 0, "last_processed": None,
                                "target1": px * (1 + (C.EXIT_MIN_PROFIT + C.TOTAL_COST_PCT) / 100),
                                "stop": (px * (1 + (C.TOTAL_COST_PCT - C.STOP_LOSS_NET_PCT) / 100)
                                         if C.STOP_LOSS_NET_PCT is not None else None)})
    pf["pending"] = pending
    still = []
    for pos in pf["positions"]:
        h = histories.get(pos["code"])
        if h is None:
            still.append(pos)
            continue
        bars = h[(h.index >= pd.Timestamp(pos["entry_date"]))
                 & (h.index <= pd.Timestamp(trade_date))]
        if pos.get("last_processed"):
            bars = bars[bars.index > pd.Timestamp(pos["last_processed"])]
        closed = False
        for date, bar in bars.iterrows():
            pos["held"] += 1
            pos["last_processed"] = str(date)[:10]
            px = float(bar["close"])
            net = strategy.net_return(pos["entry"], px)
            reason = strategy.exit_reason(net, pos["held"])
            if not reason:
                continue
            pf["cash"] += pos["shares"] * (px - pos["entry"] * half_cost)
            pf["trades"].append({"code": pos["code"], "name": pos["name"],
                                  "entry_date": pos["entry_date"], "exit_date": str(date)[:10],
                                  "entry": pos["entry"], "exit": px, "net_pct": round(net, 4),
                                  "pnl": pos["shares"] * pos["entry"] * net / 100,
                                  "success": strategy.target_met(net),
                                  "reason": reason, "held": pos["held"]})
            closed = True
            break
        if not closed:
            still.append(pos)
    pf["positions"] = still
    mv = 0.0
    for pos in still:
        h = histories.get(pos["code"])
        valid = h[h.index <= pd.Timestamp(trade_date)] if h is not None else None
        px = float(valid["close"].iloc[-1]) if valid is not None and len(valid) else pos["entry"]
        mv += pos["shares"] * (px - pos["entry"] * half_cost)
    if not pf.get("forward_start"):
        pf["forward_start"] = trade_date
    if pf.get("start_index") is None and index_close:
        pf["start_index"] = float(index_close)
    pf["equity"].append({"date": trade_date, "equity": round(pf["cash"] + mv, 4),
                         "index": index_close})
    held = {p["code"] for p in still + pf["pending"]}
    slots = C.PAPER_MAX_POSITIONS - len(held)
    if slots > 0 and not policy["blocked"]:
        budget = pf["cash"] / slots
        for row in rows:
            if slots <= 0:
                break
            if not row.get("trade_eligible") or row["code"] in held:
                continue
            pf["pending"].append({"code": row["code"], "name": row.get("name", ""),
                                   "budget": budget, "signal_date": trade_date,
                                   "entry_approved": True,
                                   "published_at": now.isoformat()})
            slots -= 1
    pf["last_date"] = trade_date
    return pf, _summarize(pf, index_close)
