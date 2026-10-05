"""Chronological cash ledger. Research books never authorize real allocation."""
from datetime import datetime
import math
import pandas as pd
import config as C
import strategy
import execution
import risk_budget


def _mark(pf, histories, day):
    value = pf["cash"]
    for p in pf["positions"]:
        h = histories.get(p["code"])
        valid = h[(h.index <= pd.Timestamp(day)) & h["close"].notna() & (h["close"] > 0)] if h is not None else None
        px = float(valid["close"].iloc[-1]) if valid is not None and len(valid) else p["entry"]
        value += execution.sell_cash(px, p["shares"], p["code"])
    return value


def update(pf, rows, trade_date, index_close, histories, published_at=None, capital_policy=None, research=False):
    from tracking import _blank_portfolio, _summarize
    if not strategy.compatible(pf):
        cancelled = [{**o, "cancel_reason": "策略版本更換，舊待成交單撤銷"} for o in pf.get("pending", [])]
        pf = _blank_portfolio()
        pf.update(strategy=strategy.contract(), cancelled_pending=cancelled)
    policy = strategy.capital_policy()
    if not policy["blocked"]:
        if capital_policy and strategy.compatible(capital_policy) and capital_policy.get("research_validated"):
            policy = dict(capital_policy)
        else:
            policy.update(blocked=True, reason="風險報酬綜合排名尚未通過樣本外驗證，暫不配置資金")
    pf["capital_policy"] = policy
    pf["mode"] = "research_forward" if research else "forward_test"
    pf.setdefault("skipped", [])
    pf.setdefault("cancelled_pending", [])
    # A forward book starts when it was created, not at the beginning of the
    # downloaded price history. Remove only pre-start cash marks from early v4
    # runs; never change a trade, pending order or publication record.
    if pf.get("forward_start"):
        pf["equity"] = [e for e in pf["equity"] if e["date"] >= pf["forward_start"]]
    if policy["blocked"] and not research and pf.get("pending"):
        pf["cancelled_pending"].extend({**o, "cancel_reason": policy["reason"]} for o in pf["pending"])
        pf["pending"] = []
    if pf.get("last_date") and pf["last_date"] >= trade_date:
        return pf, _summarize(pf, index_close)
    now = pd.Timestamp(published_at or datetime.now(C.TZ))
    if now.tzinfo is None:
        now = now.tz_localize(C.TZ)
    dates = {pd.Timestamp(trade_date)}
    start_date = pf.get("last_date")
    if not start_date and (pf["positions"] or pf["pending"]):
        first_known = min(p.get("entry_date") or p["signal_date"] for p in pf["positions"] + pf["pending"])
        start_date = str(pd.Timestamp(first_known) - pd.Timedelta(days=1))[:10]
    for h in histories.values():
        if h is not None and start_date:
            dates.update(d for d in h.index if str(d)[:10] <= trade_date and
                         str(d)[:10] > start_date)
    for date in sorted(dates):
        day = str(date)[:10]
        if pf.get("last_date") and day <= pf["last_date"]:
            continue
        remaining = []
        for p in pf["positions"]:
            p["held"] += 1
            if p["held"] >= C.EXIT_MAX_DAYS and not p.get("exit_pending"):
                p.update(exit_pending="持有到期", trigger_date=day)
            h = histories.get(p["code"])
            if h is None or date not in h.index or not math.isfinite(float(h.loc[date,"close"])):
                p["missing_quote_days"] = p.get("missing_quote_days",0) + 1
                remaining.append(p)
                continue
            bar = h.loc[date]
            before = h[(h.index < date) & h["close"].notna() & (h["close"] > 0)]
            prev = float(before["close"].iloc[-1]) if len(before) else p["entry"]
            if p.get("exit_pending") and execution.tradable(bar, "sell", prev):
                px = float(bar["open"])
                proceeds = execution.sell_cash(px, p["shares"], p["code"])
                paid = p.get("entry_cash", execution.buy_cash(p["entry"], p["shares"]))
                net = (proceeds / paid - 1) * 100
                pf["cash"] += proceeds
                pf["trades"].append({"code": p["code"], "name": p.get("name", ""),
                    "entry_date": p["entry_date"], "exit_date": day, "entry": p["entry"], "exit": px,
                    "shares": p["shares"], "entry_cash": paid, "exit_cash": proceeds,
                    "net_pct": round(net, 6), "pnl": round(proceeds-paid, 6),
                    "success": strategy.target_met(net), "reason": p["exit_pending"],
                    "held": p["held"], "delayed_days": p.get("delayed_days", 0),
                    "planned_risk": p.get("planned_risk"), "trigger_date": p.get("trigger_date")})
            else:
                if p.get("exit_pending"):
                    p["delayed_days"] = p.get("delayed_days", 0) + 1
                remaining.append(p)
        pf["positions"] = remaining
        pending = []
        # Quantities at this open use previous-close equity, never today's close.
        equity = _mark(pf, histories, date - pd.Timedelta(days=1))
        for order in pf["pending"]:
            if not order.get("entry_approved"):
                continue
            pub = pd.Timestamp(order["published_at"])
            if pub.tzinfo is None:
                pub = pub.tz_localize(C.TZ)
            if day <= order["signal_date"] or pd.Timestamp(day+" 09:00", tz=C.TZ) <= pub:
                pending.append(order)
                continue
            h = histories.get(order["code"])
            if h is None or date not in h.index:
                pf["skipped"].append({"code": order["code"], "date": day, "reason": "預定進場日缺少行情"})
                continue
            bar = h.loc[date]
            before = h[(h.index < date) & h["close"].notna() & (h["close"] > 0)]
            prev = float(before["close"].iloc[-1]) if len(before) else None
            if not execution.tradable(bar, "buy", prev):
                pf["skipped"].append({"code": order["code"], "date": day, "reason": "進場無法成交，不追價補買"})
                continue
            group = order.get("group") or "其他"
            sector = sum(p.get("entry_cash", 0) for p in pf["positions"] if p.get("group") == group)
            risk = sum(p.get("planned_risk", 0) for p in pf["positions"])
            px = float(bar["open"])
            size = risk_budget.position_size(equity, min(pf["cash"], order.get("budget", pf["cash"])), px,
                                             order["code"], committed_risk=risk, sector_value=sector)
            if not size["shares"] or len(pf["positions"]) >= C.PAPER_MAX_POSITIONS:
                pf["skipped"].append({"code": order["code"], "date": day, "reason": size["reason"]})
                continue
            shares = size["shares"]
            paid = execution.buy_cash(px, shares)
            pf["cash"] -= paid
            pf["positions"].append({"code": order["code"], "name": order.get("name", ""), "group": group,
                "shares": shares, "entry": px, "entry_cash": paid, "entry_date": day, "held": 1,
                "planned_risk": size["planned_risk"], "stop_pct": C.STOP_LOSS_NET_PCT,
                "target1": execution.price_for_net(px, C.EXIT_MIN_PROFIT, shares, order["code"]),
                "stop": execution.price_for_net(px, -C.STOP_LOSS_NET_PCT, shares, order["code"]) if C.STOP_LOSS_NET_PCT else None})
        pf["pending"] = pending
        for p in pf["positions"]:
            h = histories.get(p["code"])
            if h is None or date not in h.index or p.get("exit_pending") or not math.isfinite(float(h.loc[date,"close"])):
                continue
            reason = execution.close_decision(p["entry"], float(h.loc[date,"close"]), p["held"],
                p["shares"], p["code"], p.get("stop_pct", C.STOP_LOSS_NET_PCT))
            if reason:
                p.update(exit_pending=reason, trigger_date=day)
        pf["equity"].append({"date": day, "equity": round(_mark(pf, histories, date), 4), "index": index_close})
    if not pf.get("forward_start"):
        pf["forward_start"] = min([trade_date] + [e["date"] for e in pf["equity"]])
    if pf.get("start_index") is None and index_close:
        pf["start_index"] = float(index_close)
    held_codes = {p["code"] for p in pf["positions"] + pf["pending"]}
    slots = C.PAPER_MAX_POSITIONS - len(held_codes)
    if slots > 0 and (research or not policy["blocked"]):
        for row in rows:
            eligible = row.get("research_eligible") if research else row.get("trade_eligible")
            if not eligible or row["code"] in held_codes:
                continue
            pf["pending"].append({"code": row["code"], "name": row.get("name", ""),
                "group": row.get("group") or "其他", "signal_date": trade_date,
                "entry_approved": True, "published_at": now.isoformat()})
            held_codes.add(row["code"])
            slots -= 1
            if slots <= 0:
                break
    pf["last_date"] = trade_date
    return pf, _summarize(pf, index_close)
