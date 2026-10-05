"""Cash- and risk-constrained quantities, used by historical and forward books."""
from __future__ import annotations

import config as C
import math
import execution


def position_size(equity, cash, price, code="", stop_pct=None, committed_risk=0., sector_value=0.):
    stop = C.STOP_LOSS_NET_PCT if stop_pct is None else stop_pct
    if equity <= 0 or cash <= 0 or price <= 0 or not stop or stop <= 0:
        return {"shares": 0, "cash_required": 0., "planned_risk": 0., "reason": "缺少有效資金或停損設定"}
    risk_room = min(equity * C.RISK_PER_TRADE_PCT / 100,
                    max(0., equity * C.MAX_TOTAL_RISK_PCT / 100 - committed_risk))
    money_room = min(cash, equity * C.MAX_POSITION_PCT / 100,
                     max(0., equity * C.MAX_SECTOR_POSITION_PCT / 100 - sector_value))
    # Minimum fees are included by recomputing cash requirements for the actual
    # integer quantity. Net stop + gap buffer is a planning budget, not a cap.
    risk_pct = stop + C.RISK_GAP_BUFFER_PCT
    lo, hi = 0, max(0, int(money_room / price))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        paid = execution.buy_cash(price, mid)
        planned = paid * risk_pct / 100
        if paid <= money_room + 1e-9 and planned <= risk_room + 1e-9:
            lo = mid
        else:
            hi = mid - 1
    paid = execution.buy_cash(price, lo) if lo else 0.
    return {"shares": lo, "cash_required": round(paid, 4),
            "planned_risk": round(paid * risk_pct / 100, 4), "risk_pct": risk_pct,
            "reason": "符合計畫風險額度" if lo else "資金、產業或總風險額度不足"}


def attach_plans(rows):
    for r in rows:
        px = float(r.get("close") or 0)
        if px <= 0:
            continue
        code = r["code"]
        shares = max(1, int(C.REFERENCE_NOTIONAL_TWD / px))
        r["entry_plan"] = execution.make_entry_plan(px, r.get("ma20"))
        r["trade_plan"] = {
            "entry_ceiling": (math.floor(r["entry_plan"]["max_open_price"] * 100) / 100
                              if r["entry_plan"].get("available") else None),
            "reference_price": px,
            "target_trigger": round(execution.price_for_net(px, C.EXIT_MIN_PROFIT, shares, code), 2),
            "stop_trigger": round(execution.price_for_net(px, -C.STOP_LOSS_NET_PCT, shares, code), 2) if C.STOP_LOSS_NET_PCT else None,
            "stop_net_pct": C.STOP_LOSS_NET_PCT,
            "risk_per_trade_pct": C.RISK_PER_TRADE_PCT,
            "gap_buffer_pct": C.RISK_GAP_BUFFER_PCT,
            "max_days": C.EXIT_MAX_DAYS,
            "status": "符合模擬候選條件" if r.get("trade_eligible") else "研究觀察 · 尚不可配置",
            "note": "價位以最新收盤及參考金額估算；實際進場後重算。收盤觸發、次開盤退出，跳空可能低於獲利目標或超過停損。"}
