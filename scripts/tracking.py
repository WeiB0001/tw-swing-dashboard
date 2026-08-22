# -*- coding: utf-8 -*-
"""
tracking.py — 訊號生命週期 + 模擬投資組合（Paper Portfolio）

兩件事都只用 data/ 底下的 JSON 存檔，沒有資料庫、沒有外部服務。
所有讀檔都容錯：檔案不存在、格式壞掉、欄位缺失都會安靜地重新開始，
不會讓 build 或 GitHub Actions 掛掉。

**絕不使用未來資料**：
  - 訊號歷史只比對「今天以前」已經存檔的排名
  - 模擬組合在 t 日收盤後掛單，t+1 日用當天的開盤價成交
    （成交價來自下一次執行時證交所回報的開盤價，不是預測值）
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import config as C

log = logging.getLogger("tracking")
ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 讀寫小工具
# ---------------------------------------------------------------------------
def _load(rel: str, default):
    try:
        p = ROOT / rel
        if not p.exists():
            return default
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("%s 讀取失敗，改用預設值：%s", rel, e)
        return default


def _save(rel: str, data) -> None:
    try:
        p = ROOT / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception as e:
        log.warning("%s 寫入失敗：%s", rel, e)


def _num(x, default=0.0) -> float:
    try:
        v = float(x)
        return v if v == v and abs(v) != float("inf") else default
    except Exception:
        return default


# ---------------------------------------------------------------------------
# 1) 訊號生命週期
# ---------------------------------------------------------------------------
def update_signals(rows: list[dict], trade_date: str, regime: str) -> dict:
    """
    把今天的排名寫進 data/signals.json，並回傳每一檔的狀態標記。
    回傳 {code: {"badge": "🆕", "streak": 3, "rank_delta": +5, ...}}
    """
    hist = _load(C.SIGNALS_JSON, {"days": []})
    days = hist.get("days", []) if isinstance(hist, dict) else []
    days = [d for d in days if isinstance(d, dict) and d.get("date") != trade_date]

    prev_day = days[-1] if days else None
    prev_rank = {}
    if prev_day:
        for e in prev_day.get("entries", []):
            prev_rank[e.get("symbol")] = e.get("rank")

    # 連續入榜天數：往回數，只要中斷就停
    def streak_of(code: str) -> int:
        n = 0
        for d in reversed(days):
            if any(e.get("symbol") == code for e in d.get("entries", [])):
                n += 1
            else:
                break
        return n

    marks, entries = {}, []
    top_n = C.PAPER_MAX_POSITIONS * 6      # 只追蹤前段名次，檔案才不會無限膨脹
    for r in rows[:top_n]:
        code = r["code"]
        old = prev_rank.get(code)
        streak = streak_of(code) + 1
        delta = (old - r["rank"]) if isinstance(old, int) else None

        if old is None:
            badge, label = "🆕", "今日新訊號"
        elif streak >= C.STREAK_HOT:
            badge, label = "🔥", f"連續入榜 {streak} 日"
        elif delta is not None and delta >= C.RANK_MOVE_MIN:
            badge, label = "↑", f"名次上升 {delta} 名"
        elif delta is not None and delta <= -C.RANK_MOVE_MIN:
            badge, label = "↓", f"名次下降 {-delta} 名"
        else:
            badge, label = "", ""

        marks[code] = {"badge": badge, "label": label, "streak": streak,
                       "rank_delta": delta, "prev_rank": old}
        entries.append({
            "symbol": code, "date": trade_date, "rank": r["rank"],
            "ref_price": _num(r.get("close")), "score": _num(r.get("score")),
            "pattern": r.get("kind"), "regime": regime,
        })

    # 昨天在榜、今天掉出去的 → 訊號失效
    dropped = []
    if prev_day:
        now = {r["code"] for r in rows[:top_n]}
        for e in prev_day.get("entries", []):
            if e.get("symbol") not in now:
                dropped.append({"symbol": e.get("symbol"), "prev_rank": e.get("rank"),
                                "pattern": e.get("pattern")})

    days.append({"date": trade_date, "regime": regime, "entries": entries})
    days = days[-C.SIGNALS_KEEP_DAYS:]
    _save(C.SIGNALS_JSON, {"days": days})

    return {
        "marks": marks,
        "dropped": dropped[:8],
        "new_count": sum(1 for m in marks.values() if m["badge"] == "🆕"),
        "hot_count": sum(1 for m in marks.values() if m["badge"] == "🔥"),
        "history_days": len(days),
    }


# ---------------------------------------------------------------------------
# 2) 模擬投資組合
# ---------------------------------------------------------------------------
def _blank_portfolio() -> dict:
    return {"cash": float(C.PAPER_INITIAL_CASH), "positions": [], "pending": [],
            "trades": [], "equity": [], "start_index": None, "last_date": None}


def update_portfolio(rows: list[dict], trade_date: str, index_close: float | None) -> dict:
    """
    每天收盤後跑一次：
      1. 先用今天的開盤價成交昨天掛的單（t+1 open，不是預測）
      2. 檢查持股是否觸及停損／目標／持有上限
      3. 用今天收盤價結算淨值
      4. 依今天的排名，替明天掛新單

    rows 需要有 day_open / day_high / day_low（build_row 會帶）。
    """
    pf = _load(C.PORTFOLIO_JSON, None)
    if not isinstance(pf, dict) or "cash" not in pf:
        pf = _blank_portfolio()
    for k, v in _blank_portfolio().items():
        pf.setdefault(k, v)

    if pf.get("last_date") == trade_date:      # 同一天重跑（例如 18:00 那次）不重複交易
        return _summarize(pf, index_close)

    by_code = {r["code"]: r for r in rows}
    cost = C.PAPER_COST_SIDE_PCT / 100

    # --- 1) 成交昨天掛的單：用今天的開盤價 ---
    filled = []
    for order in pf.get("pending", []):
        r = by_code.get(order.get("code"))
        if not r:
            continue                            # 今天沒進榜／沒資料 → 這張單作廢
        px = _num(r.get("day_open")) or _num(r.get("close"))
        if px <= 0:
            continue
        budget = _num(order.get("budget"))
        shares = int(budget // px)              # 零股買法，1 股為單位
        if shares < 1:
            continue
        cash_out = shares * px * (1 + cost)
        if cash_out > pf["cash"]:
            shares = int(pf["cash"] / (px * (1 + cost)))
            cash_out = shares * px * (1 + cost)
        if shares < 1:
            continue
        pf["cash"] -= cash_out
        pf["positions"].append({
            "code": r["code"], "name": r.get("name", ""), "shares": shares,
            "entry": round(px, 2), "entry_date": trade_date, "held": 0,
            "stop": _num(order.get("stop")), "target1": _num(order.get("target1")),
            "pattern": order.get("pattern"),
        })
        filled.append(r["code"])
    pf["pending"] = []

    # --- 2) 出場檢查（用今天的高低價，不用未來資料）---
    still = []
    for pos in pf.get("positions", []):
        r = by_code.get(pos["code"])
        if not r:
            still.append(pos)                   # 今天沒資料就先留著，明天再看
            continue
        if pos["code"] in filled:
            still.append(pos)                   # 今天才成交，不當天出場
            continue

        pos["held"] = int(pos.get("held", 0)) + 1
        hi = _num(r.get("day_high")) or _num(r.get("close"))
        lo = _num(r.get("day_low")) or _num(r.get("close"))
        cl = _num(r.get("close"))
        stop, t1 = _num(pos.get("stop")), _num(pos.get("target1"))

        exit_px, reason = None, None
        if stop > 0 and lo <= stop:
            exit_px, reason = stop, "停損"      # 保守：同日觸及停損視為以停損價成交
        elif t1 > 0 and hi >= t1:
            exit_px, reason = t1, "達標"
        elif pos["held"] >= C.PAPER_MAX_HOLD_DAYS:
            exit_px, reason = cl, "持有到期"

        if exit_px and exit_px > 0:
            proceeds = pos["shares"] * exit_px * (1 - cost)
            pf["cash"] += proceeds
            gross = (exit_px / pos["entry"] - 1) * 100 if pos["entry"] else 0
            net = gross - C.TRADE_COST_PCT
            pf["trades"].append({
                "code": pos["code"], "name": pos.get("name", ""),
                "entry_date": pos.get("entry_date"), "exit_date": trade_date,
                "entry": pos["entry"], "exit": round(exit_px, 2),
                "net_pct": round(net, 2), "reason": reason,
                "held": pos["held"], "pattern": pos.get("pattern"),
            })
        else:
            still.append(pos)
    pf["positions"] = still

    # --- 3) 用今天收盤價結算淨值 ---
    mv = 0.0
    for pos in pf["positions"]:
        r = by_code.get(pos["code"])
        px = _num(r.get("close")) if r else pos["entry"]
        mv += pos["shares"] * (px or pos["entry"])
    equity = pf["cash"] + mv
    if pf.get("start_index") is None and index_close:
        pf["start_index"] = _num(index_close)
    pf["equity"].append({"date": trade_date, "equity": round(equity, 2),
                         "index": _num(index_close) or None})
    pf["equity"] = pf["equity"][-250:]

    # --- 4) 替明天掛新單：今天排名前 N、還沒持有、且計畫不是「禁止追價」---
    held = {p["code"] for p in pf["positions"]}
    slots = C.PAPER_MAX_POSITIONS - len(pf["positions"])
    if slots > 0 and pf["cash"] > 1000:
        per = pf["cash"] / slots                # 等權配置
        for r in rows:
            if slots <= 0:
                break
            if r["code"] in held:
                continue
            pl = r.get("plan") or {}
            if pl.get("status") == "禁止追價":
                continue
            pf["pending"].append({
                "code": r["code"], "budget": round(per, 2),
                "stop": _num(pl.get("stop")), "target1": _num(pl.get("target1")),
                "pattern": r.get("kind"),
            })
            slots -= 1

    pf["last_date"] = trade_date
    _save(C.PORTFOLIO_JSON, pf)
    return _summarize(pf, index_close)


def _summarize(pf: dict, index_close: float | None) -> dict:
    """把組合狀態濃縮成頁面要顯示的數字。全部除零防呆。"""
    init = float(C.PAPER_INITIAL_CASH)
    eq = pf.get("equity", [])
    equity = _num(eq[-1]["equity"], init) if eq else init

    trades = [t for t in pf.get("trades", []) if isinstance(t, dict)]
    nets = [_num(t.get("net_pct")) for t in trades]
    wins = [x for x in nets if x > 0]
    losses = [x for x in nets if x <= 0]
    gross_w, gross_l = sum(wins), -sum(losses)

    # 最大回撤：用淨值曲線的高點回落
    peak, mdd = init, 0.0
    for p in eq:
        v = _num(p.get("equity"), init)
        peak = max(peak, v)
        if peak > 0:
            mdd = min(mdd, (v / peak - 1) * 100)

    bench = None
    start_idx = _num(pf.get("start_index"))
    if start_idx > 0 and index_close:
        bench = round((_num(index_close) / start_idx - 1) * 100, 2)

    positions = []
    for p in pf.get("positions", []):
        positions.append({"code": p.get("code"), "name": p.get("name", ""),
                          "entry": p.get("entry"), "held": p.get("held", 0),
                          "stop": p.get("stop"), "target1": p.get("target1")})

    return {
        "initial": init,
        "equity": round(equity, 2),
        "total_return": round((equity / init - 1) * 100, 2) if init else 0.0,
        "benchmark_return": bench,
        "trades": len(trades),
        "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else None,
        "profit_factor": round(gross_w / gross_l, 2) if gross_l > 0 else None,
        "avg_net": round(sum(nets) / len(nets), 2) if nets else None,
        "mdd": round(mdd, 2),
        "positions": positions,
        "pending": len(pf.get("pending", [])),
        "recent": trades[-5:][::-1],
        "days": len(eq),
    }
