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
import strategy

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
    import strategy
    days = hist.get("days", []) if isinstance(hist, dict) and strategy.compatible(hist) else []
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
        delta = (old - r.get("final_rank", r.get("rank", 999))) if isinstance(old, int) else None

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
            "symbol": code, "date": trade_date, "rank": r.get("final_rank", r.get("rank", 999)),
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
    _save(C.SIGNALS_JSON, {"strategy": strategy.contract(), "days": days})

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
            "trades": [], "equity": [], "start_index": None, "last_date": None,
            # 前瞻測試：從這一天開始，每一筆都是當下決定、隔日成交，
            # 沒有任何一筆是事後用歷史資料補算出來的
            "forward_start": None, "mode": "forward_test"}


def update_portfolio(rows: list[dict], trade_date: str, index_close: float | None,
                     hist_map: dict | None = None) -> dict:
    from paper_target import update
    previous = _load(C.PORTFOLIO_JSON, {})
    if previous and not strategy.compatible(previous):
        # Preserve prior target/version results instead of silently overwriting them.
        import hashlib
        old_id = hashlib.sha256(json.dumps(previous.get("strategy"), sort_keys=True).encode()).hexdigest()[:12]
        _save(f"data/portfolio_archive_{old_id}.json", previous)
    pf, summary = update(previous, rows, trade_date, index_close, hist_map or {})
    _save(C.PORTFOLIO_JSON, pf)
    return summary


def _summarize(pf: dict, index_close: float | None) -> dict:
    """把組合狀態濃縮成頁面要顯示的數字。全部除零防呆。"""
    init = float(C.PAPER_INITIAL_CASH)
    eq = pf.get("equity", [])
    equity = _num(eq[-1]["equity"], init) if eq else init

    trades = [t for t in pf.get("trades", []) if isinstance(t, dict)]
    nets = [_num(t.get("net_pct")) for t in trades]
    wins = [x for x in nets if x > 0]
    losses = [x for x in nets if x <= 0]
    pnls = [_num(t.get("pnl"), _num(t.get("net_pct"))) for t in trades]
    gross_w, gross_l = sum(x for x in pnls if x > 0), -sum(x for x in pnls if x <= 0)

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
        "strategy": pf.get("strategy"),
        "success_rate": round(sum(bool(t.get("success")) for t in trades) / len(trades) * 100, 1) if trades else None,
        "mode": pf.get("mode", "forward_test"),
        "forward_start": pf.get("forward_start"),
        "cost_pct": C.TOTAL_COST_PCT,
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
