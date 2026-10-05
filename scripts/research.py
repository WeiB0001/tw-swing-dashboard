"""Frozen publication records, shadow books and matched exit experiments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pandas as pd
import config as C
import execution
import strategy
import paper_target

ROOT = Path(__file__).resolve().parents[1]


def strategy_id():
    return hashlib.sha256(json.dumps(strategy.contract(), sort_keys=True).encode()).hexdigest()[:12]


def freeze(rows, day, published_at):
    folder = ROOT / "data/research_snapshots"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{day}_{strategy_id()}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    keys = ("code", "name", "group", "close", "quote_date", "final_rank", "hist_risk_reward",
            "hist_expectancy", "hist_samples", "hist_signal_dates", "research_eligible", "trade_eligible",
            "ma20", "entry_plan", "overseas_context", "overseas_rank", "overseas_hist_score", "overseas_hist_source")
    record = {"strategy": strategy.contract(), "data_date": day, "published_at": published_at,
              "rows": [{k:r.get(k) for k in keys} for r in rows]}
    # The first publication for this date/version is immutable, including reruns.
    with path.open("x", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=2)
    return record


def update_forward(rows, day, published_at, histories, index_close=None):
    import tracking
    record = freeze(rows, day, published_at)
    state = tracking._load(C.RESEARCH_FORWARD_JSON, {})
    if not strategy.compatible(state):
        if state:
            old = hashlib.sha256(json.dumps(state.get("strategy"), sort_keys=True).encode()).hexdigest()[:12]
            tracking._save(f"data/research_archive_{old}.json", state)
        state = {"strategy": strategy.contract(), "books": {}}
    summary = {"strategy": strategy.contract(), "published_at": record["published_at"],
               "snapshot": f"data/research_snapshots/{day}_{strategy_id()}.json", "books": {}}
    candidates = [{**r, "research_eligible": True} for r in record["rows"]
                  if r.get("hist_risk_reward") is not None and r.get("quote_date") == day]
    streams = {"composite": candidates,
               "ev_only": sorted(candidates, key=lambda r: -(r.get("hist_expectancy") or 0)),
               "overseas": sorted([r for r in candidates if r.get("overseas_rank") is not None],
                                  key=lambda r: r["overseas_rank"])}
    for name, ordered in streams.items():
        book, stats = paper_target.update(state["books"].get(name, {}), ordered[:C.WF_TOP_N], day,
            index_close, histories, record["published_at"], research=True)
        state["books"][name] = book
        summary["books"][name] = stats
    tracking._save(C.RESEARCH_FORWARD_JSON, state)
    return summary


def historical_book(signals, histories):
    import tracking
    if not signals:
        return {"available": False, "reason": "尚無樣本外研究候選"}
    by_day = {}
    for s in signals:
        by_day.setdefault(s["date"], []).append(s)
    first, last = min(by_day), max(s["exit"]["label_end"] for s in signals)
    dates = sorted({str(d)[:10] for h in histories.values() for d in h.index if first <= str(d)[:10] <= last})
    book = {}
    for day in dates:
        rows = [{**s["replay"], "research_eligible": True} for s in by_day.get(day, [])]
        book, _ = paper_target.update(book, rows, day, None, histories, day+"T16:00:00+08:00", research=True)
    stats = tracking._summarize(book, None)
    stats.update(available=True, mode="historical_research", cash=round(book["cash"],2),
                 skipped=len(book.get("skipped", [])), equity_curve=book["equity"],
                 planned_risk=sum(p.get("planned_risk",0) for p in book["positions"]),
                 note="樣本外每日候選、有限本金及產業/部位上限；仍持有部位以最後可得價格估值，未強制假造平倉。")
    return stats


def exit_study(signals, histories):
    import backtest
    variants = [("none", "無價格停損", None), ("fixed2", "淨損 2% 觸發", 2.),
                ("fixed3", "淨損 3% 觸發（目前研究設定）", 3.), ("atr", "1.5 ATR（2～6%）", "atr")]
    output = []
    for key, label, stop in variants:
        outcomes, pending, unfilled, periods = [], 0, 0, {}
        for s in signals:
            df = histories[s["code"]]
            pos = df.index.get_indexer([pd.Timestamp(s["date"])])[0] + 1
            actual_stop = stop
            if stop == "atr":
                atr = float(s["replay"].get("atr_pct") or 0)
                actual_stop = max(2., min(6., 1.5 * atr))
            ex = execution.simulate(df, pos, s["code"], actual_stop)
            if ex and ex.get("closed"):
                row = {"date": s["date"], "exit": ex}
                outcomes.append(row)
                periods.setdefault(s["date"][:7], []).append(row)
            elif ex and ex.get("entered") is False:
                unfilled += 1
            else:
                pending += 1
        output.append({"id": key, "label": label, "current": key == "fixed3",
                       "unresolved": pending, "unfilled": unfilled, **backtest._pack(outcomes),
                       "months": [{"month": m, **backtest._pack(v)} for m,v in sorted(periods.items())]})
    return {"variants": output, "auto_selected": False, "signals": len(signals),
            "basis": "固定同一組有歷史統計的樣本外前 3 名與進場日（含未過交易資格者），僅改出場；不是各方案重新訓練後的獨立驗證。",
            "note": "所有方案一併列出，不依本次歷史最佳值自動切換；下一次變更須另用新資料驗證。"}


def entry_study(signals, histories):
    import backtest
    variants = []
    for guard, label in ((False, "無進場上限"), (True, "防追高上限（目前設定）")):
        outcomes, unfilled, skipped, unresolved = [], 0, 0, 0
        for s in signals:
            df = histories[s["code"]]
            pos = df.index.get_indexer([pd.Timestamp(s["date"])])[0] + 1
            ex = execution.simulate(df, pos, s["code"], guard=guard)
            if ex and ex.get("closed"):
                outcomes.append({"date": s["date"], "exit": ex})
            elif ex and ex.get("entered") is False:
                unfilled += 1
                skipped += bool(ex.get("entry_skipped"))
            else:
                unresolved += 1
        variants.append({"label": label, "guard": guard, "unfilled": unfilled, "skipped": skipped,
                         "unresolved": unresolved, **backtest._pack(outcomes)})
    return {"signals": len(signals), "variants": variants, "auto_selected": False,
            "basis": "固定同一組樣本外研究前 3 名，只切換進場上限；統計限已成交結算，跳過另列。不是兩套重新訓練策略的獨立勝負驗證。"}
