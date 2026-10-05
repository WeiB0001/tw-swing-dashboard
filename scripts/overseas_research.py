"""Reconstructed as-of US context and a separate, never auto-promoted rank study.

Regular US closes are aligned in America/New_York (including DST). Taiwan
signals use 13:35 Asia/Taipei on their quote date. Vendor revisions are not a
historical vintage archive; first live publications are frozen by research.py.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import math
import pandas as pd
import config as C

ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = {"spx": "^GSPC", "nasdaq": "^IXIC", "sox": "^SOX", "vix": "^VIX"}


def load_history(refresh=False):
    path = ROOT / "data/overseas_history.json"
    try:
        saved = json.loads(path.read_text())
    except (OSError, ValueError):
        saved = {"series": {}}
    if refresh:
        import yfinance as yf
        changed = False
        for key, ticker in SYMBOLS.items():
            try:
                df = yf.download(ticker, period="3y", auto_adjust=False, progress=False,
                                 threads=False, timeout=15)
                if df is None or df.empty:
                    continue
                close = df["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                values = {str(day)[:10]: float(value) for day, value in close.items()
                          if math.isfinite(float(value)) and float(value) > 0}
                if values:
                    saved.setdefault("series", {})[key] = values
                    changed = True
            except Exception:
                continue  # Retain dated cache; stale/missing is never neutral.
        if changed:
            saved.update(fetched_at=datetime.now(timezone.utc).isoformat(), source="Yahoo Finance via yfinance")
            path.write_text(json.dumps(saved, ensure_ascii=False, indent=2))
    out = {}
    for key in SYMBOLS:
        values = saved.get("series", {}).get(key, {})
        if values:
            df = pd.DataFrame({"close": pd.Series(values, dtype=float)})
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df["ret1"] = df.close.pct_change(fill_method=None) * 100
            df["ret5"] = df.close.pct_change(5, fill_method=None) * 100
            out[key] = df
    return out


def context_for_day(histories, signal_date):
    cutoff = pd.Timestamp(str(signal_date)[:10] + " 13:35", tz="Asia/Taipei")
    values, dates, unavailable = {}, {}, []
    for key in SYMBOLS:
        df = histories.get(key)
        if df is None or df.empty:
            unavailable.append(key)
            continue
        index = pd.DatetimeIndex(df.index)
        if index.tz is not None:
            index = index.tz_localize(None)
        closes = (index.normalize() + pd.Timedelta(hours=16)).tz_localize("America/New_York")
        positions = [i for i, when in enumerate(closes) if when <= cutoff]
        if not positions:
            unavailable.append(key)
            continue
        pos = positions[-1]
        date = index[pos]
        dates[key] = date.strftime("%Y-%m-%d")
        stale = (cutoff.tz_localize(None).normalize() - date.normalize()).days > C.OVERSEAS_MAX_AGE_DAYS
        cols = ["close"] if key == "vix" else ["ret1", "ret5"]
        row = df.iloc[pos]
        if stale or any(not math.isfinite(float(row.get(col, float("nan")))) for col in cols):
            unavailable.append(key)
            continue
        values[key] = {col: float(row[col]) for col in cols}
    return {"available": not unavailable, "cutoff": cutoff.isoformat(), "dates": dates,
            "values": values, "missing_or_stale": unavailable,
            "basis": "訊號日台灣 13:35 前已完成的美股正常盤；缺漏或過期不填成中性"}


def attach_context(rows, context):
    for row in rows:
        category = "tech" if row.get("asset_type") == "tech" else "broad"
        state = "unknown"
        if context.get("available"):
            values = context["values"]
            returns = ([values[k]["ret5"] for k in ("nasdaq", "sox")] if category == "tech"
                       else [values["spx"]["ret5"]])
            state = ("headwind" if values["vix"]["close"] >= C.OVERSEAS_VIX_RISK_LEVEL or all(x < 0 for x in returns)
                     else "tailwind" if all(x >= 0 for x in returns) else "mixed")
        row.update(overseas_context=context, overseas_group=category, overseas_state=state,
                   overseas_label={"unknown": "海外資料不足", "headwind": "海外逆風",
                                   "tailwind": "海外順風", "mixed": "海外分歧"}[state])


def calibration(signals):
    import backtest
    out = []
    for category in ("tech", "broad"):
        for state in ("tailwind", "headwind", "mixed"):
            for lo, hi in C.BACKTEST_SCORE_BUCKETS:
                sub = [s for s in signals if s.get("replay", {}).get("overseas_group") == category
                       and s["replay"].get("overseas_state") == state and lo <= s["score"] < hi]
                if sub:
                    out.append({"overseas_group": category, "overseas_state": state,
                                "lo": lo, "hi": hi, **backtest._pack(sub)})
    return out


def shadow_rank(rows, tables):
    import build
    import ranking
    eligible_context = [deepcopy(r) for r in rows if r.get("overseas_context", {}).get("available")]
    ranking.attach(eligible_context, tables, use_overseas=True)
    build.add_momentum(eligible_context)
    build.add_final_score(eligible_context)
    return ranking.sort(eligible_context)


def attach_shadow(rows, tables):
    ranked = shadow_rank(rows, tables)
    lookup = {r["code"]: r for r in ranked if r.get("hist_risk_reward") is not None}
    for row in rows:
        alternative = lookup.get(row["code"], {})
        row.update(overseas_rank=alternative.get("final_rank"),
                   overseas_hist_score=alternative.get("hist_risk_reward"),
                   overseas_hist_source=alternative.get("hist_source"))


def compare_day(study, ranked, tables, lookup, fold):
    """Same dates/pool/top-N. Conditional bucket fallback is counted explicitly."""
    study.setdefault("total_dates", 0)
    study["total_dates"] += 1
    pool = [r for r in ranked if r.get("hist_risk_reward") is not None
            and r.get("overseas_context", {}).get("available")]
    if not pool:
        return
    alternative = shadow_rank(pool, tables)[:C.WF_TOP_N]
    baseline = pool[:C.WF_TOP_N]
    if len(alternative) != len(baseline):
        return
    study.setdefault("matched_dates", 0)
    study["matched_dates"] += 1
    study.setdefault("fallback", 0)
    study["fallback"] += sum(r.get("hist_source") != "overseas" for r in alternative)
    for key, rows in (("baseline", baseline), ("overseas", alternative)):
        study.setdefault(key, []).extend({**lookup[r["code"]], "study_fold": fold} for r in rows)


def summarize(study):
    import backtest
    import risk_stats
    totals = study.get("total_dates", 0)
    matched = study.get("matched_dates", 0)
    variants = []
    for key, label in (("baseline", "原綜合排序"), ("overseas", "海外分組排序（研究）")):
        signals = study.get(key, [])
        variants.append({"id": key, "label": label, **backtest._pack(signals),
                         "selected": len(signals),
                         "unfilled": sum(s["exit"].get("entered") is False for s in signals),
                         "unresolved": sum(not s["exit"].get("closed") and s["exit"].get("entered") is not False for s in signals)})
    # Date-paired results include abstentions as cash (zero) only for the explicit
    # per-signal policy comparison; unresolved exits invalidate that day's pair.
    grouped = {}
    for key in ("baseline", "overseas"):
        for s in study.get(key, []):
            grouped.setdefault(s["date"], {}).setdefault(key, []).append(s)
    pairs = []
    for date, groups in grouped.items():
        if len(groups) != 2 or any(not s["exit"].get("closed") and s["exit"].get("entered") is not False
                                   for group in groups.values() for s in group):
            continue
        average = {key: sum(s["exit"].get("net", 0.) for s in group) / len(group) for key, group in groups.items()}
        pairs.append({"date": date, "exit": {"closed": True, "net": average["overseas"]-average["baseline"]}})
    uncertainty = risk_stats.expected_return_lower(pairs)
    return {"available": bool(matched), "active_in_main_rank": False, "auto_selected": False,
            "total_dates": totals, "matched_dates": matched,
            "coverage_pct": round(100 * matched / totals, 1) if totals else 0.,
            "fallback_picks": study.get("fallback", 0), "variants": variants,
            "paired_dates": len(pairs), "paired_ev_lift": round(sum(s["exit"]["net"] for s in pairs)/len(pairs),4) if pairs else None,
            "paired_ev_lift_lower": uncertainty["ev_lower"],
            "status": "獨立研究中，尚未納入主排名；需新資料持續驗證",
            "basis": "逐段只用已成熟訓練樣本，對同日同一股票池取前 3 名；條件樣本不足回退原統計。未成交與未結算另列。",
            "note": "歷史美股日線按已收盤時間重建，仍有供應商修訂與股票池選樣限制。配對差以每次訊號平均計算，跳過視為現金 0%，未結算日期整組排除；不是帳戶報酬，也不改列成交勝率。"}
