# -*- coding: utf-8 -*-
"""
backtest.py — 與首頁共用排序程式的淨利目標回測。
訊號日收盤後選股，次日開盤進場，第 2～10 日收盤淨利達標才成功，
未達標第 10 日出場；小幅正報酬仍納入 EV，但不計入達標次數。
同股校準樣本間隔 10 日，逐段驗證剔除尚未成熟的訓練結果。
目前股票池回溯存在選樣限制，跨股與連續日樣本亦非互相獨立。

python scripts/backtest.py --days 250
python scripts/backtest.py --demo --no-save
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import config as C
import indicators
import scoring
import strategy
import ranking
import risk_stats

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("backtest")

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 資料
# ---------------------------------------------------------------------------
def load_universe_history(demo: bool) -> dict[str, pd.DataFrame]:
    if demo:
        import demo_data
        return {code: hist for code, _, _, hist in demo_data.build_dataset()}

    import fetch
    snapshot = fetch.fetch_twse_snapshot()
    if snapshot.empty:
        raise RuntimeError("證交所行情取得失敗，無法決定回測範圍。")
    import universe as universe_mod
    universe = universe_mod.build_core(snapshot, fetch.fetch_stock_info())
    if universe.empty:
        universe = fetch.build_universe(snapshot)
    # 與 build.py 共用同一份 data/history 快取，不重抓
    return fetch.fetch_history(universe["code"].tolist())


# ---------------------------------------------------------------------------
# 勝率平滑
# ---------------------------------------------------------------------------
def first_profitable_exit(df, pos: int, cost: float) -> dict | None:
    """Shared net-target outcome; only fully matured cohorts."""
    result = strategy.outcome(df, pos + 1, cost)
    return result if result and result.get("closed") else None


def calibrate(wins: int, samples: int) -> float:
    """
    (wins + 10) / (samples + 20)，等同於加上「10 勝 10 敗」的先驗。
    樣本 0 時回 50%，樣本大時趨近實際勝率。
    """
    return (wins + C.SMOOTH_WINS) / (samples + C.SMOOTH_N)


def describe(nets: list[float], mdds: list[float]) -> dict:
    """一組淨報酬的完整統計。nets 已經扣過交易成本。"""
    a = np.array(nets, dtype=float)
    m = np.array(mdds, dtype=float)
    wins, losses = a[a > 0], a[a <= 0]
    gross_win, gross_loss = float(wins.sum()), float(-losses.sum())
    n = int(len(a))
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    wr = (len(wins) / n) if n else 0.0
    # 期望值：勝率×平均獲利 ＋ 敗率×平均虧損（avg_loss 本身是負值）
    expectancy = wr * avg_win + (1 - wr) * avg_loss
    return {
        **risk_stats.losses(nets),
        "expectancy": round(expectancy, 3) if n else None,
        "samples": n,
        "wins": int(len(wins)),
        "win_rate": round(float(len(wins) / n * 100), 1) if n else None,
        "calibrated_win_rate": round(calibrate(len(wins), n) * 100, 1),
        "avg_return": round(float(a.mean()), 2) if n else None,
        "median_return": round(float(np.median(a)), 2) if n else None,
        "max_gain": round(float(a.max()), 2) if n else None,
        "max_loss": round(float(a.min()), 2) if n else None,
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "payoff": round(float(wins.mean() / abs(losses.mean())), 2)
                  if len(wins) and len(losses) and losses.mean() != 0 else None,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
        "avg_mdd": round(float(m.mean()), 2) if n else None,
        "worst_mdd": round(float(m.min()), 2) if n else None,
    }


# ---------------------------------------------------------------------------
# 核心：逐日重算分數並記錄樣本
# ---------------------------------------------------------------------------
def load_regimes(demo: bool) -> pd.Series | None:
    """每個交易日的大盤狀態。抓不到就回 None，統計時全部當 sideways。"""
    if demo:
        return None
    try:
        import fetch
        twii = fetch.fetch_twii_history()
        return indicators.regime_series(twii) if twii is not None else None
    except Exception as e:
        log.warning("大盤狀態取得失敗，回測不分多空：%s", e)
        return None


def run_backtest(hist_map: dict[str, pd.DataFrame], lookback_days: int,
                 regimes: pd.Series | None = None) -> dict:
    max_hold = max(C.BACKTEST_HOLD_DAYS)
    cost = C.TOTAL_COST_PCT   # 手續費＋證交稅＋滑價
    cooldown = C.SIGNAL_COOLDOWN_DAYS

    # 先把每檔的特徵一次算完（向量化），回測時只取值不重算
    frames = {}
    for code, hist in hist_map.items():
        if hist is None or len(hist) < C.MIN_BARS + max_hold + 10:
            continue
        df = indicators.compute_frame(hist)
        df.index = pd.to_datetime(df.index)
        frames[code] = df
    if not frames:
        raise RuntimeError("沒有足夠長的歷史資料可以回測。")
    log.info("回測標的：%d 檔", len(frames))

    all_dates = sorted(set().union(*[set(df.index) for df in frames.values()]))
    # 尾端要留 max_hold + 1 根（+1 是因為進場價用的是隔日開盤）
    usable = all_dates[C.MIN_BARS: len(all_dates) - max(max_hold, C.EXIT_MAX_DAYS) - 1]
    if lookback_days > 0:
        usable = usable[-lookback_days:]
    if not usable:
        raise RuntimeError("沒有可用的完整持有期間")
    log.info("回測期間：%s ～ %s（%d 個交易日）",
             str(usable[0])[:10], str(usable[-1])[:10], len(usable))

    import build as build_mod
    signals, all_signals = [], []
    last_signal_day = {}          # code -> 上次採樣是第幾個交易日（用來做冷卻）

    for n, day in enumerate(usable):
        day_rows = []
        for code, df in frames.items():
            pos = df.index.get_indexer([day])[0]
            if pos < C.MIN_BARS - 1 or pos + max(max_hold, C.EXIT_MAX_DAYS) + 1 >= len(df):
                continue
            f = indicators.features_at(df, pos)
            if not f:
                continue
            res = scoring.score_stock(f)
            if res["score"] < C.BACKTEST_MIN_SCORE:
                continue

            # --- 進場價：隔日開盤。收盤後才產生的訊號，當日收盤價已經買不到 ---
            entry = float(df["open"].iloc[pos + 1])
            if not entry or entry <= 0:
                continue

            fwd = {}
            for h in C.BACKTEST_HOLD_DAYS:
                exit_px = float(df["close"].iloc[pos + h])
                gross = (exit_px / entry - 1) * 100
                net = gross - cost                      # 扣掉來回交易成本
                trough = float(df["low"].iloc[pos + 1: pos + h + 1].min())
                fwd[h] = {
                    "net": net,
                    "mdd": (trough / entry - 1) * 100,   # 持有期間最糟的帳面虧損
                }

            reg = "sideways"
            if regimes is not None:
                try:
                    v = regimes.get(day)
                    if isinstance(v, str):
                        reg = v
                except Exception:
                    pass

            ex = first_profitable_exit(df, pos, cost)
            if ex is None:
                continue

            replay = build_mod.build_row(code, code, f, res)
            replay["prev_low"] = float(df["low"].iloc[pos - 1])
            replay["regime"] = reg
            day_rows.append({
                "replay": replay,
                "exit": ex,
                "regime": reg,
                "day_index": n,
                "date": str(day)[:10],
                "code": code,
                "score": res["score"],
                "pattern": res["kind"],
                "risk": res["risk"],
                "rr_ratio": res["rr_ratio"],
                "downside_pct": res["downside_pct"],
                "breakdown": res["breakdown"],
                "hist_calibrated": None,     # 回測當下不能用未來的統計結果
                "fwd": fwd,
            })

        if not day_rows:
            continue

        # 當日名次仍用技術分數排（回測時還沒有勝率可用）
        day_rows.sort(key=scoring.sort_key)

        n_day = len(day_rows)
        for rank, r in enumerate(day_rows, 1):
            r["rank"] = rank
            r["pct"] = (rank - 1) / max(1, n_day)
            all_signals.append(r)
            # --- 冷卻：同一檔在 N 個交易日內只採樣一次 ---
            prev = last_signal_day.get(r["code"])
            if prev is not None and n - prev < cooldown:
                continue
            last_signal_day[r["code"]] = n
            signals.append(r)

        if (n + 1) % 20 == 0:
            log.info("進度 %d/%d（累積 %d 筆有效樣本）", n + 1, len(usable), len(signals))

    if not signals:
        raise RuntimeError("回測期間沒有任何達標訊號，請放寬 MIN_SCORE_TO_SHOW 再試。")

    dates = sorted({r["date"] for r in signals})
    cut_date = dates[min(len(dates) - 1, int(len(dates) * C.OOS_SPLIT))]
    ins = [r for r in signals if r["exit"]["label_end"] < cut_date]
    oos = [r for r in signals if r["date"] >= cut_date]
    wf = walk_forward(all_signals, signals)
    tables = calibration_tables(signals)

    return {
        "strategy": strategy.contract(),
        "research_only": True,
        "capital_policy": strategy.capital_policy(),
        "calibration": tables,
        "in_sample_overall": _pack(ins),
        "generated_at": datetime.now(C.TZ).strftime("%Y-%m-%d %H:%M"),
        "oos_period": (f"{oos[0]['date']} ～ {oos[-1]['date']}" if oos else ""),
        "oos_total": len(oos),
        "oos_overall": _pack(oos, C.PRIMARY_HOLD_DAYS) if oos else None,
        "oos_regime_buckets": stats_by_regime_pattern_bucket(oos),  # OOS 的大盤分層
        "oos_buckets": stats_by_pattern_bucket(oos),        # 排名查表優先用這個
        "oos_score_buckets": stats_by_bucket(oos),
        "deciles_in_sample": stats_by_decile(ins),
        "deciles_oos": wf.get("deciles", []),
        "period": f"{str(usable[0])[:10]} ～ {str(usable[-1])[:10]}",
        "trading_days": len(usable),
        "universe_size": len(frames),
        "total_signals": len(signals),
        "primary_hold_days": C.PRIMARY_HOLD_DAYS,
        "cost_pct": cost,
        "cost_detail": {"fee_tax": C.TRADE_COST_PCT, "slippage": C.SLIPPAGE_PCT},
        "cooldown_days": cooldown,
        "entry_rule": ("訊號日 t 收盤後產生，t+1 開盤進場；之後每天收盤檢查，"
                       "第 %d～%d 個交易日淨利至少 %.1f%% 才算達標；未達標第 %d 日出場（收盤成交模擬）"
                       % (C.EXIT_MIN_DAYS, C.EXIT_MAX_DAYS, C.EXIT_MIN_PROFIT, C.EXIT_MAX_DAYS)),
        "exit_max_days": C.EXIT_MAX_DAYS,
        "topk": stats_by_topk(signals),
        "score_buckets": stats_by_bucket(signals),
        "regime_buckets": stats_by_regime_pattern_bucket(signals),
        "pattern_buckets": stats_by_pattern_bucket(signals),
        "walk_forward": wf,
        "patterns": stats_by_pattern(signals),
        "note": "歷史模擬結果，不代表未來績效。已扣 %.1f%% 來回成本（手續費稅 %.1f%% ＋ 滑價 %.1f%%）。"
                % (cost, C.TRADE_COST_PCT, C.SLIPPAGE_PCT),
    }


# ---------------------------------------------------------------------------
# 統計
# ---------------------------------------------------------------------------
def _pack(sub: list[dict], h: int = 0) -> dict:
    """
    統計一組樣本。**主指標是「N 天內獲利出場的成功率」**，
    h 參數保留是為了相容舊呼叫，實際已不使用固定持有期。
    """
    ex = [s["exit"] for s in sub if s.get("exit")]
    if ex:
        nets = [e["net"] for e in ex]
        mdds = [e["mdd"] for e in ex]
        d = describe(nets, mdds)
        d.update(risk_stats.expected_return_lower(sub))
        wins = sum(1 for e in ex if e["success"])
        d["successes"] = wins
        d["success_rate"] = round(wins / len(ex) * 100, 1)
        d["calibrated_success"] = round(calibrate(wins, len(ex)) * 100, 1)
        d["success_lower"] = round(strategy.wilson_lower(wins, len(ex)), 2)
        d["avg_days"] = round(sum(e["days"] for e in ex) / len(ex), 1)
        d["avg_days_win"] = (round(sum(e["days"] for e in ex if e["success"])
                                   / max(1, wins), 1) if wins else None)
        return d
    return _pack_hold(sub, h or C.PRIMARY_HOLD_DAYS)


def _pack_hold(sub: list[dict], h: int) -> dict:
    if not sub:
        return {**describe([], []), "signal_dates": 0, "ev_lower": None}
    return describe([s["fwd"][h]["net"] for s in sub], [s["fwd"][h]["mdd"] for s in sub])


def stats_by_topk(signals: list[dict]) -> dict:
    out = {}
    for k in C.BACKTEST_TOP_K:
        sub = [s for s in signals if s["rank"] <= k]
        out[f"top{k}"] = {str(h): _pack_hold(sub, h) for h in C.BACKTEST_HOLD_DAYS} if sub else {}
    out["all"] = {str(h): _pack_hold(signals, h) for h in C.BACKTEST_HOLD_DAYS}
    return out


def stats_by_bucket(signals: list[dict]) -> list[dict]:
    """依技術分數級距統計（持有 5 日）。"""
    h = C.PRIMARY_HOLD_DAYS
    out = []
    for lo, hi in C.BACKTEST_SCORE_BUCKETS:
        sub = [s for s in signals if lo <= s["score"] < hi]
        d = _pack(sub, h) if sub else describe([], [])
        out.append({"lo": lo, "hi": hi, "hold_days": h, **d})
    return out


def stats_by_pattern_bucket(signals: list[dict]) -> list[dict]:
    """
    依「型態 + 分數級距」統計。這是排名的第一順位依據——
    同樣 65 分，帶量突破跟低檔止跌轉強的實際勝率可能差很多。
    """
    h = C.PRIMARY_HOLD_DAYS
    out = []
    patterns = sorted(set(s["pattern"] for s in signals))
    for pat in patterns:
        for lo, hi in C.BACKTEST_SCORE_BUCKETS:
            sub = [s for s in signals if s["pattern"] == pat and lo <= s["score"] < hi]
            if not sub:
                continue
            out.append({"pattern": pat, "lo": lo, "hi": hi, "hold_days": h, **_pack(sub, h)})
    return out


def stats_by_regime_pattern_bucket(signals: list[dict]) -> list[dict]:
    """大盤狀態 × 型態 × 分數級距。這是排名查表的第一順位。"""
    h = C.PRIMARY_HOLD_DAYS
    out = []
    keys = sorted(set((s.get("regime", "sideways"), s["pattern"]) for s in signals))
    for reg, pat in keys:
        for lo, hi in C.BACKTEST_SCORE_BUCKETS:
            sub = [s for s in signals
                   if s.get("regime", "sideways") == reg and s["pattern"] == pat
                   and lo <= s["score"] < hi]
            if not sub:
                continue
            out.append({"regime": reg, "pattern": pat, "lo": lo, "hi": hi,
                        "hold_days": h, **_pack(sub, h)})
    return out


def calibration_tables(signals: list[dict]) -> dict:
    return {"overall": _pack(signals),
            "regime_buckets": stats_by_regime_pattern_bucket(signals),
            "pattern_buckets": stats_by_pattern_bucket(signals),
            "score_buckets": stats_by_bucket(signals)}


def training_before(signals: list[dict], test_start: str) -> list[dict]:
    # Purge overlapping outcome windows, not merely signal dates.
    return [r for r in signals if r["date"] < test_start
            and r["exit"]["label_end"] < test_start]


def walk_forward(signals: list[dict], calibration: list[dict] | None = None) -> dict:
    """Frozen chronological folds replay the same attach/momentum/rank functions.

    No overseas or next-day model adjustment is applied to the live target rank,
    so historical and live feature paths remain identical.
    """
    import build as build_mod
    from copy import deepcopy
    calibration = signals if calibration is None else calibration
    dates = sorted({s["date"] for s in signals})
    k = max(2, C.WF_FOLDS)
    if len(dates) < k * 5:
        return {"available": False, "reason": "交易日期不足，尚無獨立驗證"}
    folds = [list(x) for x in np.array_split(dates, k)]
    by_day = {}
    for r in signals:
        by_day.setdefault(r["date"], []).append(r)
    picked, tested, fold_stats, comparisons, success_comparisons = [], [], [], [], []
    for i, test_dates in enumerate(folds[1:], 1):
        prior = training_before(calibration, test_dates[0])
        if len(prior) < C.MIN_SAMPLES_SCORE:
            continue
        tables = calibration_tables(prior)
        fold_picked = []
        for day in test_dates:
            source = by_day[day]
            rows = [deepcopy(r["replay"]) for r in source]
            ranking.attach(rows, tables)
            build_mod.add_momentum(rows)
            build_mod.add_final_score(rows)
            ranked = ranking.sort(rows)
            lookup = {s["code"]: s for s in source}
            n = len(ranked)
            # Research replay remains measurable while capital policy blocks buys.
            eligible = [r for r in ranked if r["research_eligible"]][:C.WF_TOP_N]
            today = [lookup[r["code"]] for r in eligible]
            picked.extend(today)
            fold_picked.extend(today)
            if today:
                comparisons.append(sum(s["exit"]["net"] for s in today) / len(today)
                                   - sum(s["exit"]["net"] for s in source) / len(source))
                success_comparisons.append(100 * (sum(s["exit"]["success"] for s in today) / len(today)
                                           - sum(s["exit"]["success"] for s in source) / len(source)))
            for rank, r in enumerate(ranked, 1):
                tested.append({**lookup[r["code"]], "rank": rank,
                               "pct": (rank - 1) / max(n, 1)})
        fs = _pack(fold_picked)
        fold_stats.append({"fold": i, "samples": fs["samples"],
                           "expectancy": fs["expectancy"],
                           "profit_factor": fs["profit_factor"],
                           "win_rate": fs["win_rate"],
                           "success_rate": fs.get("success_rate"),
                           "loss_rate": fs.get("loss_rate"), "worst_net": fs.get("worst_net"),
                           "train_last_outcome": max(r["exit"]["label_end"] for r in prior),
                           "test_start": test_dates[0]})
    if not tested:
        return {"available": False, "reason": "成熟訓練樣本不足"}
    positive = sum(1 for f in fold_stats if (f["expectancy"] or 0) > 0)
    summary = _pack(picked)
    deciles = stats_by_decile(tested)
    return {"available": bool(picked), "reason": "沒有通過候選條件的訊號" if not picked else "",
            "selection": "research_candidates", "research_only": True,
            "folds": k, "top_n": C.WF_TOP_N, "fold_stats": fold_stats,
            "positive_folds": positive, "total_folds": len(fold_stats),
            "stability": positive / len(fold_stats) if fold_stats else 0,
            "period": f"{folds[1][0]} ～ {folds[-1][-1]}",
            "deciles": deciles, "rank_order": risk_stats.rank_order_check(deciles), "baseline": _pack(tested),
            "lift_ev": round(float(np.mean(comparisons)), 3) if comparisons else None,
            "lift_success_pp": round(float(np.mean(success_comparisons)), 3) if success_comparisons else None,
            "note": "同日跨股票與連續日結果相關；筆數不等於獨立樣本數。", **summary}


DECILES = [("Top 10%", 0.0, 0.10), ("10–30%", 0.10, 0.30),
           ("30–60%", 0.30, 0.60), ("Bottom 40%", 0.60, 1.01)]


def stats_by_decile(signals: list[dict]) -> list[dict]:
    """
    依當日名次百分位分組。**這是驗證「排名越前，期望值越高」的核心指標**——
    如果 Top 10% 的 EV 沒有明顯高於後段，排名就沒有價值，勝率再高也沒用。
    """
    h = C.PRIMARY_HOLD_DAYS
    out = []
    for label, lo, hi in DECILES:
        sub = [s for s in signals if lo <= s.get("pct", 1) < hi]
        d = _pack(sub, h) if sub else describe([], [])
        out.append({"label": label, "lo": lo, "hi": hi, **d})
    return out


def stats_by_pattern(signals: list[dict]) -> list[dict]:
    """不分級距，單看型態的整體表現。"""
    h = C.PRIMARY_HOLD_DAYS
    out = []
    for pat in sorted(set(s["pattern"] for s in signals)):
        sub = [s for s in signals if s["pattern"] == pat]
        out.append({"pattern": pat, "hold_days": h, **_pack(sub, h)})
    return sorted(out, key=lambda x: -(x["calibrated_win_rate"] or 0))


# ---------------------------------------------------------------------------
# 報表
# ---------------------------------------------------------------------------
def print_report(bt: dict, demo: bool) -> None:
    print("回測口徑：", bt["strategy"])
    print("期間：", bt["period"], "股票：", bt["universe_size"], "校準樣本：", bt["total_signals"])
    if demo:
        print("示範資料只驗證流程，不代表市場績效")
    wf = bt.get("walk_forward") or {}
    if wf.get("available"):
        print("逐段樣本外：", {k: wf.get(k) for k in
              ("samples", "success_rate", "win_rate", "expectancy", "profit_factor", "lift_ev")})
        print("各段：", wf.get("fold_stats"))
        for row in wf.get("deciles", []):
            print(row["label"], {k: row.get(k) for k in
                  ("samples", "success_rate", "expectancy", "profit_factor")})
    else:
        print("尚無可用樣本外驗證：", wf.get("reason"))
    print("股票池回溯有選樣限制；同日與連續日訊號相關，筆數不等於獨立樣本數。")


def main() -> int:
    ap = argparse.ArgumentParser(description="回測：用歷史結果決定排名依據")
    ap.add_argument("--demo", action="store_true", help="用模擬資料跑，只驗證流程")
    ap.add_argument("--days", type=int, default=0, help="只回測最近 N 個交易日（0＝全部）")
    ap.add_argument("--no-save", action="store_true", help="不要寫入 data/backtest.json")
    args = ap.parse_args()

    hist_map = load_universe_history(args.demo)
    regimes = load_regimes(args.demo)
    bt = run_backtest(hist_map, args.days, regimes)
    bt["mode"] = "demo" if args.demo else "live"
    print_report(bt, args.demo)

    # 模擬資料的統計沒有意義，預設不寫入，免得儀表板拿假勝率去排名
    if args.no_save or args.demo:
        log.info("未寫入 %s（示範模式或指定不儲存）", C.BACKTEST_JSON)
        return 0

    path = ROOT / C.BACKTEST_JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bt, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("已寫入 %s，下次 build 排名就會改用歷史勝率", C.BACKTEST_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
