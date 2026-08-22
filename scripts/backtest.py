# -*- coding: utf-8 -*-
"""
backtest.py — 歷史回測（決定排名的依據）

這支的輸出不只是「看看分數準不準」，而是**首頁排名的實際依據**。
儀表板會拿這裡算出來的歷史勝率去排序，技術分數只是分層用的特徵。

交易假設（刻意保守，寧可低估）：
  - 訊號用第 t 日**收盤後**的資料算出來
  - 進場價是 **t+1 日的開盤價**，不是 t 日收盤價
    （t 日收盤價在收盤後已經買不到了，用它當進場價等於偷看）
  - 出場價是 t+h 日的收盤價
  - 淨報酬 = 毛報酬 − 來回交易成本（config.TRADE_COST_PCT，預設 0.3%）
  - **淨報酬 > 0 才算贏**
  - 同一檔股票出訊號後 config.SIGNAL_COOLDOWN_DAYS 個交易日內不重複採樣，
    避免同一段行情被算成好幾個獨立樣本，把樣本數灌水

統計輸出：
  - 以持有 5 日為主，同時保留 3 / 10 / 20 日
  - 依「分數級距」統計
  - 再依「pattern（型態）＋ 分數級距」統計，這才是排名的主要依據
  - 勝率一律做平滑：(wins + 10) / (samples + 20)，避免 3 戰 3 勝就變成 100%

不使用 sklearn，只用 numpy / pandas。

用法：
    python scripts/backtest.py                 # 用線上資料跑（慢，約 10～25 分鐘）
    python scripts/backtest.py --demo          # 用模擬資料跑，僅驗證程式流程
    python scripts/backtest.py --days 250      # 只回測最近 250 個交易日
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
    universe = fetch.build_universe(snapshot)
    return fetch.fetch_history_yf(universe["code"].tolist())


# ---------------------------------------------------------------------------
# 勝率平滑
# ---------------------------------------------------------------------------
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
    return {
        "samples": n,
        "wins": int(len(wins)),
        "win_rate": round(float(len(wins) / n * 100), 1) if n else None,
        "calibrated_win_rate": round(calibrate(len(wins), n) * 100, 1),
        "avg_return": round(float(a.mean()), 2) if n else None,
        "median_return": round(float(np.median(a)), 2) if n else None,
        "max_gain": round(float(a.max()), 2) if n else None,
        "max_loss": round(float(a.min()), 2) if n else None,
        "avg_win": round(float(wins.mean()), 2) if len(wins) else 0.0,
        "avg_loss": round(float(losses.mean()), 2) if len(losses) else 0.0,
        "payoff": round(float(wins.mean() / abs(losses.mean())), 2)
                  if len(wins) and len(losses) and losses.mean() != 0 else None,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
        "avg_mdd": round(float(m.mean()), 2) if n else None,
        "worst_mdd": round(float(m.min()), 2) if n else None,
    }


# ---------------------------------------------------------------------------
# 核心：逐日重算分數並記錄樣本
# ---------------------------------------------------------------------------
def run_backtest(hist_map: dict[str, pd.DataFrame], lookback_days: int) -> dict:
    max_hold = max(C.BACKTEST_HOLD_DAYS)
    cost = C.TRADE_COST_PCT
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
    usable = all_dates[C.MIN_BARS: len(all_dates) - max_hold - 1]
    if lookback_days > 0:
        usable = usable[-lookback_days:]
    log.info("回測期間：%s ～ %s（%d 個交易日）",
             str(usable[0])[:10], str(usable[-1])[:10], len(usable))

    signals = []
    last_signal_day = {}          # code -> 上次採樣是第幾個交易日（用來做冷卻）

    for n, day in enumerate(usable):
        day_rows = []
        for code, df in frames.items():
            pos = df.index.get_indexer([day])[0]
            if pos < C.MIN_BARS - 1 or pos + max_hold + 1 >= len(df):
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

            day_rows.append({
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

        rank = 0
        for r in day_rows:
            # --- 冷卻：同一檔在 N 個交易日內只採樣一次 ---
            prev = last_signal_day.get(r["code"])
            if prev is not None and n - prev < cooldown:
                continue
            last_signal_day[r["code"]] = n
            rank += 1
            r["rank"] = rank
            signals.append(r)

        if (n + 1) % 20 == 0:
            log.info("進度 %d/%d（累積 %d 筆有效樣本）", n + 1, len(usable), len(signals))

    if not signals:
        raise RuntimeError("回測期間沒有任何達標訊號，請放寬 MIN_SCORE_TO_SHOW 再試。")

    return {
        "generated_at": datetime.now(C.TZ).strftime("%Y-%m-%d %H:%M"),
        "period": f"{str(usable[0])[:10]} ～ {str(usable[-1])[:10]}",
        "trading_days": len(usable),
        "universe_size": len(frames),
        "total_signals": len(signals),
        "primary_hold_days": 5,
        "cost_pct": cost,
        "cooldown_days": cooldown,
        "entry_rule": "訊號日 t 收盤後產生，t+1 開盤價進場，t+h 收盤價出場",
        "topk": stats_by_topk(signals),
        "score_buckets": stats_by_bucket(signals),
        "pattern_buckets": stats_by_pattern_bucket(signals),
        "patterns": stats_by_pattern(signals),
        "note": "歷史模擬結果，不代表未來績效。已扣 %.1f%% 來回交易成本，未計滑價。" % cost,
    }


# ---------------------------------------------------------------------------
# 統計
# ---------------------------------------------------------------------------
def _pack(sub: list[dict], h: int) -> dict:
    return describe([s["fwd"][h]["net"] for s in sub], [s["fwd"][h]["mdd"] for s in sub])


def stats_by_topk(signals: list[dict]) -> dict:
    out = {}
    for k in C.BACKTEST_TOP_K:
        sub = [s for s in signals if s["rank"] <= k]
        out[f"top{k}"] = {str(h): _pack(sub, h) for h in C.BACKTEST_HOLD_DAYS} if sub else {}
    out["all"] = {str(h): _pack(signals, h) for h in C.BACKTEST_HOLD_DAYS}
    return out


def stats_by_bucket(signals: list[dict]) -> list[dict]:
    """依技術分數級距統計（持有 5 日）。"""
    h = 5
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
    h = 5
    out = []
    patterns = sorted(set(s["pattern"] for s in signals))
    for pat in patterns:
        for lo, hi in C.BACKTEST_SCORE_BUCKETS:
            sub = [s for s in signals if s["pattern"] == pat and lo <= s["score"] < hi]
            if not sub:
                continue
            out.append({"pattern": pat, "lo": lo, "hi": hi, "hold_days": h, **_pack(sub, h)})
    return out


def stats_by_pattern(signals: list[dict]) -> list[dict]:
    """不分級距，單看型態的整體表現。"""
    h = 5
    out = []
    for pat in sorted(set(s["pattern"] for s in signals)):
        sub = [s for s in signals if s["pattern"] == pat]
        out.append({"pattern": pat, "hold_days": h, **_pack(sub, h)})
    return sorted(out, key=lambda x: -(x["calibrated_win_rate"] or 0))


# ---------------------------------------------------------------------------
# 報表
# ---------------------------------------------------------------------------
def print_report(bt: dict, demo: bool) -> None:
    print("\n" + "=" * 74)
    print(f"回測期間 {bt['period']}｜{bt['universe_size']} 檔｜{bt['total_signals']} 筆有效樣本")
    print(f"進場規則：{bt['entry_rule']}")
    print(f"交易成本 {bt['cost_pct']}%（已扣除）｜訊號冷卻 {bt['cooldown_days']} 個交易日")
    if demo:
        print("⚠️ 這是【模擬資料】跑出來的，只驗證程式流程，數字沒有任何預測意義。")
    print("=" * 74)

    for k in ["top1", "top3", "top5", "top10", "all"]:
        if not bt["topk"].get(k):
            continue
        print(f"\n■ {k.upper()}（淨報酬，已扣成本）")
        print(f"{'持有':>4} {'樣本':>6} {'勝率':>7} {'平滑勝率':>9} {'平均':>8} "
              f"{'中位':>8} {'最大虧損':>9} {'PF':>6}")
        for h in C.BACKTEST_HOLD_DAYS:
            d = bt["topk"][k].get(str(h))
            if not d or not d["samples"]:
                continue
            print(f"{h:>3}日 {d['samples']:>6} {d['win_rate']:>6.1f}% "
                  f"{d['calibrated_win_rate']:>8.1f}% {d['avg_return']:>7.2f}% "
                  f"{d['median_return']:>7.2f}% {d['max_loss']:>8.2f}% "
                  f"{(d['profit_factor'] if d['profit_factor'] is not None else 0):>6.2f}")

    print("\n■ 分數級距 vs 持有 5 日")
    for b in bt["score_buckets"]:
        if b["samples"]:
            print(f"  {b['lo']:>3}–{b['hi']:<3} 分｜樣本 {b['samples']:>5}｜"
                  f"勝率 {b['win_rate']:>5.1f}%（平滑 {b['calibrated_win_rate']:>5.1f}%）｜"
                  f"平均 {b['avg_return']:+.2f}%｜PF {b['profit_factor']}")
        else:
            print(f"  {b['lo']:>3}–{b['hi']:<3} 分｜無樣本")

    print("\n■ 型態整體表現（持有 5 日）")
    for p in bt["patterns"]:
        flag = "" if p["samples"] >= C.PATTERN_MIN_SAMPLES else "  ← 樣本不足，排名不會採用"
        print(f"  {p['pattern']:<12}樣本 {p['samples']:>5}｜勝率 {p['win_rate']:>5.1f}%"
              f"（平滑 {p['calibrated_win_rate']:>5.1f}%）｜平均 {p['avg_return']:+.2f}%"
              f"｜PF {p['profit_factor']}{flag}")

    usable = [b for b in bt["pattern_buckets"] if b["samples"] >= C.PATTERN_MIN_SAMPLES]
    print(f"\n■ 型態 × 分數級距：共 {len(bt['pattern_buckets'])} 組，"
          f"其中 {len(usable)} 組樣本達 {C.PATTERN_MIN_SAMPLES} 筆，排名時會優先採用")
    for b in sorted(usable, key=lambda x: -x["calibrated_win_rate"])[:10]:
        print(f"  {b['pattern']:<12}{b['lo']:>3}–{b['hi']:<3} 分｜樣本 {b['samples']:>4}｜"
              f"平滑勝率 {b['calibrated_win_rate']:>5.1f}%｜平均 {b['avg_return']:+.2f}%")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description="回測：用歷史結果決定排名依據")
    ap.add_argument("--demo", action="store_true", help="用模擬資料跑，只驗證流程")
    ap.add_argument("--days", type=int, default=0, help="只回測最近 N 個交易日（0＝全部）")
    ap.add_argument("--no-save", action="store_true", help="不要寫入 data/backtest.json")
    args = ap.parse_args()

    hist_map = load_universe_history(args.demo)
    bt = run_backtest(hist_map, args.days)
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
