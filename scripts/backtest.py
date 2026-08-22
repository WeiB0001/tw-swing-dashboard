# -*- coding: utf-8 -*-
"""
backtest.py — 歷史回測

回答的是規格書第十四、十五點：
    「分數高的股票，未來真的比較會漲嗎？」

做法（逐日重跑一次今天的流程）：
  1. 抓掃描池所有個股的長期日線
  2. 對每一個交易日 t：用「只到 t 為止」的資料算出每檔的獲利可能性分數
     （嚴格避免用到未來資料）
  3. 取當日排名前 K 名，計算持有 3 / 5 / 10 / 20 天的報酬
  4. 統計勝率、平均報酬、中位數報酬、最大虧損、最大回撤、盈虧比、Profit Factor
  5. 另外依分數級距統計勝率，寫進 data/backtest.json

只有跑過這支，儀表板才會顯示「歷史相似訊號勝率」。
沒跑過的話，頁面上只會有模型分數，並明確標示那不是真實機率。

用法：
    python scripts/backtest.py                 # 用線上資料跑（慢，約 10～20 分鐘）
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
# 取得回測用的歷史資料
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
# 核心：逐日重算分數
# ---------------------------------------------------------------------------
def run_backtest(hist_map: dict[str, pd.DataFrame], lookback_days: int) -> dict:
    max_hold = max(C.BACKTEST_HOLD_DAYS)

    # 先把每檔的特徵一次算完（向量化），回測時只做取值，不重算 rolling
    frames, closes = {}, {}
    for code, hist in hist_map.items():
        if hist is None or len(hist) < C.MIN_BARS + max_hold + 10:
            continue
        df = indicators.compute_frame(hist)
        df.index = pd.to_datetime(df.index)
        frames[code] = df
        closes[code] = df["close"]
    if not frames:
        raise RuntimeError("沒有足夠長的歷史資料可以回測。")
    log.info("回測標的：%d 檔", len(frames))

    # 取所有股票共同的交易日，往前留暖身期、往後留持有期
    all_dates = sorted(set().union(*[set(df.index) for df in frames.values()]))
    usable = all_dates[C.MIN_BARS: len(all_dates) - max_hold]
    if lookback_days > 0:
        usable = usable[-lookback_days:]
    log.info("回測期間：%s ～ %s（%d 個交易日）",
             str(usable[0])[:10], str(usable[-1])[:10], len(usable))

    signals = []          # 每一筆「當天被選出的標的」
    for n, day in enumerate(usable, 1):
        day_rows = []
        for code, df in frames.items():
            pos = df.index.get_indexer([day])[0]
            if pos < C.MIN_BARS - 1 or pos + max_hold >= len(df):
                continue
            f = indicators.features_at(df, pos)
            if not f:
                continue
            res = scoring.score_stock(f)
            if res["score"] < C.BACKTEST_MIN_SCORE:
                continue

            entry = float(df["close"].iloc[pos])
            fwd = {}
            for h in C.BACKTEST_HOLD_DAYS:
                exit_px = float(df["close"].iloc[pos + h])
                # 期間最低價，用來衡量抱單過程中的最大痛苦（最大回撤）
                trough = float(df["low"].iloc[pos + 1: pos + h + 1].min())
                fwd[h] = {
                    "ret": (exit_px / entry - 1) * 100,
                    "mdd": (trough / entry - 1) * 100,
                }
            day_rows.append({
                "date": str(day)[:10], "code": code,
                "score": res["score"], "risk": res["risk"],
                "rr_ratio": res["rr_ratio"], "downside_pct": res["downside_pct"],
                "breakdown": res["breakdown"], "hist_winrate": None,
                "fwd": fwd,
            })

        if not day_rows:
            continue
        day_rows.sort(key=scoring.sort_key)
        for rank, r in enumerate(day_rows, 1):
            r["rank"] = rank
            signals.append(r)

        if n % 20 == 0:
            log.info("進度 %d/%d（累積 %d 筆訊號）", n, len(usable), len(signals))

    if not signals:
        raise RuntimeError("回測期間沒有任何達標訊號，請放寬 MIN_SCORE_TO_SHOW 再試。")

    return {
        "generated_at": datetime.now(C.TZ).strftime("%Y-%m-%d %H:%M"),
        "period": f"{str(usable[0])[:10]} ～ {str(usable[-1])[:10]}",
        "trading_days": len(usable),
        "universe_size": len(frames),
        "total_signals": len(signals),
        "primary_hold_days": 5,
        "topk": _stats_by_topk(signals),
        "score_buckets": _stats_by_bucket(signals),
        "note": "此為歷史模擬結果，不代表未來績效。手續費、稅、滑價未計入。",
    }


# ---------------------------------------------------------------------------
# 統計
# ---------------------------------------------------------------------------
def _describe(rets: list[float], mdds: list[float]) -> dict:
    a = np.array(rets, dtype=float)
    wins, losses = a[a > 0], a[a <= 0]
    gross_win, gross_loss = wins.sum(), -losses.sum()
    return {
        "samples": int(len(a)),
        "win_rate": round(float((a > 0).mean() * 100), 1),
        "avg_return": round(float(a.mean()), 2),
        "median_return": round(float(np.median(a)), 2),
        "max_gain": round(float(a.max()), 2),
        "max_loss": round(float(a.min()), 2),
        "avg_win": round(float(wins.mean()), 2) if len(wins) else 0.0,
        "avg_loss": round(float(losses.mean()), 2) if len(losses) else 0.0,
        # 盈虧比：平均獲利 / 平均虧損
        "payoff": round(float(wins.mean() / abs(losses.mean())), 2)
                  if len(wins) and len(losses) and losses.mean() != 0 else None,
        # Profit Factor：總獲利 / 總虧損
        "profit_factor": round(float(gross_win / gross_loss), 2) if gross_loss > 0 else None,
        # 最大回撤：持有期間最糟的帳面虧損（平均與最差）
        "avg_mdd": round(float(np.mean(mdds)), 2),
        "worst_mdd": round(float(np.min(mdds)), 2),
    }


def _stats_by_topk(signals: list[dict]) -> dict:
    out = {}
    for k in C.BACKTEST_TOP_K:
        sub = [s for s in signals if s["rank"] <= k]
        out[f"top{k}"] = {
            str(h): _describe([s["fwd"][h]["ret"] for s in sub],
                              [s["fwd"][h]["mdd"] for s in sub])
            for h in C.BACKTEST_HOLD_DAYS
        }
    # 對照組：所有達標訊號（不限名次），用來看排名到底有沒有鑑別力
    out["all"] = {
        str(h): _describe([s["fwd"][h]["ret"] for s in signals],
                          [s["fwd"][h]["mdd"] for s in signals])
        for h in C.BACKTEST_HOLD_DAYS
    }
    return out


def _stats_by_bucket(signals: list[dict]) -> list[dict]:
    """依分數級距統計，供儀表板顯示「歷史相似訊號勝率」。"""
    h = 5   # 以持有 5 天為主
    out = []
    for lo, hi in C.BACKTEST_SCORE_BUCKETS:
        sub = [s for s in signals if lo <= s["score"] < hi]
        if not sub:
            out.append({"lo": lo, "hi": hi, "samples": 0,
                        "win_rate": None, "avg_return": None})
            continue
        d = _describe([s["fwd"][h]["ret"] for s in sub], [s["fwd"][h]["mdd"] for s in sub])
        out.append({"lo": lo, "hi": hi, "samples": d["samples"],
                    "win_rate": d["win_rate"], "avg_return": d["avg_return"],
                    "hold_days": h})
    return out


# ---------------------------------------------------------------------------
def print_report(bt: dict, demo: bool) -> None:
    print("\n" + "=" * 66)
    print(f"回測期間 {bt['period']}｜{bt['universe_size']} 檔｜{bt['total_signals']} 筆訊號")
    if demo:
        print("⚠️ 這是【模擬資料】跑出來的，只驗證程式流程，數字沒有任何預測意義。")
    print("=" * 66)
    for k in ["top1", "top3", "top5", "top10", "all"]:
        if k not in bt["topk"]:
            continue
        print(f"\n■ {k.upper()}")
        print(f"{'持有':>4} {'樣本':>6} {'勝率':>7} {'平均':>8} {'中位':>8} "
              f"{'最大虧損':>9} {'盈虧比':>7} {'PF':>6}")
        for h in C.BACKTEST_HOLD_DAYS:
            d = bt["topk"][k][str(h)]
            print(f"{h:>3}日 {d['samples']:>6} {d['win_rate']:>6.1f}% "
                  f"{d['avg_return']:>7.2f}% {d['median_return']:>7.2f}% "
                  f"{d['max_loss']:>8.2f}% "
                  f"{(d['payoff'] if d['payoff'] is not None else 0):>7.2f} "
                  f"{(d['profit_factor'] if d['profit_factor'] is not None else 0):>6.2f}")
    print("\n■ 分數級距 vs 持有 5 日勝率")
    for b in bt["score_buckets"]:
        if b["samples"]:
            print(f"  {b['lo']:>3}–{b['hi']:<3} 分｜樣本 {b['samples']:>5}｜"
                  f"勝率 {b['win_rate']:>5.1f}%｜平均 {b['avg_return']:+.2f}%")
        else:
            print(f"  {b['lo']:>3}–{b['hi']:<3} 分｜無樣本")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description="獲利可能性分數的歷史回測")
    ap.add_argument("--demo", action="store_true", help="用模擬資料跑，只驗證流程")
    ap.add_argument("--days", type=int, default=0, help="只回測最近 N 個交易日（0＝全部）")
    ap.add_argument("--no-save", action="store_true", help="不要寫入 data/backtest.json")
    args = ap.parse_args()

    hist_map = load_universe_history(args.demo)
    bt = run_backtest(hist_map, args.days)
    bt["mode"] = "demo" if args.demo else "live"
    print_report(bt, args.demo)

    # 模擬資料的統計沒有意義，預設不寫入，免得儀表板顯示假勝率
    if args.no_save or args.demo:
        log.info("未寫入 %s（示範模式或指定不儲存）", C.BACKTEST_JSON)
        return 0

    path = ROOT / C.BACKTEST_JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bt, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("已寫入 %s，下次 build 會自動顯示歷史勝率", C.BACKTEST_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
