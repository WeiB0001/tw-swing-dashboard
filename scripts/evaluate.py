# -*- coding: utf-8 -*-
"""
evaluate.py — 排名的歷史驗證：前幾名隔日到底表現如何？

回應「排名前幾名隔日下跌機率偏高、甚至暴跌」這個實測回饋。
這支不改任何模型，只是**把首頁的排序邏輯逐日重放一遍**，然後老實地量出來：

  - Top 1 / 3 / 5 / 10 隔日的勝率、平均報酬、中位數
  - **大跌率**（隔日跌超過 NEXTDAY_CRASH_PCT）與最差單筆
  - 隔日進入當日漲幅前段的命中率（也就是「有沒有猜中官方漲幅排行」）
  - 跟「全體達標標的」的對照組比較，看排名到底有沒有鑑別力

禁止 look-ahead：第 t 日收盤排名，t+1 開盤買、t+1 收盤賣。

用法：
    python scripts/evaluate.py --days 120          # 用線上資料（走既有快取）
    python scripts/evaluate.py --demo --days 80    # 用模擬資料驗證流程
    python scripts/evaluate.py --days 120 --no-guard   # 關掉隔日風險過濾，比較差異
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import build as build_mod
import config as C
import indicators
import nextday as nextday_mod
import plan as plan_mod
import scoring

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("evaluate")


def load_history(demo: bool) -> dict[str, pd.DataFrame]:
    if demo:
        import demo_data
        return {c: h for c, _, _, h in demo_data.build_dataset()}
    import fetch
    snap = fetch.fetch_twse_snapshot()
    if snap.empty:
        raise RuntimeError("證交所行情取得失敗")
    uni = fetch.build_universe(snap)
    return fetch.fetch_history(uni["code"].tolist())


def describe(rets: list[float]) -> dict:
    if not rets:
        return {"n": 0}
    a = np.array(rets, dtype=float)
    return {
        "n": len(a),
        "win": round(float((a > 0).mean() * 100), 1),
        "avg": round(float(a.mean()), 2),
        "med": round(float(np.median(a)), 2),
        "crash": round(float((a <= C.NEXTDAY_CRASH_PCT).mean() * 100), 1),
        "worst": round(float(a.min()), 2),
        "p10": round(float(np.percentile(a, 10)), 2),
    }


def run(hist_map: dict, days: int, use_guard: bool) -> dict:
    frames = {}
    for code, h in hist_map.items():
        if h is None or len(h) < C.MIN_BARS + 3:
            continue
        try:
            df = indicators.compute_frame(h)
            df.index = pd.to_datetime(df.index)
            frames[code] = df
        except Exception:
            continue
    if not frames:
        raise RuntimeError("沒有足夠的歷史資料")

    nd_stats = nextday_mod.build_stats(hist_map) if use_guard else {}

    all_dates = sorted(set().union(*[set(d.index) for d in frames.values()]))
    usable = all_dates[C.MIN_BARS: len(all_dates) - 1]
    if days > 0:
        usable = usable[-days:]
    log.info("持有 %d 天｜驗證期間：%s ～ %s（%d 個交易日）%s", C.HOLD_DAYS,
             str(usable[0])[:10], str(usable[-1])[:10], len(usable),
             "" if use_guard else "｜已關閉隔日風險過濾")

    topk = {k: [] for k in (1, 3, 5, 10)}
    every, hits = [], {k: 0 for k in (1, 3, 5, 10)}
    day_count = 0

    for day in usable:
        rows, nxt = [], {}
        for code, df in frames.items():
            pos = df.index.get_indexer([day])[0]
            if pos < C.MIN_BARS - 1 or pos + 1 + C.HOLD_DAYS >= len(df):
                continue
            f = indicators.features_at(df, pos)
            if not f:
                continue
            entry = float(df["open"].iloc[pos + 1])
            if entry <= 0:
                continue
            res = scoring.score_stock(f)
            if res["score"] < C.WEAK_SCORE:
                continue

            row = build_mod.build_row(code, code, f, res)
            row["prev_low"] = float(df["low"].iloc[pos - 1]) if pos > 0 else None
            rows.append(row)
            # 隔日：開盤買、收盤賣（實際可執行的做法）
            # 跟實際做法一致：t+1 開盤買、持有 HOLD_DAYS 天後收盤賣
            nxt[code] = (float(df["close"].iloc[pos + 1 + C.HOLD_DAYS]) / entry - 1) * 100

        if len(rows) < 5:
            continue
        day_count += 1

        build_mod.add_momentum(rows)
        if use_guard:
            nextday_mod.attach(rows, nd_stats)
        else:
            for r in rows:
                r["nextday"], r["nextday_source"], r["crash_penalty"] = None, "關閉", 0.0
        build_mod.add_final_score(rows)
        ranked = build_mod.sort_by_final(rows)

        # 當日「隔日漲幅前 20%」的門檻，用來看有沒有猜中強勢股
        vals = sorted(nxt.values(), reverse=True)
        cut = vals[max(0, int(len(vals) * 0.2) - 1)] if vals else 0

        for k in topk:
            picked = ranked[:k]
            for r in picked:
                v = nxt.get(r["code"])
                if v is not None:
                    topk[k].append(v)
                    if v >= cut:
                        hits[k] += 1
        every.extend(nxt.get(r["code"]) for r in ranked if nxt.get(r["code"]) is not None)

    out = {"days": day_count, "guard": use_guard,
           "top": {k: describe(v) for k, v in topk.items()},
           "all": describe(every),
           "hit": {k: (round(hits[k] / max(1, len(topk[k])) * 100, 1)) for k in topk}}
    return out


def report(r: dict) -> None:
    print("\n" + "=" * 72)
    print(f"排名驗證：{r['days']} 個交易日｜隔日風險過濾 {'開啟' if r['guard'] else '關閉'}")
    print(f"進場假設：第 t 日收盤排名，t+1 開盤買、持有 {C.HOLD_DAYS} 天後收盤賣")
    print("=" * 72)
    print(f"{'':<8}{'樣本':>6}{'上漲率':>8}{'平均':>8}{'中位':>8}"
          f"{'大跌率':>8}{'最差一成':>10}{'最差':>8}{'猜中強勢':>10}")
    for k in (1, 3, 5, 10):
        d = r["top"][k]
        if not d.get("n"):
            continue
        print(f"Top {k:<4}{d['n']:>6}{d['win']:>7.1f}%{d['avg']:>7.2f}%{d['med']:>7.2f}%"
              f"{d['crash']:>7.1f}%{d['p10']:>9.2f}%{d['worst']:>7.2f}%"
              f"{r['hit'][k]:>9.1f}%")
    d = r["all"]
    if d.get("n"):
        print(f"{'全體':<7}{d['n']:>6}{d['win']:>7.1f}%{d['avg']:>7.2f}%{d['med']:>7.2f}%"
              f"{d['crash']:>7.1f}%{d['p10']:>9.2f}%{d['worst']:>7.2f}%")
    print("\n「大跌率」= 持有期間跌幅超過 %.1f%% 的比例；「猜中強勢」= 隔日進入當日漲幅前 20%% 的比例。"
          % abs(C.NEXTDAY_CRASH_PCT))
    print("要看的是：Top 1 是否優於 Top 10、Top 10 是否優於全體。三者差不多就代表排名沒有鑑別力。")


def main() -> int:
    ap = argparse.ArgumentParser(description="排名的歷史驗證")
    ap.add_argument("--demo", action="store_true", help="用模擬資料")
    ap.add_argument("--days", type=int, default=120, help="驗證最近幾個交易日")
    ap.add_argument("--no-guard", action="store_true", help="關閉隔日風險過濾以做對照")
    ap.add_argument("--compare", action="store_true", help="同時跑開／關兩種，直接比較")
    args = ap.parse_args()

    hist_map = load_history(args.demo)
    if args.compare:
        report(run(hist_map, args.days, False))
        report(run(hist_map, args.days, True))
    else:
        report(run(hist_map, args.days, not args.no_guard))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
