# -*- coding: utf-8 -*-
"""
autotune.py — 用你的真實台股資料自動調校參數

背景：先前實測「排名前幾名隔日大跌」，而參數（動能權重、停損倍數、目標倍數、
持有天數）沒有辦法靠模擬資料決定——在隨機資料上調出來的值換到真實市場只會更糟。
這支就是把調校交給真實資料。

做法：
  1. 用既有的 data/history 快取（不夠才向 FinMind 補），逐日重放整條排序邏輯
  2. 時間切兩段：前 70% 校準、後 30% 只拿來驗證（out-of-sample）
  3. 逐項座標下降，不做全網格，避免在噪音上過度擬合
  4. 目標：out-of-sample 的 Top5 平均淨報酬（已扣 TRADE_COST_PCT 來回成本），
     打平時比大跌率
  5. **只有 out-of-sample 的 Top5 真的贏過「全體平均」才寫入結果**；
     贏不了就明說這套排序在你的資料上不成立，不寫任何參數

輸出 data/tuned_params.json，config.py 啟動時會自動套用。
刪掉那個檔案就回到預設值。

用法：
    python scripts/autotune.py --days 250
    python scripts/autotune.py --days 250 --dry-run     # 只看結果不寫檔
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

import build as build_mod
import config as C
import indicators
import plan as plan_mod
import scoring

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("autotune")

ROOT = Path(__file__).resolve().parent.parent

# 座標下降的候選值。想試更多就改這裡，但每多一個值就多跑一輪。
GRID = {
    "HOLD_DAYS": [1, 2],
    "PLAN_STOP_MAX_ATR": [1.0, 1.5, 2.0, 2.5],
    "PLAN_T1_ATR": [1.0, 1.2, 1.5, 2.0],
    "RANK_W_MOM": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
}
TUNE_ORDER = ["HOLD_DAYS", "PLAN_STOP_MAX_ATR", "PLAN_T1_ATR", "RANK_W_MOM"]


# ---------------------------------------------------------------------------
def load_history(demo: bool) -> dict[str, pd.DataFrame]:
    if demo:
        import demo_data
        return {c: h for c, _, _, h in demo_data.build_dataset()}
    import fetch
    snap = fetch.fetch_twse_snapshot()
    if snap.empty:
        raise RuntimeError("證交所行情取得失敗，無法調校")
    info = fetch.fetch_stock_info()
    import universe as universe_mod
    uni = universe_mod.build_core(snap, info)
    codes = uni["code"].tolist()
    log.info("調校標的：%d 檔", len(codes))
    return fetch.fetch_history(codes)


def prepare(hist_map: dict, days: int):
    """
    先把不受調校參數影響的東西算完：技術指標、技術分數、未來報酬。
    這些跟 HOLD_DAYS 以外的參數無關，先算好後面每一輪就只要重算計畫與排序。
    """
    frames = {}
    for code, h in hist_map.items():
        if h is None or len(h) < C.MIN_BARS + 6:
            continue
        try:
            df = indicators.compute_frame(h)
            df.index = pd.to_datetime(df.index)
            frames[code] = df
        except Exception:
            continue
    if not frames:
        raise RuntimeError("沒有足夠的歷史資料")

    all_dates = sorted(set().union(*[set(d.index) for d in frames.values()]))
    usable = all_dates[C.MIN_BARS: len(all_dates) - 4]
    if days > 0:
        usable = usable[-days:]

    cache = []          # [(date, [(row, df, pos), ...])]
    for day in usable:
        items = []
        for code, df in frames.items():
            pos = df.index.get_indexer([day])[0]
            if pos < C.MIN_BARS - 1 or pos + 4 >= len(df):
                continue
            f = indicators.features_at(df, pos)
            if not f:
                continue
            res = scoring.score_stock(f)
            if res["score"] < C.WEAK_SCORE:
                continue
            row = build_mod.build_row(code, code, f, res)
            row["prev_low"] = float(df["low"].iloc[pos - 1]) if pos > 0 else None
            items.append({"row": row, "f": f, "res": res, "df": df, "pos": pos})
        if len(items) >= 5:
            cache.append((day, items))
    log.info("可用交易日 %d 天，平均每天 %.0f 檔候選",
             len(cache), np.mean([len(x[1]) for x in cache]) if cache else 0)
    return cache


def net_return(df, pos, hold) -> float | None:
    """t+1 開盤買、持有 hold 天後收盤賣，扣掉來回交易成本。"""
    try:
        entry = float(df["open"].iloc[pos + 1])
        exit_px = float(df["close"].iloc[pos + 1 + hold])
        if entry <= 0:
            return None
        return (exit_px / entry - 1) * 100 - C.TRADE_COST_PCT
    except Exception:
        return None


def replay(cache, params: dict, lo: int, hi: int) -> dict:
    """用指定參數重放 [lo, hi) 這段期間，回傳 Top5 與全體的表現。"""
    for k, v in params.items():
        setattr(C, k, v)
    C.RANK_W_FINAL = 1.0 - C.RANK_W_MOM
    C.PRIMARY_HOLD_DAYS = C.HOLD_DAYS

    top, every = [], []
    for day, items in cache[lo:hi]:
        rows = []
        for it in items:
            r = dict(it["row"])
            r["plan"] = plan_mod.build_plan(it["f"], it["res"])   # 計畫跟參數有關，要重算
            rows.append(r)
        build_mod.add_momentum(rows)
        build_mod.add_final_score(rows)
        ranked = build_mod.sort_by_final(rows)

        idx = {it["row"]["code"]: it for it in items}
        for i, r in enumerate(ranked):
            it = idx.get(r["code"])
            if not it:
                continue
            v = net_return(it["df"], it["pos"], C.HOLD_DAYS)
            if v is None:
                continue
            every.append(v)
            if i < 5:
                top.append(v)

    def pack(a):
        if not a:
            return {"n": 0, "avg": -99, "win": 0, "crash": 100}
        x = np.array(a)
        return {"n": len(x), "avg": round(float(x.mean()), 3),
                "win": round(float((x > 0).mean() * 100), 1),
                "crash": round(float((x <= C.NEXTDAY_CRASH_PCT).mean() * 100), 1)}

    return {"top5": pack(top), "all": pack(every)}


def score_of(r: dict) -> tuple:
    """排序目標：Top5 平均淨報酬優先，打平比大跌率（越低越好）。"""
    return (r["top5"]["avg"], -r["top5"]["crash"])


def main() -> int:
    ap = argparse.ArgumentParser(description="用真實資料自動調校參數")
    ap.add_argument("--days", type=int, default=250)
    ap.add_argument("--demo", action="store_true", help="用模擬資料（只驗證流程）")
    ap.add_argument("--dry-run", action="store_true", help="只顯示結果，不寫入")
    args = ap.parse_args()

    base = {k: getattr(C, k) for k in GRID}
    cache = prepare(load_history(args.demo), args.days)
    if len(cache) < 40:
        log.error("可用交易日只有 %d 天，太少不足以調校", len(cache))
        return 1

    split = int(len(cache) * 0.7)
    log.info("校準期 %d 天，驗證期（out-of-sample）%d 天", split, len(cache) - split)

    cur = dict(base)
    print("\n=== 校準期逐項調校 ===")
    for key in TUNE_ORDER:
        best, best_v = None, None
        for v in GRID[key]:
            trial = dict(cur, **{key: v})
            r = replay(cache, trial, 0, split)
            s = score_of(r)
            mark = ""
            if best is None or s > best:
                best, best_v, mark = s, v, "  ←"
            print(f"  {key:<20}{v:<6} Top5 平均 {r['top5']['avg']:+.3f}%"
                  f"  勝率 {r['top5']['win']:.1f}%  大跌率 {r['top5']['crash']:.1f}%{mark}")
        cur[key] = best_v
        print(f"  → {key} 選定 {best_v}\n")

    print("=== 驗證期（out-of-sample，沒有參與調校）===")
    oos_base = replay(cache, base, split, len(cache))
    oos_tuned = replay(cache, cur, split, len(cache))
    for name, r in (("預設參數", oos_base), ("調校後", oos_tuned)):
        t, a = r["top5"], r["all"]
        print(f"  {name:<8} Top5 平均 {t['avg']:+.3f}%  勝率 {t['win']:.1f}%  "
              f"大跌率 {t['crash']:.1f}%  ｜全體平均 {a['avg']:+.3f}%")

    t, a = oos_tuned["top5"], oos_tuned["all"]
    beat_baseline = t["avg"] > a["avg"]
    improved = t["avg"] > oos_base["top5"]["avg"]

    print()
    if not beat_baseline:
        print("❌ 調校後的 Top5 仍然輸給「全體平均」——這套排序在你的資料上不成立。")
        print("   沒有寫入任何參數。建議把首頁排名只當觀察清單，不要照名次下單。")
        return 0
    if not improved:
        print("⚠️ 調校後沒有比預設參數更好，維持預設，不寫入。")
        return 0

    for k, v in cur.items():
        print(f"   {k} = {v}   （原本 {base[k]}）")
    if args.dry_run:
        print("\n--dry-run，未寫入檔案。")
        return 0

    out = {
        "generated_at": datetime.now(C.TZ).strftime("%Y-%m-%d %H:%M"),
        "days": args.days, "calibrate_days": split,
        "oos_days": len(cache) - split,
        "oos_top5_avg": t["avg"], "oos_all_avg": a["avg"],
        "oos_top5_win": t["win"], "oos_top5_crash": t["crash"],
        "params": cur,
    }
    path = ROOT / "data/tuned_params.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ 已寫入 {path.relative_to(ROOT)}，下次 build 會自動套用。")
    print("   要退回預設值，把這個檔案刪掉即可。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
