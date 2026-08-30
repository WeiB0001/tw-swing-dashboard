# -*- coding: utf-8 -*-
"""
nextday.py — 隔日續強／暴跌的歷史統計

回應的問題：**排名前幾名隔日下跌的機率偏高，甚至出現暴跌。**

做法是把「排名當下的條件」拿去歷史裡查：同樣條件的股票，隔日
  - 上漲的機率有多高
  - 出現大跌（跌幅超過門檻）的機率有多高
  - 進入當日漲幅前段的機率有多高

然後在排序時，對**歷史暴跌率明顯偏高**的標的扣分。這不會改動 scoring、EV、
backtest、walk-forward，只是在最終排序上多一層風險過濾。

禁止 look-ahead：條件取自第 t 日收盤，報酬一律 t+1 開盤進場、t+1 收盤結算
（跟 backtest.py 與交易計畫的假設一致）。

樣本不足就往上退一層，再不足就標「樣本不足」，不會生假機率。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config as C
import indicators

log = logging.getLogger("nextday")


def _f(x, d=0.0):
    try:
        v = float(x)
        return v if np.isfinite(v) else d
    except Exception:
        return d


# ---------------------------------------------------------------------------
# 分級：排序時與回測時必須用同一套，否則查表會對不上
# ---------------------------------------------------------------------------
def tier_of(close, ma5, ma20, vol, chg, bias, prev_low, high20_prev,
            macd_h, macd_hp) -> int:
    """
    簡化版的動能分層（build.add_momentum 的歷史代理版本）。
    用的欄位跟正式版一致，只是省略了需要 plan 才能算的項目。
    """
    hard = (chg < C.MOM_DROP_BAD or (ma5 > 0 and close < ma5)
            or bias > C.MOM_BIAS_MAX or (vol >= C.MOM_BLOWOFF and chg < 0)
            or (prev_low and close < prev_low))
    if hard:
        return 2
    confirmed = (high20_prev > 0 and close >= high20_prev) or (ma5 > 0 and close >= ma5)
    if confirmed and vol >= C.MOM_VOL_MIN:
        return 0
    return 1


def chg_b(x):
    if x >= 6:
        return "漲6+"
    if x >= 3:
        return "漲3-6"
    if x >= 0:
        return "漲0-3"
    return "跌"


def vol_b(x):
    if x >= 2.5:
        return "爆量"
    if x >= 1.5:
        return "放量"
    if x >= 1.0:
        return "量增"
    return "量縮"


def bias_b(x):
    if x >= 8:
        return "乖離大"
    if x >= 3:
        return "乖離中"
    return "貼近均線"


def _pack(rets: list[float], hits: list[int] | None = None) -> dict:
    a = np.array(rets, dtype=float)
    n = len(a)
    wins = int((a > 0).sum())
    crashes = int((a <= C.NEXTDAY_CRASH_PCT).sum())
    hit = int(sum(hits)) if hits else 0
    return {
        # 隔日進入當日漲幅前段（官方漲幅排行的前 20%）的比例。
        # 隨機挑的話大約就是 20%，明顯高於 20% 才算真的有預測力。
        "top_hit_rate": round((hit + 4) / (n + 20) * 100, 1) if hits else None,
        "samples": n,
        "win_rate": round(float(wins / n * 100), 1),
        # 平滑後的機率，避免小樣本出現 0% 或 100% 這種假數字
        "calibrated_win_rate": round((wins + C.SMOOTH_WINS) / (n + C.SMOOTH_N) * 100, 1),
        "crash_rate": round(float(crashes / n * 100), 1),
        "calibrated_crash_rate": round((crashes + 2) / (n + 20) * 100, 1),
        "avg": round(float(a.mean()), 2),
        "median": round(float(np.median(a)), 2),
        "p10": round(float(np.percentile(a, 10)), 2),   # 最差的一成長什麼樣
        "worst": round(float(a.min()), 2),
    }


# ---------------------------------------------------------------------------
# 歷史取樣
# ---------------------------------------------------------------------------
def build_stats(hist_map: dict[str, pd.DataFrame]) -> dict:
    """
    掃過快取裡所有個股的所有交易日，記錄「當日條件 → 隔日報酬」。
    隔日報酬 = t+1 開盤買、t+1 收盤賣，跟實際可執行的做法一致。
    """
    rows = []
    for code, hist in hist_map.items():
        if hist is None or len(hist) < C.MIN_BARS + 3:
            continue
        try:
            df = indicators.compute_frame(hist)
        except Exception:
            continue

        n = len(df)
        close = df["close"].to_numpy()
        opn = df["open"].to_numpy()
        low = df["low"].to_numpy()
        get = lambda k: df[k].to_numpy() if k in df.columns else np.zeros(n)
        ma5, ma20 = get("ma5"), get("ma20")
        vol, chg, bias = get("vol_ratio"), get("ret1"), get("bias20")
        h20p = get("high20_prev")
        mh, mhp = get("macd_hist"), get("macd_hist_prev")
        vol_ma = get("vol_ma20")

        for i in range(C.MIN_BARS, n - 1):
            if not np.isfinite(close[i]) or close[i] <= 0:
                continue
            if close[i] * vol_ma[i] < C.NEXTDAY_MIN_TURNOVER:
                continue                      # 太冷門的不列入統計
            entry = opn[i + 1]
            if not np.isfinite(entry) or entry <= 0:
                continue

            t = tier_of(close[i], ma5[i], ma20[i], vol[i], chg[i], bias[i],
                        low[i - 1] if i > 0 else 0, h20p[i], mh[i], mhp[i])
            rows.append({
                "date": str(df.index[i])[:10],
                "tier": t,
                "chg": chg_b(chg[i]),
                "vol": vol_b(vol[i]),
                "bias": bias_b(bias[i]),
                "r": (close[i + 1] / entry - 1) * 100,
                # 隔日的當日漲跌（收盤對收盤），用來跟同一天其他股票比排名
                "next_chg": (close[i + 1] / close[i] - 1) * 100,
            })

    if not rows:
        return {"total": 0}

    # 先算出每一天「漲幅前 20%」的門檻，標記哪些樣本隔日進了漲幅前段
    by_date = {}
    for r in rows:
        by_date.setdefault(r["date"], []).append(r["next_chg"])
    cuts = {}
    for d0, vals in by_date.items():
        vals = sorted(vals, reverse=True)
        cuts[d0] = vals[max(0, int(len(vals) * 0.2) - 1)] if vals else 0.0
    for r in rows:
        r["hit"] = 1 if r["next_chg"] >= cuts.get(r["date"], 0.0) else 0

    def group(keyfn):
        d = {}
        for r in rows:
            d.setdefault(keyfn(r), []).append(r)
        return {k: _pack([x["r"] for x in v], [x["hit"] for x in v])
                for k, v in d.items() if len(v) >= 5}

    stats = {
        "total": len(rows),
        "full": group(lambda r: "%d|%s|%s|%s" % (r["tier"], r["chg"], r["vol"], r["bias"])),
        "tier_chg": group(lambda r: "%d|%s" % (r["tier"], r["chg"])),
        "tier": group(lambda r: str(r["tier"])),
        "overall": _pack([r["r"] for r in rows], [r["hit"] for r in rows]),
        "entry_rule": "第 t 日收盤判定條件，t+1 開盤買、t+1 收盤賣",
    }
    log.info("隔日統計：%d 筆樣本，整體上漲率 %.1f%%、大跌率 %.1f%%",
             stats["total"], stats["overall"]["win_rate"], stats["overall"]["crash_rate"])
    return stats


def lookup(stats: dict, tier: int, chg: float, vol: float, bias: float):
    """三層 fallback：完整條件 → 分層＋漲幅 → 分層 → 樣本不足。"""
    if not stats or not stats.get("total"):
        return None, "無統計資料"
    keys = [
        ("%d|%s|%s|%s" % (tier, chg_b(chg), vol_b(vol), bias_b(bias)), "full", "完整條件"),
        ("%d|%s" % (tier, chg_b(chg)), "tier_chg", "分層＋漲幅"),
        (str(tier), "tier", "分層"),
    ]
    for k, bucket, label in keys:
        d = (stats.get(bucket) or {}).get(k)
        if d and d["samples"] >= C.NEXTDAY_MIN_SAMPLES:
            return d, label
    return None, "樣本不足"


# ---------------------------------------------------------------------------
# 排序用的風險調整
# ---------------------------------------------------------------------------
def attach(rows: list[dict], stats: dict) -> None:
    """
    把隔日統計掛到每一列，並算出風險扣分。

    扣分只看**暴跌率高出基準多少**——不是「上漲機率不夠高就扣」，
    因為那會退化成追高。我們要擋的是隔日崩掉的那種。
    """
    if not stats or not stats.get("total"):
        for r in rows:
            r["nextday"] = None
            r["nextday_source"] = "無統計資料"
            r["crash_penalty"] = 0.0
        return

    base = (stats.get("overall") or {}).get("crash_rate", 0) or 0
    for r in rows:
        tier = r.get("momentum_tier", 1)
        d, src = lookup(stats, tier, _f(r.get("chg_pct")), _f(r.get("vol_ratio")),
                        _f(r.get("bias20")))
        r["nextday"] = d
        r["nextday_source"] = src
        if not d:
            r["crash_penalty"] = 0.0
            continue
        # 1) 暴跌率每高出基準 1 個百分點，扣 NEXTDAY_CRASH_WEIGHT 分
        excess = max(0.0, d["calibrated_crash_rate"] - base)
        crash_pen = min(C.NEXTDAY_MAX_PENALTY, excess * C.NEXTDAY_CRASH_WEIGHT)

        # 2) 歷史隔日平均報酬直接加減分。
        #    實測發現排名前段的隔日表現比全體差——強勢股隔日常跳空高開後收斂，
        #    只擋「暴跌」擋不到這種普遍性的小虧，得把平均報酬也算進排序。
        avg_adj = max(-C.NEXTDAY_AVG_CAP,
                      min(C.NEXTDAY_AVG_CAP, d["avg"] * C.NEXTDAY_AVG_WEIGHT))

        # 3) 隔日進入漲幅前段的命中率。這是「有沒有提前猜中官方漲幅排行」，
        #    高於隨機基準（20%）的部分才加分，低於就扣。
        hit = d.get("top_hit_rate")
        hit_adj = 0.0
        if hit is not None:
            hit_adj = max(-C.NEXTDAY_HIT_CAP,
                          min(C.NEXTDAY_HIT_CAP, (hit - 20.0) * C.NEXTDAY_HIT_WEIGHT))

        r["crash_penalty"] = round(crash_pen - avg_adj - hit_adj, 1)
        r["nextday_avg_adj"] = round(avg_adj, 1)
        r["nextday_hit_adj"] = round(hit_adj, 1)
