# -*- coding: utf-8 -*-
"""
radar.py — 雷達：掃 Extended 股票池，找特殊狀況

**只做標記，不參與任何排名、不改任何評分公式。**
用的全是 indicators 已經算好的欄位，沒有新的模型。

標記種類：
  🚀 突破      收盤突破前 20 日高（同時破 60 日高會另外標記）
  🔥 冷門放量  不在 Core、量能 >= 2 倍、且站上 MA20
  🚀 冷門突破  上面兩個同時成立
  📉 探底      RSI < 30 或貼近近 20 日低點
  ↗ 跌深反彈  低檔且今日收紅、站回 MA5
  🔎 Extended  只要不在 Core 就會有

歷史資料一律走 fetch.fetch_history()，與首頁掃描、回測共用 data/history 快取，
同一檔不會重抓。
"""

from __future__ import annotations

import logging
import math

import pandas as pd

import config as C
import indicators

log = logging.getLogger("radar")


def _f(x, default=None):
    """擋掉 NaN / inf，避免壞資料讓整個流程掛掉。"""
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def scan(candidates: list[dict], hist_map: dict[str, pd.DataFrame],
         core_codes: set[str]) -> list[dict]:
    """
    candidates 來自 universe.build_extended()，hist_map 是共用的歷史快取。
    回傳通過流動性門檻、且附上雷達標記的清單。
    """
    out = []
    skipped_short = skipped_thin = 0

    for c in candidates:
        code = c["symbol"]
        hist = hist_map.get(code)
        if hist is None or len(hist) < C.EXTENDED_MIN_BARS:
            skipped_short += 1
            continue

        try:
            df = indicators.compute_frame(hist)
            f = indicators.features_at(df, -1)
        except Exception:
            continue
        if not f:
            continue

        close = _f(f.get("close"), 0) or 0
        if close < C.EXTENDED_MIN_PRICE:
            continue

        # 20 日平均成交金額（用收盤價 × 20 日均量估算，不另外抓資料）
        vol_ma = _f(f.get("vol_ma20"), 0) or 0
        avg_turnover = close * vol_ma
        if avg_turnover < C.EXTENDED_MIN_AVG_TURNOVER:
            skipped_thin += 1
            continue

        bars = len(df)
        week52 = None
        if bars >= C.WEEK52_MIN_BARS:
            # 資料夠長才算 52 週高低，不足就是 None，不偽造
            w = df.tail(250)
            hi, lo = _f(w["high"].max()), _f(w["low"].min())
            if hi and lo and close > 0:
                week52 = {"high": round(hi, 2), "low": round(lo, 2),
                          "from_high": round((hi / close - 1) * -100, 1),
                          "from_low": round((close / lo - 1) * 100, 1)}

        is_core = code in core_codes
        vol_ratio = _f(f.get("vol_ratio"), 0) or 0
        rsi = _f(f.get("rsi"), 50) or 50
        above_ma20 = close > (_f(f.get("ma20"), 0) or 0)
        breakout20 = (_f(f.get("is_breakout"), 0) or 0) > 0
        breakout60 = close > (_f(f.get("high60"), 0) or 0) * 0.999 and bars >= 60
        near_low = (_f(f.get("pct_above_low20"), 99) or 99) <= C.RADAR_NEAR_LOW_PCT
        quiet = (not is_core) and avg_turnover >= C.RADAR_QUIET_TURNOVER

        tags = []
        if not is_core:
            tags.append({"key": "extended", "icon": "🔎", "label": "Extended"})
        if quiet and vol_ratio >= C.RADAR_VOL_SURGE and above_ma20:
            if breakout20 or breakout60:
                tags.append({"key": "quiet_breakout", "icon": "🚀", "label": "冷門突破"})
            else:
                tags.append({"key": "quiet_volume", "icon": "🔥", "label": "冷門放量"})
        elif breakout20 or breakout60:
            tags.append({"key": "breakout", "icon": "🚀",
                         "label": "突破 60 日高" if breakout60 else "突破 20 日高"})
        if rsi < C.RADAR_OVERSOLD_RSI or near_low:
            tags.append({"key": "oversold", "icon": "📉", "label": "探底"})
        if (near_low or rsi < 40) and (_f(f.get("is_up_day"), 0) or 0) > 0 \
                and close > (_f(f.get("ma5"), 0) or 0):
            tags.append({"key": "rebound", "icon": "↗", "label": "跌深反彈"})

        out.append({
            "symbol": code,
            "name": c.get("name") or "",
            "sector": c.get("sector") or "",
            "kind": c.get("kind"),
            "market": c.get("market"),
            "is_core": is_core,
            "close": round(close, 2),
            "chg_pct": round(_f(f.get("chg_pct"), 0) or 0, 2),
            "rsi": round(rsi, 1),
            "vol_ratio": round(vol_ratio, 2),
            "avg_turnover": int(avg_turnover),
            "ma20": round(_f(f.get("ma20"), 0) or 0, 2),
            "pct_below_high20": round(_f(f.get("pct_below_high20"), 0) or 0, 1),
            "pct_above_low20": round(_f(f.get("pct_above_low20"), 0) or 0, 1),
            "bars": bars,
            "week52": week52,
            "tags": tags,
        })

    log.info("雷達：%d 檔通過（資料不足 %d、成交量過低 %d）",
             len(out), skipped_short, skipped_thin)
    # 有標記的排前面，其餘依成交金額
    out.sort(key=lambda r: (-len([t for t in r["tags"] if t["key"] != "extended"]),
                            -r["avg_turnover"]))
    return out
