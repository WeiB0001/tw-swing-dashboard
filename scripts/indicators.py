# -*- coding: utf-8 -*-
"""
indicators.py — 技術指標計算（純函式，不碰網路，方便單獨測試）

指標一律用「日線」計算：
  - RSI(14)：Wilder 平滑法，跟看盤軟體一致
  - MA5 / MA10 / MA20
  - 量能倍數：當日成交量 / 20 日均量
  - 近 5 日、近 20 日高低點與收盤價的距離
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as C


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder RSI。用 ewm(alpha=1/period) 等價於 Wilder 的遞迴平滑，
    跟券商軟體算出來的數字才會一致（用簡單移動平均會有落差）。
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(100.0).where(avg_loss.notna(), np.nan)


def compute_features(hist: pd.DataFrame) -> dict | None:
    """
    輸入單一個股的日線 DataFrame（需含 open/high/low/close/volume），
    輸出「最新一日」的所有特徵值 dict。資料不足回傳 None。

    這個 dict 就是 scoring.py 的唯一輸入，之後要新增指標，
    在這裡多算一個欄位、再到 scoring.py 用它就好。
    """
    if hist is None or len(hist) < C.MIN_BARS:
        return None

    df = hist.copy()
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]

    df["rsi"] = rsi(close, C.RSI_PERIOD)
    df["ma_s"] = close.rolling(C.MA_SHORT).mean()      # MA5
    df["ma_m"] = close.rolling(C.MA_MID).mean()        # MA10
    df["ma_l"] = close.rolling(C.MA_LONG).mean()       # MA20
    df["vol_ma"] = vol.rolling(C.VOL_MA_PERIOD).mean()

    # 近 N 日高低（不含當日的極端值也一起算進去，跟看盤軟體一致）
    df["low_s"] = low.rolling(C.RANGE_SHORT).min()
    df["high_s"] = high.rolling(C.RANGE_SHORT).max()
    df["low_l"] = low.rolling(C.RANGE_LONG).min()
    df["high_l"] = high.rolling(C.RANGE_LONG).max()

    cur, prev = df.iloc[-1], df.iloc[-2]

    # 必要欄位有 NaN 就代表暖身期不足，不算
    need = ["rsi", "ma_s", "ma_m", "ma_l", "vol_ma", "low_l", "high_l"]
    if any(pd.isna(cur[k]) for k in need):
        return None

    rng = float(cur["high_l"] - cur["low_l"])
    # 收盤在近 20 日區間的相對位置：0 = 貼著低點，1 = 貼著高點
    pos_in_range = float((cur["close"] - cur["low_l"]) / rng) if rng > 0 else 0.5

    vol_ma = float(cur["vol_ma"]) or 1.0
    ma_l_prev5 = float(df["ma_l"].iloc[-6]) if len(df) >= 6 else float(cur["ma_l"])

    return {
        "close": float(cur["close"]),
        "open": float(cur["open"]),
        "high": float(cur["high"]),
        "low": float(cur["low"]),
        "prev_close": float(prev["close"]),
        "chg_pct": float((cur["close"] / prev["close"] - 1) * 100) if prev["close"] else 0.0,

        "rsi": float(cur["rsi"]),
        "rsi_prev": float(prev["rsi"]) if not pd.isna(prev["rsi"]) else float(cur["rsi"]),

        "ma5": float(cur["ma_s"]),
        "ma10": float(cur["ma_m"]),
        "ma20": float(cur["ma_l"]),
        "ma5_prev": float(prev["ma_s"]) if not pd.isna(prev["ma_s"]) else float(cur["ma_s"]),
        "ma10_prev": float(prev["ma_m"]) if not pd.isna(prev["ma_m"]) else float(cur["ma_m"]),
        # MA20 五日斜率（%）：> 0 代表中期趨勢還沒壞
        "ma20_slope_pct": float((cur["ma_l"] / ma_l_prev5 - 1) * 100) if ma_l_prev5 else 0.0,

        "volume": float(cur["volume"]),
        "vol_ma20": vol_ma,
        "vol_ratio": float(cur["volume"] / vol_ma),

        "low5": float(cur["low_s"]),
        "high5": float(cur["high_s"]),
        "low20": float(cur["low_l"]),
        "high20": float(cur["high_l"]),
        "pos_in_range20": pos_in_range,
        # 距離近 20 日低／高點還有幾 %
        "pct_above_low20": float((cur["close"] / cur["low_l"] - 1) * 100) if cur["low_l"] else 0.0,
        "pct_below_high20": float((cur["high_l"] / cur["close"] - 1) * 100) if cur["close"] else 0.0,
        "pct_above_low5": float((cur["close"] / cur["low_s"] - 1) * 100) if cur["low_s"] else 0.0,
        "pct_below_high5": float((cur["high_s"] / cur["close"] - 1) * 100) if cur["close"] else 0.0,
    }
