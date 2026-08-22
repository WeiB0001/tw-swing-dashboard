# -*- coding: utf-8 -*-
"""
indicators.py — 技術指標計算（純函式，不碰網路）

設計重點：**先一次把整段歷史的所有欄位算出來（compute_frame），
再從任一天取出特徵（features_at）。**
這樣回測時可以在同一支股票上逐日取值，不必每天重算整個 rolling，
否則 150 檔 × 250 天會慢到不能用。

原有指標全部保留（RSI、MA5/10/20、20 日均量、近 5/20 日高低點），
新增：MA60、ATR(14)、MACD、近 60 日高低、K 棒結構（上下影線、收盤位置）、
      量價配合、突破偵測、壓力與支撐推估。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as C


# ---------------------------------------------------------------------------
# 基礎指標
# ---------------------------------------------------------------------------
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI，跟券商看盤軟體一致（用 SMA 會有落差）。"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(100.0).where(avg_loss.notna(), np.nan)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder ATR：衡量這檔股票「平常一天會動多少」，用來推估目標與停損距離。"""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    """回傳 (macd, signal, histogram)。histogram 轉正／收斂代表動能改善。"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


# ---------------------------------------------------------------------------
# 一次算完整段歷史
# ---------------------------------------------------------------------------
def compute_frame(hist: pd.DataFrame) -> pd.DataFrame:
    """
    輸入日線 DataFrame（open/high/low/close/volume），
    回傳加上所有特徵欄位的 DataFrame。
    """
    df = hist.copy()
    c, h, l, o, v = df["close"], df["high"], df["low"], df["open"], df["volume"]

    # --- 動能 ---
    df["rsi"] = rsi(c, C.RSI_PERIOD)
    df["rsi_prev"] = df["rsi"].shift(1)
    df["rsi_min5"] = df["rsi"].shift(1).rolling(5).min()      # 前 5 日的 RSI 最低點

    # --- 均線（MA60 為新增） ---
    df["ma5"] = c.rolling(C.MA_SHORT).mean()
    df["ma10"] = c.rolling(C.MA_MID).mean()
    df["ma20"] = c.rolling(C.MA_LONG).mean()
    df["ma60"] = c.rolling(C.MA_TREND).mean()
    df["ma5_prev"] = df["ma5"].shift(1)
    df["ma10_prev"] = df["ma10"].shift(1)

    # 斜率：均線最近有沒有在往上
    df["ma5_slope"] = (df["ma5"] / df["ma5"].shift(2) - 1) * 100
    df["ma10_slope"] = (df["ma10"] / df["ma10"].shift(3) - 1) * 100
    df["ma20_slope"] = (df["ma20"] / df["ma20"].shift(5) - 1) * 100
    df["ma60_slope"] = (df["ma60"] / df["ma60"].shift(10) - 1) * 100

    # 乖離率：離均線多遠（正值＝在均線之上）
    df["bias5"] = (c / df["ma5"] - 1) * 100
    df["bias10"] = (c / df["ma10"] - 1) * 100
    df["bias20"] = (c / df["ma20"] - 1) * 100

    # --- 量能 ---
    df["vol_ma20"] = v.rolling(C.VOL_MA_PERIOD).mean()
    df["vol_ratio"] = v / df["vol_ma20"]
    up_day = c.diff() > 0
    # 量價配合：近 20 日「上漲日均量」vs「下跌日均量」，> 1 代表買盤比賣盤積極
    df["vol_up_ma"] = v.where(up_day).rolling(20, min_periods=4).mean()
    df["vol_dn_ma"] = v.where(~up_day).rolling(20, min_periods=4).mean()
    df["vol_ud_ratio"] = df["vol_up_ma"] / df["vol_dn_ma"]

    # --- 價格區間 ---
    df["high5"] = h.rolling(C.RANGE_SHORT).max()
    df["low5"] = l.rolling(C.RANGE_SHORT).min()
    df["high20"] = h.rolling(C.RANGE_LONG).max()
    df["low20"] = l.rolling(C.RANGE_LONG).min()
    df["high60"] = h.rolling(C.RANGE_TREND).max()
    df["low60"] = l.rolling(C.RANGE_TREND).min()
    df["low10"] = l.rolling(10).min()
    # 「不含今天」的前高／前低，用來判斷突破與破底
    df["high20_prev"] = h.shift(1).rolling(C.RANGE_LONG).max()
    df["low20_prev"] = l.shift(1).rolling(C.RANGE_LONG).min()
    df["low5_prev"] = l.shift(1).rolling(C.RANGE_SHORT).min()

    rng20 = (df["high20"] - df["low20"]).replace(0, np.nan)
    df["pos20"] = ((c - df["low20"]) / rng20).clip(0, 1)      # 0＝貼低點，1＝貼高點
    df["pct_above_low20"] = (c / df["low20"] - 1) * 100
    df["pct_below_high20"] = (df["high20"] / c - 1) * 100
    df["pct_above_low5"] = (c / df["low5"] - 1) * 100
    df["pct_below_high5"] = (df["high5"] / c - 1) * 100
    df["pct_below_high60"] = (df["high60"] / c - 1) * 100
    df["pct_above_low60"] = (c / df["low60"] - 1) * 100

    # --- K 棒結構：分辨「好的爆量」跟「危險的爆量」 ---
    bar = (h - l).replace(0, np.nan)
    df["close_pos_bar"] = ((c - l) / bar).clip(0, 1)          # 收盤在當日振幅的哪個位置
    df["upper_shadow"] = ((h - np.maximum(o, c)) / bar).clip(0, 1)
    df["lower_shadow"] = ((np.minimum(o, c) - l) / bar).clip(0, 1)
    df["is_up_day"] = (c > c.shift(1)).astype(float)
    df["ret1"] = (c / c.shift(1) - 1) * 100
    df["ret5"] = (c / c.shift(5) - 1) * 100
    df["ret10"] = (c / c.shift(10) - 1) * 100

    # 連續上漲天數（追高風險的原料之一）
    up_int = (c.diff() > 0).astype(int)
    grp = (up_int != up_int.shift()).cumsum()
    df["up_streak"] = up_int * (up_int.groupby(grp).cumcount() + 1)

    # --- ATR / MACD ---
    df["atr"] = atr(h, l, c, C.ATR_PERIOD)
    df["atr_pct"] = df["atr"] / c * 100
    line, sig, hist_ = macd(c, C.MACD_FAST, C.MACD_SLOW, C.MACD_SIGNAL)
    df["macd"], df["macd_signal"], df["macd_hist"] = line, sig, hist_
    df["macd_hist_prev"] = hist_.shift(1)
    df["macd_hist_prev3"] = hist_.shift(3)

    # --- 突破 / 破底 ---
    df["is_breakout"] = (c > df["high20_prev"]).astype(float)          # 今日收盤突破前 20 日高
    df["is_new_low"] = (l <= df["low20"] + 1e-9).astype(float)         # 今日創 20 日新低
    df["held_low5"] = (l > df["low5_prev"]).astype(float)              # 近期低點沒有再被跌破

    # --- 壓力與支撐推估（向量化） ---
    # 壓力：只有「距離現價夠遠」的高點才算真壓力，貼在現價上方 0.5% 的不算，
    #       否則突破股會被算成幾乎沒有上漲空間。
    inf = np.inf
    gap = 1 + C.RES_MIN_GAP_PCT / 100
    r1 = np.where(df["high20"] > c * gap, df["high20"], inf)
    r2 = np.where(df["high60"] > c * gap, df["high60"], inf)
    resistance = np.minimum(r1, r2)
    atr_target = c + C.ATR_TARGET_MULT * df["atr"]
    # 價格已在所有壓力之上（突破／創新高）→ 用 ATR 投射目標
    resistance = np.where(np.isinf(resistance), atr_target, resistance)
    # 剛突破前高的，目標至少要有一個 ATR 投射的幅度
    resistance = np.where(df["is_breakout"] > 0, np.maximum(resistance, atr_target), resistance)
    df["target"] = resistance

    neg = -np.inf
    s_candidates = [
        np.where(df["ma20"] < c * 0.995, df["ma20"], neg),
        np.where(df["low20"] < c * 0.995, df["low20"], neg),
        np.where(df["low10"] < c * 0.995, df["low10"], neg),
        np.where((c - C.ATR_STOP_MULT * df["atr"]) < c * 0.995,
                 c - C.ATR_STOP_MULT * df["atr"], neg),
    ]
    support = np.maximum.reduce(s_candidates)
    support = np.where(np.isneginf(support), c - 1.5 * df["atr"], support)
    # 支撐不能離現價太近：停損設在日常波動範圍內只會被雜訊掃出場，
    # 所以至少拉開 SUP_MIN_ATR 個 ATR，避免高估風險報酬比。
    support = np.minimum(support, c - C.SUP_MIN_ATR * df["atr"])
    df["support"] = support

    df["upside_pct"] = (df["target"] / c - 1) * 100
    df["downside_pct"] = (1 - df["support"] / c) * 100
    df["rr_ratio"] = df["upside_pct"] / df["downside_pct"].replace(0, np.nan)

    return df


# ---------------------------------------------------------------------------
# 從某一天取出特徵
# ---------------------------------------------------------------------------
# 這些欄位有 NaN 就代表暖身期不足，該日不評分
_REQUIRED = [
    "rsi", "ma5", "ma10", "ma20", "ma60", "vol_ma20", "atr",
    "high20", "low20", "high60", "low60", "pos20", "target", "support",
]

_FIELDS = [
    # 動能
    "rsi", "rsi_prev", "rsi_min5",
    # 均線與乖離
    "ma5", "ma10", "ma20", "ma60", "ma5_prev", "ma10_prev",
    "ma5_slope", "ma10_slope", "ma20_slope", "ma60_slope",
    "bias5", "bias10", "bias20",
    # 量能
    "vol_ma20", "vol_ratio", "vol_ud_ratio",
    # 區間
    "high5", "low5", "high20", "low20", "high60", "low60", "low10",
    "high20_prev", "low5_prev", "pos20",
    "pct_above_low20", "pct_below_high20", "pct_above_low5", "pct_below_high5",
    "pct_below_high60", "pct_above_low60",
    # K 棒
    "close_pos_bar", "upper_shadow", "lower_shadow", "is_up_day",
    "ret1", "ret5", "ret10", "up_streak",
    # ATR / MACD
    "atr", "atr_pct", "macd", "macd_signal", "macd_hist",
    "macd_hist_prev", "macd_hist_prev3",
    # 型態
    "is_breakout", "is_new_low", "held_low5",
    # 風險報酬
    "target", "support", "upside_pct", "downside_pct", "rr_ratio",
]


def features_at(df: pd.DataFrame, i: int = -1) -> dict | None:
    """從已算好特徵的 DataFrame 取出第 i 天的特徵 dict；資料不足回 None。"""
    if df is None or len(df) < C.MIN_BARS:
        return None
    idx = len(df) + i if i < 0 else i
    if idx < C.MIN_BARS - 1 or idx >= len(df):
        return None

    row = df.iloc[idx]
    if any(pd.isna(row[k]) for k in _REQUIRED):
        return None

    f = {}
    for k in _FIELDS:
        val = row.get(k, np.nan)
        f[k] = float(val) if not pd.isna(val) else 0.0

    f["close"] = float(row["close"])
    f["open"] = float(row["open"])
    f["high"] = float(row["high"])
    f["low"] = float(row["low"])
    f["volume"] = float(row["volume"])
    prev_close = float(df["close"].iloc[idx - 1])
    f["prev_close"] = prev_close
    f["chg_pct"] = (f["close"] / prev_close - 1) * 100 if prev_close else 0.0
    f["date"] = str(df.index[idx])[:10]
    return f


def compute_features(hist: pd.DataFrame) -> dict | None:
    """便利函式：算完整段歷史後回傳最新一日的特徵。"""
    if hist is None or len(hist) < C.MIN_BARS:
        return None
    return features_at(compute_frame(hist), -1)
