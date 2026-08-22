# -*- coding: utf-8 -*-
"""
us_market.py — 美股連動，推估隔日台股方向

時序問題必須先講清楚，否則很容易做出偷看未來的假模型：

    台股 8/22 09:00–13:30  →  美股 8/22 盤（台北時間 8/23 04:00 收）  →  台股 8/23

也就是說，**影響台股某一天的，是前一晚的美股**。
但這支程式在台北時間 16:20 執行，今晚的美股還沒開始交易。

所以做法分兩段：
  1. 用「已收盤的前一夜美股」與台股的歷史關係跑迴歸（最小平方法，只用 numpy），
     得到 台股報酬 ≈ a + b×那斯達克 + c×費半
  2. 用「此刻正在交易的美股期貨」（NQ=F、ES=F 幾乎 24 小時交易）
     當作今晚美股方向的即時代理指標，代入迴歸式推估隔日台股

⚠️ 這是統計相關性的外推，不是預測。R² 通常只有 0.2～0.5，
   意思是美股只能解釋台股一部分的波動，剩下的來自台股自己的籌碼與消息面。
   R² 低於 config.US_MIN_R2 時直接不顯示，寧可不講也不要誤導。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config as C

log = logging.getLogger("us_market")


def _daily_returns(ticker: str, period: str = "18mo") -> pd.Series | None:
    """抓日線收盤報酬率（%）。失敗回 None，不讓整個流程掛掉。"""
    try:
        import yfinance as yf
        h = yf.Ticker(ticker).history(period=period, auto_adjust=False)
        if h is None or len(h) < 30:
            return None
        close = h["Close"].dropna()
        close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
        return (close.pct_change() * 100).dropna()
    except Exception as e:
        log.warning("%s 取得失敗：%s", ticker, e)
        return None


def _intraday_change(ticker: str) -> float | None:
    """
    期貨「此刻」相對前一個結算價的漲跌 %。
    台股收盤時，美股期貨已經走了大半天，是今晚方向最好的免費代理。
    """
    try:
        import yfinance as yf
        h = yf.Ticker(ticker).history(period="5d", auto_adjust=False)
        if h is None or len(h) < 2:
            return None
        return float((h["Close"].iloc[-1] / h["Close"].iloc[-2] - 1) * 100)
    except Exception as e:
        log.warning("%s 期貨取得失敗：%s", ticker, e)
        return None


def _ols(X: np.ndarray, y: np.ndarray):
    """
    最小平方法（不用 sklearn）。回傳 (係數含截距, R², 殘差標準差)。
    X 不含截距欄，這裡自己補。
    """
    n = len(y)
    A = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ beta
    resid = y - pred
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    dof = max(n - A.shape[1], 1)
    sigma = float(np.sqrt(ss_res / dof))
    return beta, r2, sigma


def build_us_snapshot() -> dict | None:
    """
    回傳美股概況 + 隔日台股推估。任何一步失敗都回 None 或省略推估，
    絕不讓儀表板因為美股資料抓不到就整個失敗。
    """
    if not C.US_ENABLED:
        return None

    # --- 1) 昨夜已收盤的美股 ---
    rets = {}
    for tk, label in C.US_TICKERS.items():
        r = _daily_returns(tk)
        if r is not None and len(r):
            rets[tk] = r

    if not rets:
        log.warning("美股資料全部取得失敗，略過此區塊")
        return None

    last_night = []
    for tk, label in C.US_TICKERS.items():
        if tk in rets:
            last_night.append({
                "label": label,
                "chg_pct": round(float(rets[tk].iloc[-1]), 2),
                "date": str(rets[tk].index[-1])[:10],
            })

    out = {
        "last_night": last_night,
        "futures": [],
        "forecast": None,
        "model": None,
    }

    # --- 2) 此刻的美股期貨 ---
    fut = {}
    for tk, label in C.US_FUTURES.items():
        v = _intraday_change(tk)
        if v is not None:
            fut[tk] = v
            out["futures"].append({"label": label, "chg_pct": round(v, 2)})

    # --- 3) 迴歸：台股[t] ~ 那斯達克[t-1] + 費半[t-1] ---
    twii = _daily_returns("^TWII")
    if twii is None or "^IXIC" not in rets or "^SOX" not in rets:
        log.warning("迴歸所需資料不足，只顯示美股漲跌，不做推估")
        return out

    # 對齊：台股某日的解釋變數是「前一個美股交易日」的報酬
    df = pd.DataFrame({
        "twii": twii,
        "ixic": rets["^IXIC"].shift(1),
        "sox": rets["^SOX"].shift(1),
    }).dropna()
    df = df.tail(C.US_REGRESSION_DAYS)
    if len(df) < 60:
        log.warning("迴歸樣本不足（%d 日），不做推估", len(df))
        return out

    X = df[["ixic", "sox"]].to_numpy()
    y = df["twii"].to_numpy()
    beta, r2, sigma = _ols(X, y)

    out["model"] = {
        "samples": int(len(df)),
        "r2": round(float(r2), 3),
        "beta_nasdaq": round(float(beta[1]), 3),
        "beta_sox": round(float(beta[2]), 3),
        "resid_sd": round(float(sigma), 2),
    }

    if r2 < C.US_MIN_R2:
        log.info("美股與台股關聯性偏低（R²=%.3f），不顯示推估值", r2)
        out["forecast"] = {"available": False,
                           "reason": f"美股與台股的統計關聯性偏低（R²={r2:.2f}），不做推估"}
        return out

    # --- 4) 代入期貨當作今晚方向 ---
    nq = fut.get("NQ=F")
    es = fut.get("ES=F")
    if nq is None and es is None:
        out["forecast"] = {"available": False,
                           "reason": "美股期貨資料暫時取不到，無法推估今晚方向"}
        return out

    # 沒有費半期貨，用那斯達克期貨當代理（兩者同向但費半波動較大，
    # 這裡用 1.3 倍近似歷史上的波動比，屬粗估）
    proxy_nq = nq if nq is not None else es
    proxy_sox = proxy_nq * 1.3

    point = float(beta[0] + beta[1] * proxy_nq + beta[2] * proxy_sox)
    out["forecast"] = {
        "available": True,
        "point": round(point, 2),
        "low": round(point - sigma, 2),
        "high": round(point + sigma, 2),
        "basis": f"以那斯達克期貨 {proxy_nq:+.2f}% 代入",
        "r2": round(float(r2), 3),
        "samples": int(len(df)),
    }
    return out
