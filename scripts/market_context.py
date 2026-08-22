# -*- coding: utf-8 -*-
"""
market_context.py — 美股/全球市場 context

用途：
- live：抓取最新已完成的美股交易日資料，產生台股 ranking context。
- backtest/train：把每個台股日期對應到「當時已知」的美股資訊，避免 look-ahead。

時間規則：
1) after_tw_close：台股收盤後立即產生排名，只能使用「前一個已完成的美股交易日」。
2) preopen：隔日台股開盤前刷新，可使用最新一個已完成的美股交易日。

資料來源：yfinance，無需 API key。
"""

from __future__ import annotations
from datetime import datetime, time
import numpy as np
import pandas as pd
import yfinance as yf

US_TICKERS = {
    "spx": "^GSPC",
    "nasdaq": "^IXIC",
    "sox": "^SOX",
    "vix": "^VIX",
    "dxy": "DX-Y.NYB",
    "us10y": "^TNX",
    "tsm": "TSM",
    "nvda": "NVDA",
}

TECH_SECTORS = {
    "半導體","IC設計","封測","記憶體","PCB載板","被動元件",
    "電腦週邊","工業電腦","設備儀器","散熱機構","網通","光電面板",
    "光學","電子零組件","電子通路","電信"
}

FEATURE_COLUMNS = [
    "spx_ret1","spx_ret5","nasdaq_ret1","nasdaq_ret5","sox_ret1","sox_ret5",
    "vix","vix_chg","dxy_ret1","us10y_chg","tsm_ret1","nvda_ret1",
    "risk_on","tech_context"
]

def fetch_us_history(period="3y") -> dict[str, pd.DataFrame]:
    out = {}
    for key, ticker in US_TICKERS.items():
        try:
            df = yf.download(ticker, period=period, auto_adjust=False, progress=False, threads=False)
            if df is None or df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [str(c).lower() for c in df.columns]
            if "close" not in df:
                continue
            x = pd.DataFrame(index=pd.to_datetime(df.index).tz_localize(None))
            x["close"] = pd.to_numeric(df["close"], errors="coerce")
            x["ret1"] = x["close"].pct_change()*100
            x["ret5"] = x["close"].pct_change(5)*100
            x["chg"] = x["close"].diff()
            out[key] = x.dropna(subset=["close"])
        except Exception:
            continue
    return out

def _latest_row_before(df: pd.DataFrame, cutoff_date: pd.Timestamp):
    idx = df.index[df.index <= cutoff_date]
    if len(idx) == 0:
        return None
    return df.loc[idx[-1]]

def context_for_tw_date(us_map: dict[str,pd.DataFrame], tw_date, mode="after_tw_close") -> dict:
    """
    tw_date = 台股訊號日。
    after_tw_close：只使用美股日期 <= tw_date - 1 天。
    preopen：可使用 <= tw_date 的美股資料（適合隔日台股開盤前執行）。
    """
    d = pd.Timestamp(tw_date).normalize()
    cutoff = d - pd.Timedelta(days=1) if mode == "after_tw_close" else d

    def val(key, col, default=0.0):
        df = us_map.get(key)
        if df is None or df.empty:
            return default
        row = _latest_row_before(df, cutoff)
        if row is None:
            return default
        x = row.get(col, default)
        try:
            return float(x) if np.isfinite(float(x)) else default
        except Exception:
            return default

    c = {
        "spx_ret1": val("spx","ret1"),
        "spx_ret5": val("spx","ret5"),
        "nasdaq_ret1": val("nasdaq","ret1"),
        "nasdaq_ret5": val("nasdaq","ret5"),
        "sox_ret1": val("sox","ret1"),
        "sox_ret5": val("sox","ret5"),
        "vix": val("vix","close",20.0),
        "vix_chg": val("vix","chg"),
        "dxy_ret1": val("dxy","ret1"),
        "us10y_chg": val("us10y","chg"),
        "tsm_ret1": val("tsm","ret1"),
        "nvda_ret1": val("nvda","ret1"),
    }

    # 純 context summary，不是最終模型分數
    risk_on = (
        0.22*c["spx_ret1"] + 0.22*c["nasdaq_ret1"] + 0.28*c["sox_ret1"]
        + 0.12*c["tsm_ret1"] + 0.08*c["nvda_ret1"]
        - 0.05*max(c["vix_chg"], 0) - 0.03*max(c["dxy_ret1"], 0)
    )
    c["risk_on"] = float(np.clip(risk_on, -5, 5))
    c["tech_context"] = float(np.clip(
        0.40*c["sox_ret1"] + 0.25*c["tsm_ret1"] + 0.20*c["nvda_ret1"] + 0.15*c["nasdaq_ret1"],
        -8, 8
    ))
    return c

def sector_adjusted_context(base: dict, sector: str) -> dict:
    out = dict(base)
    is_tech = sector in TECH_SECTORS
    out["sector_x_sox"] = out["sox_ret1"] if is_tech else 0.0
    out["sector_x_tsm"] = out["tsm_ret1"] if sector in {"半導體","封測","設備儀器","PCB載板"} else 0.0
    out["sector_x_nvda"] = out["nvda_ret1"] if sector in {"電腦週邊","散熱機構","PCB載板","網通","半導體","IC設計"} else 0.0
    out["sector_is_tech"] = 1.0 if is_tech else 0.0
    return out
