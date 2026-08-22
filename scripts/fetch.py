# -*- coding: utf-8 -*-
"""
fetch.py — 資料抓取層

資料來源優先順序（全部免費、全部不需要金鑰）：
  1. 證交所 OpenAPI：當日全市場行情（決定掃描池 + 當日收盤/量/漲跌）
  2. yfinance：歷史日線（台股代號加 .TW），用來算技術指標
  3. FinMind：選用備援，只有在環境變數 FINMIND_TOKEN 存在時才啟用

設計原則：任何一層失敗都不會讓整個流程崩潰，會退到下一層或回報空資料。
"""

from __future__ import annotations

import io
import os
import time
import logging
from datetime import datetime, timedelta

import pandas as pd
import requests

import config as C

log = logging.getLogger("fetch")

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; tw-swing-dashboard/1.0)",
    "Accept": "application/json",
}


# ===========================================================================
# 1) 證交所 OpenAPI：當日全市場行情
# ===========================================================================
def _to_float(x):
    """證交所欄位常有 '--'、',' 或空字串，統一轉成 float 或 None。"""
    if x is None:
        return None
    s = str(x).replace(",", "").replace("+", "").strip()
    if s in ("", "--", "---", "X", "N/A"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def fetch_twse_snapshot() -> pd.DataFrame:
    """
    抓當日上市個股行情快照。
    回傳欄位：code, name, open, high, low, close, prev_close, change, chg_pct,
             volume(股), turnover(元)
    失敗時回傳空 DataFrame。
    """
    try:
        r = requests.get(C.TWSE_STOCK_DAY_ALL, headers=_HEADERS, timeout=C.HTTP_TIMEOUT)
        r.raise_for_status()
        raw = r.json()
    except Exception as e:  # 網路/格式問題都在這裡吸收
        log.warning("證交所 STOCK_DAY_ALL 取得失敗：%s", e)
        return pd.DataFrame()

    rows = []
    for it in raw:
        code = str(it.get("Code", "")).strip()
        # 只留 4 碼純數字＝上市普通股（排除權證、ETF 5 碼、特別股等）
        if not (len(code) == 4 and code.isdigit()):
            continue

        close = _to_float(it.get("ClosingPrice"))
        change = _to_float(it.get("Change"))
        volume = _to_float(it.get("TradeVolume"))
        turnover = _to_float(it.get("TradeValue"))
        if close is None or close <= 0 or volume is None:
            continue

        prev_close = close - change if change is not None else None
        chg_pct = (change / prev_close * 100) if (change is not None and prev_close) else None

        rows.append({
            "code": code,
            "name": str(it.get("Name", "")).strip(),
            "open": _to_float(it.get("OpeningPrice")),
            "high": _to_float(it.get("HighestPrice")),
            "low": _to_float(it.get("LowestPrice")),
            "close": close,
            "prev_close": prev_close,
            "change": change,
            "chg_pct": chg_pct,
            "volume": volume,
            "turnover": turnover if turnover else close * volume,
        })

    df = pd.DataFrame(rows)
    log.info("證交所快照：%d 檔上市普通股", len(df))
    return df


def fetch_market_index() -> dict:
    """
    抓加權指數（TAIEX）當日概況。
    先試證交所 OpenAPI 的指數歷史，失敗再用 yfinance 的 ^TWII。
    回傳 dict：{date, close, change, chg_pct, source}；全失敗回 {}。
    """
    # --- 來源 A：證交所 指數歷史（含開高低收） ---
    try:
        r = requests.get(C.TWSE_INDEX_HIST, headers=_HEADERS, timeout=C.HTTP_TIMEOUT)
        r.raise_for_status()
        rows = r.json()
        if rows:
            rows = sorted(rows, key=lambda x: str(x.get("Date", "")))
            last, prev = rows[-1], rows[-2] if len(rows) > 1 else None
            close = _to_float(last.get("ClosingIndex"))
            prev_close = _to_float(prev.get("ClosingIndex")) if prev else None
            if close and prev_close:
                return {
                    "date": _fmt_roc_or_iso(last.get("Date")),
                    "close": close,
                    "change": close - prev_close,
                    "chg_pct": (close - prev_close) / prev_close * 100,
                    "source": "TWSE OpenAPI",
                }
    except Exception as e:
        log.warning("證交所指數取得失敗：%s", e)

    # --- 來源 B：yfinance ^TWII ---
    try:
        import yfinance as yf
        hist = yf.Ticker("^TWII").history(period="1mo", auto_adjust=False)
        if len(hist) >= 2:
            close = float(hist["Close"].iloc[-1])
            prev_close = float(hist["Close"].iloc[-2])
            return {
                "date": hist.index[-1].strftime("%Y-%m-%d"),
                "close": close,
                "change": close - prev_close,
                "chg_pct": (close - prev_close) / prev_close * 100,
                "source": "yfinance ^TWII",
            }
    except Exception as e:
        log.warning("yfinance 指數取得失敗：%s", e)

    return {}


def _fmt_roc_or_iso(d) -> str:
    """證交所日期可能是民國 '1130815' 或西元 '2024-08-15'，統一成 YYYY-MM-DD。"""
    s = str(d or "").strip()
    if "-" in s:
        return s
    if len(s) == 7 and s.isdigit():        # 民國 7 碼
        return f"{int(s[:3]) + 1911}-{s[3:5]}-{s[5:7]}"
    if len(s) == 8 and s.isdigit():        # 西元 8 碼
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return s


# ===========================================================================
# 2) 掃描池：決定今天要算哪些股票
# ===========================================================================
def build_universe(snapshot: pd.DataFrame) -> pd.DataFrame:
    """
    掃描池 = 成交金額前 N 名（且符合流動性門檻）+ 主要權值股白名單。
    回傳 snapshot 的子集合。
    """
    if snapshot.empty:
        return snapshot

    liquid = snapshot[
        (snapshot["close"] >= C.MIN_CLOSE_PRICE)
        & (snapshot["turnover"] >= C.MIN_TURNOVER_TWD)
    ].copy()

    top = liquid.sort_values("turnover", ascending=False).head(C.TOP_N_BY_TURNOVER)
    core = snapshot[snapshot["code"].isin(C.CORE_WEIGHTED_STOCKS)]

    uni = pd.concat([top, core]).drop_duplicates(subset="code")
    uni = uni.sort_values("turnover", ascending=False).head(C.MAX_UNIVERSE)
    log.info("掃描池：%d 檔（成交金額前 %d + 權值股）", len(uni), C.TOP_N_BY_TURNOVER)
    return uni.reset_index(drop=True)


# ===========================================================================
# 3) yfinance：歷史日線
# ===========================================================================
def fetch_history_yf(codes: list[str]) -> dict[str, pd.DataFrame]:
    """
    批次抓歷史日線。分批（chunk）避免被限流，每批失敗會重試。
    回傳 {股票代號: DataFrame(open/high/low/close/volume)}。
    """
    try:
        import yfinance as yf
    except ImportError:
        log.error("找不到 yfinance，請先 pip install yfinance")
        return {}

    out: dict[str, pd.DataFrame] = {}
    chunks = [codes[i:i + C.YF_CHUNK_SIZE] for i in range(0, len(codes), C.YF_CHUNK_SIZE)]

    for idx, chunk in enumerate(chunks, 1):
        tickers = [f"{c}.TW" for c in chunk]
        data = None
        for attempt in range(1, C.YF_RETRY + 1):
            try:
                data = yf.download(
                    tickers=" ".join(tickers),
                    period=C.HISTORY_PERIOD,
                    interval="1d",
                    group_by="ticker",
                    auto_adjust=False,
                    actions=False,
                    threads=True,
                    progress=False,
                )
                break
            except Exception as e:
                log.warning("yfinance 第 %d 批第 %d 次失敗：%s", idx, attempt, e)
                time.sleep(3 * attempt)

        if data is None or len(data) == 0:
            continue

        for code, tk in zip(chunk, tickers):
            try:
                sub = data[tk] if isinstance(data.columns, pd.MultiIndex) else data
                sub = sub.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
                sub = sub.dropna(subset=["close"])
                if len(sub) >= C.MIN_BARS:
                    out[code] = sub
            except Exception:
                continue

        log.info("yfinance 進度 %d/%d，累積 %d 檔", idx, len(chunks), len(out))
        time.sleep(1)   # 對免費服務客氣一點

    return out


# ===========================================================================
# 4) FinMind：選用備援（需要 FINMIND_TOKEN 才會啟用）
# ===========================================================================
def fetch_history_finmind(codes: list[str]) -> dict[str, pd.DataFrame]:
    """
    只有設定了環境變數 FINMIND_TOKEN 才會運作，用來補 yfinance 抓不到的個股。
    沒設 token 就直接回傳空 dict，不會影響主流程。
    """
    token = os.environ.get("FINMIND_TOKEN", "").strip()
    if not token or not codes:
        return {}

    start = (datetime.now(C.TZ) - timedelta(days=400)).strftime("%Y-%m-%d")
    out: dict[str, pd.DataFrame] = {}

    for code in codes:
        try:
            r = requests.get(
                C.FINMIND_API,
                params={
                    "dataset": "TaiwanStockPrice",
                    "data_id": code,
                    "start_date": start,
                    "token": token,
                },
                timeout=C.HTTP_TIMEOUT,
            )
            j = r.json()
            rows = j.get("data", [])
            if not rows:
                continue
            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").rename(columns={
                "open": "open", "max": "high", "min": "low",
                "close": "close", "Trading_Volume": "volume",
            })[["open", "high", "low", "close", "volume"]]
            if len(df) >= C.MIN_BARS:
                out[code] = df
            time.sleep(0.4)     # FinMind 免費版有頻率限制
        except Exception as e:
            log.warning("FinMind %s 失敗：%s", code, e)

    log.info("FinMind 備援補回 %d 檔", len(out))
    return out


# ===========================================================================
# 5) 合併：用證交所當日資料校正歷史最後一根 K 棒
# ===========================================================================
def merge_today_bar(hist: pd.DataFrame, row: pd.Series, trade_date: pd.Timestamp) -> pd.DataFrame:
    """
    yfinance 收盤後有時會延遲 1 天才更新。這裡用證交所的當日資料
    覆寫（或補上）最後一根日 K，確保「今天」的訊號用的是今天的價量。
    """
    hist = hist.copy()
    idx = pd.to_datetime(hist.index)
    if getattr(idx, "tz", None) is not None:   # yfinance 有時回傳帶時區的索引
        idx = idx.tz_localize(None)
    hist.index = idx.normalize()
    today = pd.Timestamp(trade_date).normalize()

    bar = {
        "open": row.get("open") or row["close"],
        "high": row.get("high") or row["close"],
        "low": row.get("low") or row["close"],
        "close": row["close"],
        "volume": row["volume"],
    }
    hist.loc[today] = bar          # 有就覆寫、沒有就新增
    return hist.sort_index()
