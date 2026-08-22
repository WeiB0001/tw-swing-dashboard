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

import os
import random
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

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


def _scannable(code: str) -> bool:
    """
    判斷這個代號要不要納入掃描。
    - 上市普通股：4 碼純數字，且開頭不是 00（00 開頭的 4 碼是 ETF，例如 0050）
    - ETF：00 開頭、4～6 碼。槓桿（L）、反向（R）、債券（B）依設定排除，
           因為這類商品的技術訊號跟一般股票不是同一回事。
    - 其餘（權證、特別股、存託憑證等）一律排除
    """
    if not code:
        return False
    if code.startswith("00"):
        if not C.INCLUDE_ETF:
            return False
        if not (4 <= len(code) <= 6):
            return False
        if not C.ETF_ALLOW_LEVERAGED and code[-1].upper() in ("L", "R", "B"):
            return False
        return code[:4].isdigit()
    return len(code) == 4 and code.isdigit()


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
        if not _scannable(code):
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
    n_etf = sum(1 for c in df["code"] if c.startswith("00")) if len(df) else 0
    log.info("證交所快照：%d 檔（其中 ETF %d 檔）", len(df), n_etf)
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


def fetch_stock_info() -> pd.DataFrame | None:
    """
    上市＋上櫃的股票基本資料（代號、名稱、產業分類、市場別）。
    來源 FinMind TaiwanStockInfo，快取在 data/stock_info.csv。

    產業分類幾乎不會變，所以預設 7 天才更新一次，不會每天多打 API。
    抓不到就用舊快取；連快取都沒有回 None，呼叫端會退回手動清單。
    """
    path = ROOT / C.STOCK_INFO_CSV
    if path.exists():
        try:
            age = (datetime.now().timestamp() - path.stat().st_mtime) / 86400
            cached = pd.read_csv(path, dtype={"stock_id": str})
            if age < C.STOCK_INFO_MAX_AGE_DAYS and len(cached):
                log.info("股票基本資料使用快取（%d 檔，%.1f 天前更新）", len(cached), age)
                return cached
        except Exception:
            cached = None
    else:
        cached = None

    rows = _finmind_get({"dataset": "TaiwanStockInfo"})
    if not rows:
        if cached is not None and len(cached):
            log.warning("股票基本資料取得失敗，改用既有快取")
            return cached
        log.warning("股票基本資料取得失敗且無快取，將退回手動清單")
        return None

    try:
        df = pd.DataFrame(rows)
        keep = [c for c in ["stock_id", "stock_name", "industry_category", "type"] if c in df.columns]
        df = df[keep].dropna(subset=["stock_id"]).drop_duplicates(subset="stock_id", keep="last")
        df["stock_id"] = df["stock_id"].astype(str)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        log.info("股票基本資料已更新：%d 檔", len(df))
        return df
    except Exception as e:
        log.warning("股票基本資料解析失敗：%s", e)
        return cached


def fetch_twii_history(period: str = "3y") -> pd.DataFrame | None:
    """
    加權指數日線，供大盤狀態（regime）判定與回測分層使用。
    只抓一次、失敗回 None，不讓主流程中斷。
    """
    try:
        import yfinance as yf
        h = yf.Ticker("^TWII").history(period=period, auto_adjust=False)
        if h is None or len(h) < 60:
            return None
        h = h.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
        h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
        return h
    except Exception as e:
        log.warning("加權指數日線取得失敗：%s", e)
        return None


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
    # 股價上限（0 = 不限）。買不起的就不用浪費時間算指標。
    if C.MAX_CLOSE_PRICE and C.MAX_CLOSE_PRICE > 0:
        before = len(liquid)
        liquid = liquid[liquid["close"] <= C.MAX_CLOSE_PRICE]
        log.info("股價上限 %.0f 元，濾掉 %d 檔", C.MAX_CLOSE_PRICE, before - len(liquid))

    top = liquid.sort_values("turnover", ascending=False).head(C.TOP_N_BY_TURNOVER)
    # 白名單（權值股 + 電子科技股）：一定納入，但仍要通過股價與流動性門檻，
    # 否則會掃到根本沒人交易、掛單掛不掉的冷門股。
    watchlist = set(C.CORE_WEIGHTED_STOCKS) | set(C.TECH_STOCKS) | set(C.ETF_STOCKS)
    listed = liquid[liquid["code"].isin(watchlist)]

    uni = pd.concat([top, listed]).drop_duplicates(subset="code")
    uni = uni.sort_values("turnover", ascending=False).head(C.MAX_UNIVERSE)

    n_tech = sum(1 for c in uni["code"] if C.asset_type(c) == "tech")
    n_etf = sum(1 for c in uni["code"] if C.asset_type(c) == "etf")
    log.info("掃描池：%d 檔（電子科技 %d、ETF %d）", len(uni), n_tech, n_etf)
    return uni.reset_index(drop=True)


# ===========================================================================
# 3) 台股歷史日線：FinMind + 本機 CSV 快取
#    build.py（掃描）與 backtest.py（回測、walk-forward）共用同一份快取，
#    不會各自重抓。
# ===========================================================================
ROOT = Path(__file__).resolve().parent.parent
_CACHE_DIR = ROOT / C.HISTORY_CACHE_DIR

_FINMIND_COLS = {"max": "high", "min": "low", "Trading_Volume": "volume"}
_NEEDED = ["open", "high", "low", "close", "volume"]


def _cache_path(code: str) -> Path:
    return _CACHE_DIR / f"{code}.csv"


def load_cache(code: str) -> pd.DataFrame | None:
    """讀本機快取。檔案不存在或壞掉都回 None，不拋例外。"""
    p = _cache_path(code)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        if "date" not in df.columns:
            return None
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "close"])
        for c in _NEEDED:
            if c not in df.columns:
                return None
        df = (df.drop_duplicates(subset="date", keep="last")
                .sort_values("date")
                .set_index("date")[_NEEDED]
                .astype(float))
        return df if len(df) else None
    except Exception as e:
        log.warning("%s 快取讀取失敗，將重新抓取：%s", code, e)
        return None


def save_cache(code: str, df: pd.DataFrame) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        out = df.copy()
        out.index.name = "date"
        out.reset_index().to_csv(_cache_path(code), index=False)
    except Exception as e:
        log.warning("%s 快取寫入失敗：%s", code, e)


def _finmind_get(params: dict) -> list | None:
    """
    呼叫 FinMind。回傳 list（可能是空的）代表成功，回傳 None 代表失敗。
    429 與 5xx 用指數退避重試；沒有 token 也能用免費額度。
    """
    q = dict(params)
    token = os.environ.get("FINMIND_TOKEN", "").strip()
    if token:
        q["token"] = token

    delay = C.FINMIND_BACKOFF_START
    for attempt in range(1, C.FINMIND_MAX_RETRY + 1):
        try:
            r = requests.get(C.FINMIND_API, params=q, timeout=C.HTTP_TIMEOUT)
            if r.status_code == 200:
                try:
                    return r.json().get("data", []) or []
                except ValueError:
                    return None
            if r.status_code == 429 or r.status_code >= 500:
                log.warning("FinMind %s（第 %d 次），%.0f 秒後重試",
                            r.status_code, attempt, delay)
                time.sleep(delay)
                delay = min(delay * 2, C.FINMIND_BACKOFF_MAX)
                continue
            log.warning("FinMind 回應 %s：%s", r.status_code, r.text[:120])
            return None
        except requests.RequestException as e:
            log.warning("FinMind 連線失敗（第 %d 次）：%s", attempt, e)
            time.sleep(delay)
            delay = min(delay * 2, C.FINMIND_BACKOFF_MAX)
    return None


def _finmind_to_frame(rows: list) -> pd.DataFrame | None:
    """FinMind 欄位轉成程式內部格式：max→high、min→low、Trading_Volume→volume。"""
    if not rows:
        return None
    try:
        df = pd.DataFrame(rows).rename(columns=_FINMIND_COLS)
        if "date" not in df.columns:
            return None
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        for c in _NEEDED:
            if c not in df.columns:
                return None
        df = (df.drop_duplicates(subset="date", keep="last")
                .sort_values("date")
                .set_index("date")[_NEEDED]
                .astype(float))
        # 成交量為 0 的通常是停牌，留著會讓均量失真
        df = df[(df["close"] > 0)]
        return df if len(df) else None
    except Exception as e:
        log.warning("FinMind 資料轉換失敗：%s", e)
        return None


def _last_expected_trading_day(now: datetime) -> pd.Timestamp:
    """
    快取要不要更新的判斷基準。收盤（15:00）前就以前一個工作日為準，
    免得每次都為了「今天還沒收盤」而多打一次 API。
    """
    d = now.date()
    if now.hour < 15:
        d = d - timedelta(days=1)
    ts = pd.Timestamp(d)
    while ts.weekday() >= 5:          # 週六日往前退
        ts -= pd.Timedelta(days=1)
    return ts.normalize()


def fetch_history(codes: list[str]) -> dict[str, pd.DataFrame]:
    """
    取得台股歷史日線（掃描與回測共用的唯一入口）。

    流程：
      - 有快取 → 只抓「快取最後一天之後」的新資料，合併、去重、排序
      - 沒快取 → 抓最近 HISTORY_MONTHS 個月
      - FinMind 失敗 → 用既有快取頂著，不中斷流程
      - FinMind 查無資料 → 記下代號後跳過，不影響其他股票
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(C.TZ)
    expected = _last_expected_trading_day(now)
    full_start = (now.date() - timedelta(days=int(C.HISTORY_MONTHS * 30.5))).strftime("%Y-%m-%d")

    out: dict[str, pd.DataFrame] = {}
    n_cached = n_updated = n_full = 0
    failed, empty, short = [], [], []

    for i, code in enumerate(codes, 1):
        cached = load_cache(code)

        if cached is not None and len(cached) and cached.index[-1] >= expected:
            out[code] = cached                       # 已是最新，完全不打 API
            n_cached += 1
            continue

        if cached is not None and len(cached):
            start = (cached.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            incremental = True
        else:
            start = full_start
            incremental = False

        rows = _finmind_get({"dataset": "TaiwanStockPrice", "data_id": code,
                             "start_date": start})
        time.sleep(random.uniform(C.FINMIND_MIN_INTERVAL, C.FINMIND_MAX_INTERVAL))

        if rows is None:                             # API 掛了 → 用舊快取
            failed.append(code)
            if cached is not None and len(cached):
                out[code] = cached
            continue

        new = _finmind_to_frame(rows)
        if new is None and cached is None:
            empty.append(code)                       # 查無資料，記錄後跳過
            continue

        if new is None:
            merged = cached
        elif cached is None:
            merged = new
        else:
            merged = pd.concat([cached, new])
            merged = (merged[~merged.index.duplicated(keep="last")].sort_index())

        if merged is None or merged.empty:
            empty.append(code)
            continue

        save_cache(code, merged)
        if incremental:
            n_updated += 1
        else:
            n_full += 1

        if len(merged) >= C.MIN_BARS:
            out[code] = merged
        else:
            short.append(code)

        if i % 40 == 0:
            log.info("歷史資料進度 %d/%d", i, len(codes))

    log.info("歷史日線：可用 %d 檔（快取命中 %d、增量更新 %d、首次抓取 %d）",
             len(out), n_cached, n_updated, n_full)
    if failed:
        log.warning("FinMind 取得失敗 %d 檔，已改用既有快取：%s",
                    len(failed), ",".join(failed[:12]))
    if empty:
        log.warning("FinMind 查無資料 %d 檔，已跳過：%s", len(empty), ",".join(empty[:12]))
    if short:
        log.warning("資料長度不足 %d 檔（少於 %d 根），本次不計算：%s",
                    len(short), C.MIN_BARS, ",".join(short[:12]))
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
