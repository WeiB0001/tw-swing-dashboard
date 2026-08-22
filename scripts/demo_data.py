# -*- coding: utf-8 -*-
"""
demo_data.py — 離線示範資料，同時也是排名邏輯的驗證用資料集

不是純亂數。這裡刻意產生規格書列出的幾種型態，用來檢查新的排名邏輯
是否真的把「強勢」跟「適合現在買」分開：

  A 低檔止跌轉強  ── 應該排前段（規格 Case 1）
  B 強勢但已漲多  ── 趨勢分數高，但位置扣分、風險扣分，不該排第一（Case 2）
  C 帶量突破      ── 應該仍能拿高分（Case 3）
  D 跌深未止跌    ── RSI 很低但不該排第一（Case 4）
  E 區間整理      ── 中性對照組

正式模式完全不會用到這個檔案。價格全部是模擬的，不是真實行情。
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import config as C
import indicators
import scoring

N_BARS = 220

# 代號與名稱只是讓預覽看起來像真的，價格全部模擬
DEMO_STOCKS = [
    ("2330", "台積電"), ("2317", "鴻海"), ("2454", "聯發科"), ("2308", "台達電"),
    ("2382", "廣達"), ("2412", "中華電"), ("2881", "富邦金"), ("2882", "國泰金"),
    ("2891", "中信金"), ("2603", "長榮"), ("2609", "陽明"), ("2615", "萬海"),
    ("1301", "台塑"), ("1303", "南亞"), ("1326", "台化"), ("6505", "台塑化"),
    ("2303", "聯電"), ("3231", "緯創"), ("2377", "微星"), ("2376", "技嘉"),
    ("3037", "欣興"), ("3034", "聯詠"), ("2379", "瑞昱"), ("2357", "華碩"),
    ("3008", "大立光"), ("2409", "友達"), ("2408", "南亞科"), ("4938", "和碩"),
    ("2327", "國巨"), ("2345", "智邦"), ("3661", "世芯-KY"), ("2474", "可成"),
    ("1590", "亞德客-KY"), ("2395", "研華"), ("6669", "緯穎"), ("3017", "奇鋐"),
    ("2002", "中鋼"), ("1101", "台泥"), ("2105", "正新"), ("2207", "和泰車"),
    ("2912", "統一超"), ("1216", "統一"), ("2801", "彰銀"), ("2886", "兆豐金"),
    ("5880", "合庫金"), ("2884", "玉山金"), ("2885", "元大金"), ("2892", "第一金"),
    ("3711", "日月光投控"), ("6415", "矽力-KY"), ("8046", "南電"), ("3045", "台灣大"),
    ("4904", "遠傳"), ("2356", "英業達"), ("2301", "光寶科"), ("1402", "遠東新"),
    ("2049", "上銀"), ("2610", "華航"), ("5871", "中租-KY"), ("9910", "豐泰"),
]

ARCHETYPES = ["A_低檔轉強", "B_強勢漲多", "C_帶量突破", "D_跌深未止跌", "E_區間整理"]


def _ohlcv(closes: np.ndarray, vols: np.ndarray, rng, close_pos=None, upper=None):
    """由收盤價序列補出開高低與量，close_pos/upper 可指定最後一根 K 棒的形狀。"""
    n = len(closes)
    opens = np.concatenate([[closes[0]], closes[:-1]]) * (1 + rng.normal(0, 0.003, n))
    spread = np.abs(rng.normal(0, 0.010, n)) + 0.004
    highs = np.maximum(opens, closes) * (1 + spread)
    lows = np.minimum(opens, closes) * (1 - spread)

    if close_pos is not None:
        # 指定最後一根收盤在當日振幅的哪個位置（1.0 = 收最高）
        lo, hi = lows[-1], highs[-1]
        span = max(hi - lo, closes[-1] * 0.005)
        lo = closes[-1] - span * close_pos
        hi = lo + span
        if upper is not None:                       # 指定上影線比例
            hi = closes[-1] + span * upper
        lows[-1], highs[-1] = min(lo, closes[-1], opens[-1]), max(hi, closes[-1], opens[-1])

    idx = pd.date_range(end=datetime.now().date() - timedelta(days=1), periods=n, freq="B")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols}, index=idx
    )


def _base_path(rng, n, drift, vol):
    return np.cumsum(rng.normal(drift, vol, n))


def make_case(kind: str, seed: int) -> pd.DataFrame:
    """依型態產生一段日線。刻意控制最後幾根，讓訊號可預期、可驗證。"""
    rng = np.random.default_rng(seed)
    n = N_BARS
    base = float(rng.uniform(28, 620))
    vol_base = rng.uniform(6e6, 5e7)
    vols = rng.lognormal(np.log(vol_base), 0.28, n)

    if kind == "A_低檔轉強":
        # 前段緩漲 → 中段回檔 25 日 → 最後 3 日止跌翻紅、量增
        path = _base_path(rng, n, 0.0006, 0.013)
        path[-28:] -= np.linspace(0, rng.uniform(0.14, 0.20), 28)
        path[-3:] += np.linspace(0.005, rng.uniform(0.035, 0.055), 3)
        closes = base * np.exp(path)
        vols[-3:] *= rng.uniform(1.5, 2.1)
        return _ohlcv(closes, vols, rng, close_pos=0.85)

    if kind == "B_強勢漲多":
        # 連續 45 日走高，最後貼著波段高點、RSI 過熱、量沒特別放大
        path = _base_path(rng, n, 0.0004, 0.011)
        path[-45:] += np.linspace(0, rng.uniform(0.34, 0.46), 45)
        path[-4:] += np.linspace(0.004, 0.020, 4)
        closes = base * np.exp(path)
        vols[-1] *= rng.uniform(0.9, 1.25)
        return _ohlcv(closes, vols, rng, close_pos=0.75)

    if kind == "C_帶量突破":
        # 30 日箱型整理 → 最後一日帶大量突破箱頂、收最高
        path = _base_path(rng, n, 0.0005, 0.012)
        box = path[-31]
        path[-31:-1] = box + rng.normal(0, 0.012, 30)          # 壓成箱型
        closes = base * np.exp(path)
        box_high = closes[-31:-1].max()
        closes[-1] = box_high * rng.uniform(1.032, 1.055)      # 突破箱頂
        vols[-1] = np.median(vols[-21:-1]) * rng.uniform(2.4, 3.2)
        return _ohlcv(closes, vols, rng, close_pos=0.93)

    if kind == "D_跌深未止跌":
        # 一路走低，今天再創新低、收黑、量還放大（破底追殺）
        path = _base_path(rng, n, -0.0002, 0.013)
        path[-40:] -= np.linspace(0, rng.uniform(0.26, 0.38), 40)
        closes = base * np.exp(path)
        closes[-1] = closes[-2] * rng.uniform(0.965, 0.988)    # 今日續跌創新低
        vols[-1] *= rng.uniform(1.4, 2.3)
        return _ohlcv(closes, vols, rng, close_pos=0.12)

    # E_區間整理
    path = _base_path(rng, n, 0.0001, 0.011)
    path[-40:] = path[-41] + rng.normal(0, 0.018, 40)
    closes = base * np.exp(path)
    return _ohlcv(closes, vols, rng, close_pos=0.5)


def build_dataset() -> list[tuple[str, str, str, pd.DataFrame]]:
    """回傳 [(代號, 名稱, 型態, 日線)]，型態依序輪流分配，結果可重現。"""
    out = []
    for i, (code, name) in enumerate(DEMO_STOCKS):
        kind = ARCHETYPES[i % len(ARCHETYPES)]
        out.append((code, name, kind, make_case(kind, seed=2000 + i)))
    return out


def build_demo_payload() -> dict:
    from build import build_row   # 延遲 import 避免循環相依

    now = datetime.now(C.TZ)
    rows = []
    for code, name, kind, hist in build_dataset():
        feats = indicators.compute_features(hist)
        if feats:
            row = build_row(code, name, feats, scoring.score_stock(feats))
            row["demo_archetype"] = kind
            rows.append(row)

    rows.sort(key=scoring.sort_key)
    top = [r for r in rows if r["score"] >= C.MIN_SCORE_TO_SHOW][: C.TOP_N_DISPLAY]

    return {
        "meta": {
            "generated_at": now.strftime("%Y-%m-%d %H:%M"),
            "generated_iso": now.isoformat(timespec="seconds"),
            "trade_date": now.strftime("%Y-%m-%d"),
            "scanned_count": len(rows),
            "universe_count": len(DEMO_STOCKS),
            "top_n": C.TOP_N_BY_TURNOVER,
            "min_score": C.MIN_SCORE_TO_SHOW,
            "source_note": "示範模式（模擬資料，非真實行情）",
            "mode": "demo",
        },
        "index": {
            "date": now.strftime("%Y-%m-%d"),
            "close": 23456.78,
            "change": -132.45,
            "chg_pct": -0.56,
            "source": "示範資料",
        },
        "rows": top,
    }
