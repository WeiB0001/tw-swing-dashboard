# -*- coding: utf-8 -*-
"""
demo_data.py — 離線示範資料

用固定亂數種子模擬 60 檔個股的 8 個月日線，走完真正的指標與評分流程，
只是價格是假的。用途：
  1. 你部署前先看看版面長什麼樣（python scripts/build.py --demo）
  2. 改了訊號邏輯後，不用等收盤就能確認程式沒寫壞

正式模式完全不會用到這個檔案。
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import config as C
import indicators
import scoring

# 代號與名稱只是為了讓預覽看起來像真的，價格全部是模擬產生
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


def _synth_series(seed: int, n: int = 200) -> pd.DataFrame:
    """
    生成一段有「趨勢 + 回檔 + 反彈」結構的假日線，
    讓評分結果有高有低，比純隨機更能看出版面效果。
    """
    rng = np.random.default_rng(seed)
    base = float(rng.uniform(22, 780))

    drift = rng.normal(0.0004, 0.0006)
    shock = rng.normal(0, 1, n) * rng.uniform(0.012, 0.028)
    path = np.cumsum(drift + shock)

    # 在最後 30 根裡塞一段回檔，再視情況給一段反彈，製造超賣/反彈樣本
    dip_len = int(rng.integers(8, 22))
    path[-dip_len:] -= np.linspace(0, rng.uniform(0.04, 0.16), dip_len)
    if rng.random() < 0.55:
        bounce = int(rng.integers(1, 4))
        path[-bounce:] += np.linspace(0, rng.uniform(0.01, 0.05), bounce)

    close = base * np.exp(path)
    high = close * (1 + np.abs(rng.normal(0, 0.008, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.008, n)))
    open_ = np.concatenate([[close[0]], close[:-1]]) * (1 + rng.normal(0, 0.004, n))

    vol = rng.lognormal(mean=np.log(rng.uniform(4e6, 6e7)), sigma=0.35, size=n)
    vol[-1] *= rng.uniform(0.7, 3.2)     # 最後一天量能隨機放大/縮小

    idx = pd.date_range(end=datetime.now().date() - timedelta(days=1), periods=n, freq="B")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx
    )


def build_demo_payload() -> dict:
    from build import build_row   # 延遲 import 避免循環相依

    now = datetime.now(C.TZ)
    rows = []
    for i, (code, name) in enumerate(DEMO_STOCKS):
        hist = _synth_series(seed=1000 + i)
        feats = indicators.compute_features(hist)
        if feats:
            rows.append(build_row(code, name, feats, scoring.score_stock(feats)))

    rows.sort(key=lambda r: r["score"], reverse=True)
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
