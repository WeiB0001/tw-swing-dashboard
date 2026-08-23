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
    # ETF（示範用，價格同樣是模擬的）
    ("0050", "元大台灣50"), ("0056", "元大高股息"), ("00878", "國泰永續高股息"),
    ("00919", "群益台灣精選高息"), ("00891", "中信關鍵半導體"), ("006208", "富邦台50"),
    ("00929", "復華台灣科技優息"), ("0052", "富邦科技"), ("00940", "元大台灣價值高息"),
    ("00713", "元大台灣高息低波"),
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

    # end 落在週末時，freq="B" 產出的筆數會少一天，所以多產一些再取尾端
    idx = pd.date_range(end=datetime.now().date() - timedelta(days=1),
                        periods=n + 6, freq="B")[-n:]
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols}, index=idx
    )


def _base_path(rng, n, drift, vol):
    return np.cumsum(rng.normal(drift, vol, n))


def make_case(kind: str, seed: int, etf: bool = False) -> pd.DataFrame:
    """依型態產生一段日線。刻意控制最後幾根，讓訊號可預期、可驗證。"""
    rng = np.random.default_rng(seed)
    n = N_BARS
    base = float(rng.uniform(28, 620))
    if etf:
        base = float(rng.uniform(15, 60))   # ETF 價格帶比個股低很多，一張多在 1.5～6 萬
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
        out.append((code, name, kind, make_case(kind, seed=2000 + i, etf=C.is_etf(code))))
    return out


def build_demo_payload() -> dict:
    from build import build_row   # 延遲 import 避免循環相依

    now = datetime.now(C.TZ)
    rows = []
    for code, name, kind, hist in build_dataset():
        feats = indicators.compute_features(hist)
        if feats:
            from build import make_spark
            row = build_row(code, name, feats, scoring.score_stock(feats))
            row["spark"] = make_spark(hist)
            row["demo_archetype"] = kind
            rows.append(row)

    from build import attach_backtest, _backtest_summary
    import tracking
    regime = "sideways"
    for r in rows:
        r["regime"] = regime
    bt = attach_backtest(rows, regime)               # 先掛歷史勝率，排序才有依據
    rows.sort(key=scoring.sort_key)
    for i, r in enumerate(rows, 1):
        r["rank"] = i                        # 完整名次，之後不論怎麼篩選都用這個
    strong = sum(1 for r in rows if r["score"] >= C.WEAK_SCORE)
    top = rows[: C.RENDER_LIMIT] if C.RENDER_LIMIT else rows

    # 示範模式也產生一份雷達資料，方便預覽 radar.html
    try:
        import radar as radar_mod, universe as universe_mod
        from build import _write_universe
        hist_map = {code: hist for code, _, _, hist in build_dataset()}
        core_codes = set(list(hist_map)[:20])          # 前 20 檔當作 Core
        cand = []
        for code in hist_map:
            kind = "etf" if C.is_etf(code) else ("tech" if code in C.TECH_SECTORS
                                                 else ("finance" if code in C.FINANCE_STOCKS else "tech"))
            cand.append({"symbol": code, "name": dict(DEMO_STOCKS).get(code, ""),
                         "sector": C.sector_of(code), "kind": kind,
                         "market": "ETF" if kind == "etf" else ("TPEx" if code[0] in "5678" else "TWSE"),
                         "is_core": code in core_codes})
        rd = radar_mod.scan(cand, hist_map, core_codes)
        core_df = pd.DataFrame([{"code": c, "name": dict(DEMO_STOCKS).get(c, ""),
                                 "sector": C.sector_of(c), "kind": "tech", "market": "TWSE"}
                                for c in core_codes])
        _write_universe(core_df, cand, rd, universe_mod.stats(core_df, cand),
                        now.strftime("%Y-%m-%d"))
        # 示範用的「其他產業」：借用後段幾檔當範例
        from build import _scan_others
        others = [{"symbol": c, "name": dict(DEMO_STOCKS).get(c, ""), "sector": "傳產範例",
                   "market": "TWSE"} for c in list(hist_map)[-25:]]
        _scan_others(others, hist_map, now.strftime("%Y-%m-%d"))
    except Exception as e:
        import logging; logging.getLogger("demo").warning("示範雷達產生失敗：%s", e)

    from build import add_edges
    signals = tracking.update_signals(rows, now.strftime("%Y-%m-%d"), regime)
    for r in rows:
        r["mark"] = signals["marks"].get(r["code"], {})
    portfolio = tracking.update_portfolio(rows, now.strftime("%Y-%m-%d"), 23456.78)
    add_edges(rows[: C.INITIAL_VISIBLE * 2])

    return {
        "meta": {
            "generated_at": now.strftime("%Y-%m-%d %H:%M"),
            "generated_iso": now.isoformat(timespec="seconds"),
            "trade_date": now.strftime("%Y-%m-%d"),
            "data_date": now.strftime("%Y-%m-%d"),
            "scanned_count": len(rows),
            "qualified_count": len(rows),
            "universe_count": len(DEMO_STOCKS),
            "has_backtest": bt is not None,
            "has_winrate": any(r.get("hist_calibrated") is not None for r in rows),
            "regime": regime,
            "top_n": C.TOP_N_BY_TURNOVER,
            "weak_score": C.WEAK_SCORE,
            "strong_count": strong,
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
        "us": {
            "last_night": [
                {"label": "那斯達克", "chg_pct": 0.82, "date": now.strftime("%Y-%m-%d")},
                {"label": "費城半導體", "chg_pct": 1.54, "date": now.strftime("%Y-%m-%d")},
                {"label": "S&P 500", "chg_pct": 0.41, "date": now.strftime("%Y-%m-%d")},
                {"label": "道瓊", "chg_pct": -0.12, "date": now.strftime("%Y-%m-%d")},
            ],
            "futures": [
                {"label": "那斯達克期貨", "chg_pct": 0.35},
                {"label": "S&P 500 期貨", "chg_pct": 0.21},
            ],
            "forecast": {"available": True, "point": 0.46, "low": -0.53, "high": 1.45,
                         "basis": "以那斯達克期貨 +0.35% 代入（示範數字）",
                         "r2": 0.38, "samples": 240},
            "model": {"samples": 240, "r2": 0.38, "beta_nasdaq": 0.41,
                      "beta_sox": 0.19, "resid_sd": 0.99},
        },
        "backtest": _backtest_summary(bt),
        "signals": signals,
        "portfolio": portfolio,
        "rows": top,
    }
