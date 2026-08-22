# -*- coding: utf-8 -*-
"""
build.py — 主流程（GitHub Actions 每天執行的就是這支）

流程：
  1. 抓證交所當日全市場快照 → 決定掃描池
  2. 用 yfinance 抓掃描池的歷史日線（缺的用 FinMind 備援，需 token）
  3. 用證交所當日資料校正最後一根 K 棒
  4. 算技術指標 → 算價差機會分數 → 排序取前 N
  5. 渲染 index.html + data/latest.json

用法：
  python scripts/build.py           # 正式跑（需要網路）
  python scripts/build.py --demo    # 離線示範，用模擬資料產生版面預覽
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# 讓 scripts/ 內的模組可以直接互相 import
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import config as C
import fetch
import indicators
import render
import scoring

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build")


# ---------------------------------------------------------------------------
# 組裝單一個股的輸出列
# ---------------------------------------------------------------------------
def build_row(code: str, name: str, feats: dict, result: dict) -> dict:
    return {
        "code": code,
        "name": name,
        "close": feats["close"],
        "chg_pct": feats["chg_pct"],
        "rsi": feats["rsi"],
        "vol_ratio": feats["vol_ratio"],
        "volume": feats["volume"],
        "ma5": feats["ma5"],
        "ma10": feats["ma10"],
        "ma20": feats["ma20"],
        "pct_above_low20": feats["pct_above_low20"],
        "pct_below_high20": feats["pct_below_high20"],
        "pct_below_high5": feats["pct_below_high5"],
        "score": result["score"],
        "headline": result["headline"],
        "reasons": result["reasons"],
        "flags": result["flags"],
        "breakdown": result["breakdown"],
    }


# ---------------------------------------------------------------------------
# 正式模式
# ---------------------------------------------------------------------------
def run_live() -> dict:
    now = datetime.now(C.TZ)

    # --- 1) 當日快照 & 掃描池 ---
    snapshot = fetch.fetch_twse_snapshot()
    if snapshot.empty:
        raise RuntimeError("證交所當日行情取得失敗，無法決定掃描池。可能是非交易日或 API 暫時異常。")

    universe = fetch.build_universe(snapshot)
    codes = universe["code"].tolist()
    name_map = dict(zip(universe["code"], universe["name"]))

    # --- 2) 歷史日線 ---
    hist_map = fetch.fetch_history_yf(codes)
    missing = [c for c in codes if c not in hist_map]
    if missing:
        log.warning("yfinance 缺 %d 檔，嘗試 FinMind 備援", len(missing))
        hist_map.update(fetch.fetch_history_finmind(missing[:40]))

    if not hist_map:
        raise RuntimeError("歷史日線全部取得失敗，無法計算指標。")

    # --- 3~4) 校正 + 指標 + 評分 ---
    trade_date = pd.Timestamp(now.date())
    rows = []
    for _, srow in universe.iterrows():
        code = srow["code"]
        hist = hist_map.get(code)
        if hist is None:
            continue
        try:
            hist = fetch.merge_today_bar(hist, srow, trade_date)
            feats = indicators.compute_features(hist)
            if not feats:
                continue
            rows.append(build_row(code, name_map.get(code, code), feats, scoring.score_stock(feats)))
        except Exception as e:
            log.warning("%s 計算失敗：%s", code, e)

    scanned = len(rows)
    rows.sort(key=lambda r: r["score"], reverse=True)
    top = [r for r in rows if r["score"] >= C.MIN_SCORE_TO_SHOW][: C.TOP_N_DISPLAY]
    log.info("完成計算：%d 檔，達標 %d 檔", scanned, len(top))

    return {
        "meta": {
            "generated_at": now.strftime("%Y-%m-%d %H:%M"),
            "generated_iso": now.isoformat(timespec="seconds"),
            "trade_date": now.strftime("%Y-%m-%d"),
            "scanned_count": scanned,
            "universe_count": len(universe),
            "top_n": C.TOP_N_BY_TURNOVER,
            "min_score": C.MIN_SCORE_TO_SHOW,
            "source_note": "證交所 OpenAPI（當日行情）＋ yfinance（歷史日線）",
            "mode": "live",
        },
        "index": fetch.fetch_market_index(),
        "rows": top,
    }


# ---------------------------------------------------------------------------
# 示範模式（不連網，用來預覽版面）
# ---------------------------------------------------------------------------
def run_demo() -> dict:
    import demo_data
    return demo_data.build_demo_payload()


def main() -> int:
    ap = argparse.ArgumentParser(description="台股價差機會儀表板")
    ap.add_argument("--demo", action="store_true", help="用模擬資料產生版面預覽（不連網）")
    args = ap.parse_args()

    try:
        payload = run_demo() if args.demo else run_live()
    except Exception as e:
        log.error("建置失敗：%s", e)
        # 失敗時不要覆蓋掉昨天的 index.html，讓使用者仍看得到上一版
        return 1

    render.write_outputs(payload)
    log.info("完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
