# -*- coding: utf-8 -*-
"""
build.py — 主流程（GitHub Actions 每天執行的就是這支）

流程：
  1. 抓證交所當日全市場快照 → 決定掃描池
  2. 用 yfinance 抓掃描池的歷史日線（缺的用 FinMind 備援，需 token）
  3. 用證交所當日資料校正最後一根 K 棒
  4. 算技術指標 → 算獲利可能性分數 → 依 tie-breaker 排序取前 N
  5. 若 data/backtest.json 存在，掛上「歷史相似訊號勝率」
  6. 渲染 index.html + data/latest.json

用法：
  python scripts/build.py           # 正式跑（需要網路）
  python scripts/build.py --demo    # 離線示範，用模擬資料產生版面預覽
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

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

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 組裝單一個股的輸出列
# ---------------------------------------------------------------------------
def build_row(code: str, name: str, f: dict, result: dict) -> dict:
    """
    原有的指標欄位全部保留（RSI、量能、均線、20 日高低…），
    只是排名改用 result["score"]（獲利可能性分數）。
    """
    return {
        "code": code,
        "name": name,
        # --- 原始指標（UI 展開後顯示，一個都沒刪） ---
        "close": f["close"],
        "lot_cost": round(f["close"] * 1000),   # 一張（1000 股）要多少錢
        "chg_pct": f["chg_pct"],
        "rsi": f["rsi"],
        "vol_ratio": f["vol_ratio"],
        "volume": f["volume"],
        "vol_ma20": f["vol_ma20"],
        "ma5": f["ma5"],
        "ma10": f["ma10"],
        "ma20": f["ma20"],
        "ma60": f["ma60"],
        "ma20_slope": f["ma20_slope"],
        "ma60_slope": f["ma60_slope"],
        "bias20": f["bias20"],
        "pos20": f["pos20"],
        "pct_above_low20": f["pct_above_low20"],
        "pct_below_high20": f["pct_below_high20"],
        "pct_below_high5": f["pct_below_high5"],
        "pct_below_high60": f["pct_below_high60"],
        "atr": f["atr"],
        "atr_pct": f["atr_pct"],
        "macd_hist": f["macd_hist"],
        "target": f["target"],
        "support": f["support"],
        # --- 新的排名結果 ---
        "score": result["score"],              # 獲利可能性分數（排名主鍵）
        "opportunity": result["opportunity"],
        "risk": result["risk"],
        "breakdown": result["breakdown"],
        "risk_items": result["risk_items"],
        "stars": result["stars"],
        "risk_level": result["risk_level"],
        "kind": result["kind"],
        "headline": result["headline"],
        "why": result["why"],
        "main_risk": result["main_risk"],
        "swing_low": result["swing_low"],
        "swing_high": result["swing_high"],
        "upside_pct": result["upside_pct"],
        "downside_pct": result["downside_pct"],
        "rr_ratio": result["rr_ratio"],
        "reasons": result["reasons"],
        "flags": result["flags"],
        "hist_winrate": None,                  # 有回測資料時才會被填上
        "hist_samples": None,
        "hist_avg_return": None,
    }


# ---------------------------------------------------------------------------
# 掛上回測結果（有跑過 scripts/backtest.py 才會有）
# ---------------------------------------------------------------------------
def attach_backtest(rows: list[dict]) -> dict | None:
    """
    依分數落在哪個級距，掛上該級距的歷史勝率。
    這是「歷史相似訊號勝率」，跟模型分數是兩件事，UI 上會分開顯示。
    """
    path = ROOT / C.BACKTEST_JSON
    if not path.exists():
        return None
    try:
        bt = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("回測檔讀取失敗：%s", e)
        return None

    buckets = bt.get("score_buckets", [])
    for r in rows:
        for b in buckets:
            if b["lo"] <= r["score"] < b["hi"] and b["samples"] >= C.BACKTEST_MIN_SAMPLES:
                r["hist_winrate"] = b["win_rate"]
                r["hist_samples"] = b["samples"]
                r["hist_avg_return"] = b["avg_return"]
                break
    log.info("已掛上回測勝率（回測日期：%s）", bt.get("generated_at", "?"))
    return bt


# ---------------------------------------------------------------------------
# 正式模式
# ---------------------------------------------------------------------------
def run_live() -> dict:
    now = datetime.now(C.TZ)

    snapshot = fetch.fetch_twse_snapshot()
    if snapshot.empty:
        raise RuntimeError("證交所當日行情取得失敗，無法決定掃描池。可能是非交易日或 API 暫時異常。")

    universe = fetch.build_universe(snapshot)
    codes = universe["code"].tolist()
    name_map = dict(zip(universe["code"], universe["name"]))

    hist_map = fetch.fetch_history_yf(codes)
    missing = [c for c in codes if c not in hist_map]
    if missing:
        log.warning("yfinance 缺 %d 檔，嘗試 FinMind 備援", len(missing))
        hist_map.update(fetch.fetch_history_finmind(missing[:40]))
    if not hist_map:
        raise RuntimeError("歷史日線全部取得失敗，無法計算指標。")

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
    bt = attach_backtest(rows)
    rows.sort(key=scoring.sort_key)          # 主鍵分數，接近時用 tie-breaker
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
        "backtest": _backtest_summary(bt),
        "rows": top,
    }


def _backtest_summary(bt: dict | None) -> dict | None:
    """把回測結果濃縮成 UI 要顯示的幾個數字。"""
    if not bt:
        return None
    return {
        "generated_at": bt.get("generated_at"),
        "period": bt.get("period"),
        "hold_days": bt.get("primary_hold_days"),
        "topk": bt.get("topk", {}),
        "samples": bt.get("total_signals"),
    }


# ---------------------------------------------------------------------------
# 示範模式（不連網）
# ---------------------------------------------------------------------------
def run_demo() -> dict:
    import demo_data
    payload = demo_data.build_demo_payload()
    bt = attach_backtest(payload["rows"])
    payload["backtest"] = _backtest_summary(bt)
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="台股價差機會儀表板")
    ap.add_argument("--demo", action="store_true", help="用模擬資料產生版面預覽（不連網）")
    args = ap.parse_args()

    try:
        payload = run_demo() if args.demo else run_live()
    except Exception as e:
        log.error("建置失敗：%s", e)
        # 失敗時不覆蓋昨天的 index.html，讓使用者仍看得到上一版
        return 1

    render.write_outputs(payload)
    log.info("完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
