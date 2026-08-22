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
import plan
import tracking
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
        "sector": C.sector_of(code),        # 產業分類（卡片上顯示）
        "asset_type": C.asset_type(code),   # etf / tech / other，供網頁篩選
        # --- 原始指標（UI 展開後顯示，一個都沒刪） ---
        "close": f["close"],
        "day_open": f.get("open"),              # 模擬組合用 t+1 開盤價成交
        "day_high": f.get("high"),
        "day_low": f.get("low"),
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
        # --- 歷史勝率（跑過 backtest.py 才有；沒有就是「樣本不足」） ---
        "hist_calibrated": None,    # 平滑後勝率 (wins+10)/(samples+20)
        "hist_raw": None,           # 未平滑勝率
        "hist_samples": None,
        "hist_avg_return": None,    # 平均淨報酬（已扣交易成本）
        "hist_pf": None,            # Profit Factor
        "hist_mdd": None,           # 平均最大回撤
        "hist_source": None,        # regime+pattern / pattern / bucket / None
        "hist_expectancy": None,    # 期望值 %
        "hist_confidence": 0,       # 可信度星數（0 = 樣本不足）
        "plan": plan.build_plan(f, result),     # 明日交易計畫（純規則）
    }


# ---------------------------------------------------------------------------
# 掛上回測結果（有跑過 scripts/backtest.py 才會有）
# ---------------------------------------------------------------------------
def _confidence(n) -> int:
    """依樣本數給可信度星數；不足 30 筆回 0（頁面顯示樣本不足）。"""
    try:
        n = int(n or 0)
    except Exception:
        return 0
    for need, stars in C.CONFIDENCE_TIERS:
        if n >= need:
            return stars
    return 0


def attach_backtest(rows: list[dict], regime: str = "sideways") -> dict | None:
    """
    掛上歷史統計。查表順序（樣本不足就往下退一層）：

      1. 大盤狀態 + 型態 + 分數級距   ← 最精細
      2. 型態 + 分數級距
      3. 分數級距
      4. 都不足 → 全部留空，UI 顯示「樣本不足」。絕不生假數字。
    """
    path = ROOT / C.BACKTEST_JSON
    if not path.exists():
        log.info("沒有 %s，本次排名只能用技術分數（頁面會標示樣本不足）", C.BACKTEST_JSON)
        return None
    try:
        bt = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("回測檔讀取失敗：%s", e)
        return None

    reg_buckets = bt.get("regime_buckets", []) or []
    pat_buckets = bt.get("pattern_buckets", []) or []
    buckets = bt.get("score_buckets", []) or []

    def fill(r, d, source):
        r["hist_calibrated"] = d.get("calibrated_win_rate")
        r["hist_raw"] = d.get("win_rate")
        r["hist_samples"] = d.get("samples")
        r["hist_avg_return"] = d.get("avg_return")
        r["hist_pf"] = d.get("profit_factor")
        r["hist_mdd"] = d.get("avg_mdd")
        r["hist_expectancy"] = d.get("expectancy")
        r["hist_confidence"] = _confidence(d.get("samples"))
        r["hist_source"] = source

    counts = {"regime": 0, "pattern": 0, "bucket": 0, "none": 0}
    for r in rows:
        hit = None
        for b in reg_buckets:
            if (b.get("regime") == regime and b.get("pattern") == r["kind"]
                    and b["lo"] <= r["score"] < b["hi"]
                    and b.get("samples", 0) >= C.PATTERN_MIN_SAMPLES):
                hit = (b, "regime"); break
        if hit is None:
            for b in pat_buckets:
                if (b.get("pattern") == r["kind"] and b["lo"] <= r["score"] < b["hi"]
                        and b.get("samples", 0) >= C.PATTERN_MIN_SAMPLES):
                    hit = (b, "pattern"); break
        if hit is None:
            for b in buckets:
                if (b["lo"] <= r["score"] < b["hi"]
                        and b.get("samples", 0) >= C.BACKTEST_MIN_SAMPLES):
                    hit = (b, "bucket"); break
        if hit:
            fill(r, hit[0], hit[1])
            counts[hit[1]] += 1
        else:
            counts["none"] += 1

    log.info("歷史統計掛載：大盤層 %d、型態層 %d、級距層 %d、樣本不足 %d（回測 %s）",
             counts["regime"], counts["pattern"], counts["bucket"], counts["none"],
             bt.get("generated_at", "?"))
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

    # 台股歷史日線一律走 FinMind + data/history 快取（與回測共用同一份）
    hist_map = fetch.fetch_history(codes)
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
    twii = fetch.fetch_twii_history()
    regime = indicators.regime_of(twii)
    for r in rows:
        r["regime"] = regime
    bt = attach_backtest(rows, regime)
    rows.sort(key=scoring.sort_key)          # 主鍵分數，接近時用 tie-breaker
    for i, r in enumerate(rows, 1):
        r["rank"] = i                        # 完整名次，之後不論怎麼篩選都用這個
    strong = sum(1 for r in rows if r["score"] >= C.WEAK_SCORE)
    top = rows[: C.RENDER_LIMIT] if C.RENDER_LIMIT else rows
    log.info("完成計算：%d 檔，其中 %d 檔分數 >= %d｜大盤狀態 %s",
             scanned, strong, C.WEAK_SCORE, regime)

    index_info = fetch.fetch_market_index()
    idx_close = (index_info or {}).get("close")
    signals = tracking.update_signals(rows, now.strftime("%Y-%m-%d"), regime)
    for r in rows:
        r["mark"] = signals["marks"].get(r["code"], {})
    portfolio = tracking.update_portfolio(rows, now.strftime("%Y-%m-%d"), idx_close)

    return {
        "meta": {
            "generated_at": now.strftime("%Y-%m-%d %H:%M"),
            "generated_iso": now.isoformat(timespec="seconds"),
            "trade_date": now.strftime("%Y-%m-%d"),
            "scanned_count": scanned,
            "qualified_count": len(rows),
            "universe_count": len(universe),
            "has_backtest": bt is not None,
            # 只要有任何一檔查到可信勝率，標題才叫「勝率排行」；否則叫「機會排行」
            "has_winrate": any(r.get("hist_calibrated") is not None for r in rows),
            "regime": regime,
            "top_n": C.TOP_N_BY_TURNOVER,
            "weak_score": C.WEAK_SCORE,
            "strong_count": strong,
            "source_note": "證交所 OpenAPI（當日行情）＋ FinMind（歷史日線）",
            "mode": "live",
        },
        "index": index_info,
        "us": _us_snapshot(),
        "signals": signals,
        "portfolio": portfolio,
        "backtest": _backtest_summary(bt),
        "rows": top,
    }


def _us_snapshot() -> dict | None:
    """美股連動區塊。抓不到就回 None，不影響主流程。"""
    try:
        import us_market
        return us_market.build_us_snapshot()
    except Exception as e:
        log.warning("美股區塊建立失敗：%s", e)
        return None


def _backtest_summary(bt: dict | None) -> dict | None:
    """把回測結果濃縮成 UI 要顯示的幾個數字。"""
    if not bt:
        return None
    return {
        "generated_at": bt.get("generated_at"),
        "period": bt.get("period"),
        "hold_days": bt.get("primary_hold_days"),
        "cost_pct": bt.get("cost_pct"),
        "cooldown_days": bt.get("cooldown_days"),
        "entry_rule": bt.get("entry_rule"),
        "samples": bt.get("total_signals"),
        "walk_forward": bt.get("walk_forward"),
    }


# ---------------------------------------------------------------------------
# 示範模式（不連網）
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
        # 失敗時不覆蓋昨天的 index.html，讓使用者仍看得到上一版
        return 1

    render.write_outputs(payload)
    log.info("完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
