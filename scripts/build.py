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
import momentum as momentum_mod
import radar as radar_mod
import tracking
import universe as universe_mod
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
def make_spark(hist) -> dict | None:
    """卡片展開用的迷你價量圖資料：最近幾根的收盤與成交量，壓到最小體積。"""
    try:
        tail = hist.tail(C.SPARK_BARS)
        if len(tail) < 10:
            return None
        c = [round(float(x), 2) for x in tail["close"]]
        v = [int(x) for x in tail["volume"]]
        return {"c": c, "v": v}
    except Exception:
        return None


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
        "prev_close": f.get("prev_close"),      # 記帳算今日損益用
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
        "macd_hist_prev": f.get("macd_hist_prev"),
        "ma5_slope": f.get("ma5_slope"),
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

    # --- 雙層股票池：Core 給首頁排行，Extended 給雷達與搜尋 ---
    info = fetch.fetch_stock_info()                 # 產業分類（7 天更新一次）
    universe = universe_mod.build_core(snapshot, info)
    if universe.empty:                              # 分類資料異常時的保險
        log.warning("Core 股票池是空的，退回原本的掃描池邏輯")
        universe = fetch.build_universe(snapshot)
    codes = universe["code"].tolist()
    name_map = dict(zip(universe["code"], universe["name"]))
    core_codes = set(codes)
    extended = universe_mod.build_extended(snapshot, info, core_codes)

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
            row = build_row(code, name_map.get(code, code), feats, scoring.score_stock(feats))
            row["spark"] = make_spark(hist)
            try:
                row["prev_low"] = float(hist["low"].iloc[-2])   # 判斷是否跌破昨日低點
            except Exception:
                row["prev_low"] = None
            rows.append(row)
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

    data_date = _data_date(hist_map, now)
    # 早上那班只更新海外盤與排名依據（台股還沒開盤，行情仍是前一交易日的）
    run_type = "morning" if now.hour < 12 else "close"
    index_info = fetch.fetch_market_index()
    idx_close = (index_info or {}).get("close")
    # 用 data_date 而不是執行日：早上那班的行情日跟前一天收盤班相同，
    # tracking 會判定為同一天而不重複記錄、也不會用舊開盤價成交
    signals = tracking.update_signals(rows, data_date, regime)
    for r in rows:
        r["mark"] = signals["marks"].get(r["code"], {})
    portfolio = tracking.update_portfolio(rows, data_date, idx_close)
    add_momentum(rows)                         # 動能確認：只用今天的價量分層
    add_final_score(rows)                      # 綜合分數：只是重新加權既有數字
    rows = sort_by_final(rows)
    # 排序改變了，要重新取要輸出的那一段（模擬組合與訊號追蹤已在前面用模型名次跑完）
    top = rows[: C.RENDER_LIMIT] if C.RENDER_LIMIT else rows                 # 顯示順序改用綜合排名，rank 欄位不變
    add_edges(rows[: C.INITIAL_VISIBLE * 2])   # 只有會被看到的前段需要這句話

    # --- 全站共用的最新行情（首頁與我的交易都讀這份）---
    try:
        # 名稱：上櫃股不在證交所快照裡，用 Extended 清單補上
        name_all = {u["symbol"]: u.get("name", "") for u in (extended or [])}
        name_all.update(name_map)
        _write_latest_prices(snapshot, hist_map, data_date, now, name_all)
    except Exception as e:
        log.warning("最新行情輸出失敗（不影響首頁）：%s", e)

    # --- 明日強勢預測：獨立分頁，只用既有 history 快取，不額外下載 ---
    try:
        fc = momentum_mod.build_forecast(snapshot, hist_map, extended, data_date)
        (ROOT / C.MOMENTUM_JSON).parent.mkdir(parents=True, exist_ok=True)
        (ROOT / C.MOMENTUM_JSON).write_text(
            json.dumps(fc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        log.info("明日強勢預測：%d 檔候選（歷史樣本 %d 筆）",
                 len(fc["rows"]), fc["stats"]["total_samples"])
    except Exception as e:
        log.warning("明日強勢預測失敗（不影響首頁）：%s", e)

    # --- 其他產業：不進首頁，只挑亮眼的另開一頁 ---
    try:
        others = universe_mod.build_others(snapshot, info)
        _scan_others(others, hist_map, data_date)
    except Exception as e:
        log.warning("其他產業掃描失敗（不影響首頁）：%s", e)

    # --- 雷達：只掃 Extended，與首頁共用同一份 history 快取 ---
    radar_rows, uni_stats = [], {}
    try:
        ext_codes = [c["symbol"] for c in extended][: C.EXTENDED_MAX]
        missing = [c for c in ext_codes if c not in hist_map]
        if missing:
            # 只補 cache 缺的那些；已有的完全不重抓
            log.info("雷達：補抓 %d 檔缺少的歷史資料", len(missing))
            hist_map.update(fetch.fetch_history(missing))
        radar_rows = radar_mod.scan([c for c in extended if c["symbol"] in hist_map],
                                    hist_map, core_codes)
        uni_stats = universe_mod.stats(universe, extended)
        _write_universe(universe, extended, radar_rows, uni_stats,
                        now.strftime("%Y-%m-%d"))
    except Exception as e:
        log.warning("雷達掃描失敗（不影響首頁）：%s", e)

    return {
        "meta": {
            "generated_at": now.strftime("%Y-%m-%d %H:%M"),
            "generated_iso": now.isoformat(timespec="seconds"),
            "trade_date": now.strftime("%Y-%m-%d"),      # 這次執行的日期
            "run_type": run_type,                         # morning（海外盤）/ close（收盤）
            "data_date": data_date,                       # 行情資料實際的交易日
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


def _clamp100(x) -> float:
    try:
        v = float(x)
        if v != v:
            return 0.0
        return max(0.0, min(100.0, v))
    except Exception:
        return 0.0


def _scale(x, lo, hi) -> float:
    """把一個值線性映射到 0～100，超出範圍就夾住。"""
    try:
        x = float(x)
        if x != x or hi == lo:
            return 0.0
        return _clamp100((x - lo) / (hi - lo) * 100)
    except Exception:
        return 0.0


def add_momentum(rows: list[dict]) -> None:
    """
    動能確認（0～100）：只回答「今天有沒有出現上漲確認、有沒有追高風險」。

    **不碰任何歷史模型**：EV、勝率、PF、技術分數、回測、walk-forward 全部照舊，
    這裡只用今天的價量把標的分成三層，並在同一層內做最終排序。

    加分項（符合越多分數越高）：
      收盤站上基準價、量能 >= 1.0×、收盤 >= MA5、MA5 上彎、
      MACD 柱不比昨天差、RR >= 1.2、今日漲幅落在 +0.5%～+4%

    **今日大漲不是加分項**：漲幅超過 4% 就不再加分，避免整套變成追高策略。

    扣分項（每項扣 MOM_PENALTY 分，且部分會直接打入 🔴 那一層）：
      今日跌幅 < -3%、收盤跌破 MA5、爆量收黑、跌破昨日低點、
      MACD 柱明顯轉弱、距 MA20 正乖離 > 8%
    """
    for r in rows:
        pl = r.get("plan") or {}
        chg = float(r.get("chg_pct") or 0)
        close = float(r.get("close") or 0)
        ma5 = float(r.get("ma5") or 0)
        vol = float(r.get("vol_ratio") or 0)
        # 用交易計畫的 RR（觸發價→停損 vs 觸發價→目標一），
        # 跟卡片上「明日交易計畫」顯示的是同一個數字，避免兩個 RR 打架
        rr = float(pl.get("rr") or r.get("rr_ratio") or 0)
        bias = float(r.get("bias20") or 0)
        slope5 = float(r.get("ma5_slope") or 0)
        mh = r.get("macd_hist")
        mhp = r.get("macd_hist_prev")
        confirmed = (pl.get("entry_icon") == "🟢")

        # ---- 加分 ----
        plus = [
            (2.0, confirmed, "收盤站上基準價"),
            (1.0, vol >= C.MOM_VOL_MIN, "量能 %.2f×" % vol),
            (1.0, close >= ma5 > 0, "收盤站上 MA5"),
            (1.0, slope5 > 0, "MA5 上彎"),
            (1.0, mh is not None and mhp is not None and mh >= mhp, "MACD 柱未轉弱"),
            (1.0, rr >= C.MOM_RR_MIN, "RR %.2f" % rr),
            (1.0, C.MOM_GOOD_CHG_LOW <= chg <= C.MOM_GOOD_CHG_HIGH,
             "今日 %+.2f%% 在合理帶" % chg),
        ]
        got = sum(w for w, ok, _ in plus if ok)
        total = sum(w for w, _, _ in plus)
        hits = [t for _, ok, t in plus if ok]

        # ---- 扣分 ----
        # 用「收盤」跌破昨日低點，不是盤中最低——
        # 盤中殺一下又拉回來的不算轉弱，否則上漲的股票也會被誤判
        broke_prev_low = False
        pd_low = r.get("prev_low")
        if pd_low and close:
            broke_prev_low = close < float(pd_low)
        macd_weak = (mh is not None and mhp is not None
                     and mh < mhp and (mh < 0 or mh < mhp * 0.5))

        minus = [
            (chg < C.MOM_DROP_BAD, "今日跌 %.2f%%" % chg),
            (ma5 > 0 and close < ma5, "收盤跌破 MA5"),
            (vol >= C.MOM_BLOWOFF and chg < 0, "爆量收黑（%.2f×）" % vol),
            (broke_prev_low, "收盤跌破昨日低點"),
            (macd_weak, "MACD 柱明顯轉弱"),
            (bias > C.MOM_BIAS_MAX, "距 MA20 乖離 %+.1f%%，追高風險" % bias),
        ]
        warns = [t for ok, t in minus if ok]

        score = (got / total * 100) - len(warns) * C.MOM_PENALTY
        r["momentum_score"] = round(max(0.0, min(100.0, score)), 1)
        r["momentum_hits"] = hits
        r["momentum_warns"] = warns

        # ---- 三層 ----
        # 這幾項只要中一個就是轉弱或追高，不該排在前面
        hard = (chg < C.MOM_DROP_BAD or (ma5 > 0 and close < ma5)
                or bias > C.MOM_BIAS_MAX or (vol >= C.MOM_BLOWOFF and chg < 0)
                or broke_prev_low)
        if hard:
            tier, label, icon = 2, "轉弱／追高風險", "🔴"
        elif confirmed and vol >= C.MOM_VOL_MIN and r["momentum_score"] >= C.MOM_GREEN_MIN:
            tier, label, icon = 0, "強勢確認", "🟢"
        else:
            tier, label, icon = 1, "潛力觀察", "🟡"
        r["momentum_tier"] = tier
        r["momentum_label"] = label
        r["momentum_icon"] = icon


def add_final_score(rows: list[dict]) -> None:
    """
    綜合分數（0～100）＝ 65% 模型品質 ＋ 35% 進場品質。

    **這一層完全不動任何模型**：EV、勝率、PF、MDD、技術分數、回測、walk-forward
    全部沿用既有結果，只是把它們標準化後重新加權，回答「這檔現在整體看起來如何」。
    原本的欄位一個都沒被覆蓋。

    沒有歷史統計的（樣本不足）就用中性值 50 代入，不會因為缺資料被懲罰或加分。
    """
    if not rows:
        return

    for r in rows:
        # ---- 模型品質：全部來自既有欄位 ----
        ev = r.get("hist_expectancy")
        wr = r.get("hist_calibrated")
        pf = r.get("hist_pf")
        mdd = r.get("hist_mdd")

        m = {
            # EV −1% ～ +2% 映射到 0～100；沒有樣本給中性 50
            "ev": _scale(ev, -1.0, 2.0) if ev is not None else 50.0,
            # 平滑勝率 40% ～ 65%
            "winrate": _scale(wr, 40.0, 65.0) if wr is not None else 50.0,
            # PF 0.8 ～ 2.0
            "pf": _scale(pf, 0.8, 2.0) if pf is not None else 50.0,
            # 回撤 −8% ～ 0%（越淺越高分）
            "mdd": _scale(mdd, -8.0, 0.0) if mdd is not None else 50.0,
            # 技術分數本來就是 0～100
            "tech": _clamp100(r.get("score")),
        }
        model_q = sum(m[k] * w for k, w in C.FINAL_MODEL_W.items())

        # ---- 進場品質：全部來自 plan 已經算好的欄位 ----
        pl = r.get("plan") or {}
        icon = pl.get("entry_icon") or ""
        vol = pl.get("vol_ratio")
        rr = pl.get("rr")
        dist = pl.get("dist_trigger_pct")

        confirm = {"🟢": 100.0, "🟡": 45.0, "⚠": 20.0, "⛔": 0.0}.get(icon, 30.0)
        e = {
            "confirm": confirm,
            # 量能 0.6× ～ 2.0×
            "volume": _scale(vol, 0.6, 2.0) if vol is not None else 30.0,
            # RR 0.8 ～ 2.5
            "rr": _scale(rr, 0.8, 2.5) if rr is not None else 30.0,
            # 距觸發價越近越好：0% 滿分、5% 以上 0 分
            "distance": _scale(-abs(dist), -5.0, 0.0) if dist is not None else 30.0,
        }
        entry_q = sum(e[k] * w for k, w in C.FINAL_ENTRY_W.items())

        final = C.FINAL_W_MODEL * model_q + C.FINAL_W_ENTRY * entry_q
        r["final_score"] = round(_clamp100(final), 1)
        r["model_quality"] = round(_clamp100(model_q), 1)
        r["entry_quality"] = round(_clamp100(entry_q), 1)
        r["final_parts"] = {"model": {k: round(v, 1) for k, v in m.items()},
                            "entry": {k: round(v, 1) for k, v in e.items()}}
        # 進場狀態分層：🟢 可觀察 → 🟡 等待確認 → 🔴 不追價
        r["entry_tier"] = 0 if icon == "🟢" else (1 if icon == "🟡" else 2)


def sort_by_final(rows: list[dict]) -> list[dict]:
    """
    先依動能三層（🟢 強勢確認 → 🟡 潛力觀察 → 🔴 轉弱／追高），
    同一層內再依「綜合分數 60% ＋ 動能分數 40%」排序。

    **原本的 rank（模型名次）保持不變**，這只是最終進場排序。
    前 TOP_SLOTS 名優先只從 🟢 挑；不足才由 🟡 補入，並標記 top_fill，
    卡片上會明說「這一檔是因為強勢確認不足 5 檔才補進來的」。
    """
    def key(r):
        blend = (C.RANK_W_FINAL * r.get("final_score", 0)
                 + C.RANK_W_MOM * r.get("momentum_score", 0))
        r["rank_score"] = round(blend, 1)
        return (r.get("momentum_tier", 2), -blend)

    out = sorted(rows, key=key)
    green = sum(1 for r in out if r.get("momentum_tier") == 0)
    for i, r in enumerate(out, 1):
        r["final_rank"] = i
        # 前幾名裡不是 🟢 的，就是因為強勢確認檔數不足才補上來
        r["top_fill"] = bool(i <= C.TOP_SLOTS and r.get("momentum_tier", 2) != 0)
    if green < C.TOP_SLOTS:
        log.info("強勢確認只有 %d 檔，前 %d 名由潛力觀察補入 %d 檔",
                 green, C.TOP_SLOTS, min(C.TOP_SLOTS, len(out)) - green)
    return out


def add_edges(rows: list[dict]) -> None:
    """
    每檔補一句「為什麼是它、不是隔壁那檔」。

    問題背景：原本的 why 是由分項強度組出來的，前十名幾乎都會長成
    「進場位置好、趨勢結構強、距前高仍有 X%、追高風險低」，看起來像模板。
    這裡改成**跨標的比較**——只有跟同榜其他人比起來真的突出的那一點才會被寫出來，
    而且一定帶數字。找不到突出點就老實說「各項普通」。

    不動任何評分公式，只是換一句說明文字。
    """
    if not rows:
        return

    def med(key, default=0.0):
        vals = sorted(float(r.get(key) or 0) for r in rows)
        return vals[len(vals) // 2] if vals else default

    m_vol = med("vol_ratio", 1.0) or 1.0
    m_rr = med("rr_ratio", 1.0) or 1.0
    m_up = med("upside_pct", 1.0) or 1.0
    evs = [r.get("hist_expectancy") for r in rows if r.get("hist_expectancy") is not None]
    best_ev = max(evs) if evs else None

    # 「全榜之最」：只有真的第一名才拿得到這句話
    def extreme(key, biggest=True):
        vals = [(float(r.get(key) or 0), r["code"]) for r in rows]
        if not vals:
            return None
        return (max(vals) if biggest else min(vals))[1]

    top_vol = extreme("vol_ratio")
    top_rr = extreme("rr_ratio")
    low_rsi = extreme("rsi", biggest=False)
    top_up = extreme("upside_pct")
    best_entry = max(rows, key=lambda r: r["breakdown"]["entry"]["ratio"])["code"]
    low_risk = extreme("risk", biggest=False)

    # 第一天跑的時候全部都是新訊號，這句話就沒有鑑別力，乾脆不講
    marked = [r for r in rows if (r.get("mark") or {}).get("badge")]
    new_ratio = (sum(1 for r in marked if r["mark"]["badge"] == "🆕") / len(marked)) if marked else 1.0

    for r in rows:
        cands = []          # (優先度, 文字)
        mark = r.get("mark") or {}
        vol = float(r.get("vol_ratio") or 0)


        rr = float(r.get("rr_ratio") or 0)
        up = float(r.get("upside_pct") or 0)
        ev = r.get("hist_expectancy")
        streak = int(mark.get("streak") or 0)

        # 歷史期望值是全榜最高 → 這是最有力的差異
        if ev is not None and best_ev is not None and ev >= best_ev - 0.01 and ev > 0:
            cands.append((0, "同型態歷史期望值 %+.2f%%，是本榜最高的一組" % ev))

        # 量能明顯比同榜其他人大
        if m_vol > 0 and vol >= m_vol * 1.6 and vol >= 1.5:
            cands.append((1, "量能 %.1f 倍，是同榜中位數（%.1f 倍）的 %.1f 倍" %
                          (vol, m_vol, vol / m_vol)))

        # 風報比突出
        if rr >= m_rr * 1.5 and rr >= 1.5:
            cands.append((2, "風險報酬比 %.1f:1，同榜中位數只有 %.1f:1" % (rr, m_rr)))

        # 上檔空間突出
        if up >= m_up * 1.5 and up >= 5:
            cands.append((3, "上檔空間 +%.1f%%，同榜多數只有 +%.1f%%" % (up, m_up)))

        # 全榜之最
        if r["code"] == top_vol and vol >= 1.3:
            cands.append((1, "量能 %.1f 倍，是本榜最大的" % vol))
        if r["code"] == top_rr and rr >= 1.2:
            cands.append((2, "風險報酬比 %.1f:1，本榜最好" % rr))
        if r["code"] == low_rsi:
            cands.append((3, "RSI %.0f 是本榜最低，位置最靠近底部" % float(r.get("rsi") or 0)))
        if r["code"] == top_up and up >= 4:
            cands.append((3, "上檔空間 +%.1f%% 是本榜最大" % up))
        if r["code"] == best_entry:
            cands.append((3, "進場位置分數 %.2f 是本榜最高" % r["breakdown"]["entry"]["ratio"]))

        # 訊號狀態（大家都是新訊號時就不講，沒有鑑別力）
        if mark.get("badge") == "🆕" and new_ratio < 0.5:
            cands.append((6, "今日新進榜，前一次更新還沒出現"))
        elif streak >= C.STREAK_HOT:
            cands.append((4, "已連續入榜 %d 日，訊號沒有斷過" % streak))
        elif mark.get("rank_delta") and mark["rank_delta"] >= C.RANK_MOVE_MIN:
            cands.append((5, "名次比前一次上升 %d 名" % mark["rank_delta"]))

        # 型態本身就少見的話也算差異
        kinds = [x.get("kind") for x in rows]
        if r.get("kind") and kinds.count(r["kind"]) <= max(2, len(rows) // 10):
            cands.append((5, "本榜少數的「%s」型態，只有 %d 檔"
                          % (r["kind"], kinds.count(r["kind"]))))

        # 風險面的差異：只有真的全榜最低才講，否則大家都「風險低」等於沒講
        if r["code"] == low_risk and float(r.get("risk") or 0) <= 15:
            cands.append((6, "風險分數 %.0f 是本榜最低，過熱與追高項目都沒觸發"
                          % float(r.get("risk") or 0)))

        if cands:
            cands.sort(key=lambda x: x[0])
            r["edge"] = cands[0][1] + "。"
        else:
            # 沒有哪一項特別突出時，就直接把這檔的實際數字攤開，
            # 至少每檔看到的是自己的數字，不是同一句模板。
            r["edge"] = ("各項都在中段：RSI %.0f、量能 %.1f 倍、距前高 %.1f%%、風報 %.1f:1，"
                         "沒有明顯優於同榜其他標的的地方。"
                         % (float(r.get("rsi") or 0), vol,
                            float(r.get("pct_below_high20") or 0), rr))


def _data_date(hist_map: dict, now) -> str:
    """
    行情資料實際的交易日。取所有個股「最後一根 K 棒」的眾數——
    非交易日執行時，這個日期會停在最近一個有效交易日，不會跟著今天跑。
    """
    try:
        from collections import Counter
        last = [str(df.index[-1])[:10] for df in hist_map.values() if len(df)]
        if last:
            return Counter(last).most_common(1)[0][0]
    except Exception:
        pass
    return now.strftime("%Y-%m-%d")


def _write_latest_prices(snapshot, hist_map: dict, data_date: str, now,
                         names: dict | None = None) -> None:
    """
    產生 data/latest_prices.json —— 首頁與「我的交易」共用的唯一一份最新行情。

    來源全部重用這次已經抓好的資料，**不會多打任何一次 API**：
      1. 證交所當日快照（上市，最完整）
      2. data/history 快取的最後一根 K 棒（補上快照沒有的，例如上櫃股）
      3. 前一次的 latest_prices.json（這次沒出現的股票沿用最後有效價）

    每一檔都帶自己的 market_date，所以非交易日或某檔停牌時，
    畫面看到的是「那一檔最後一個有效交易日的收盤價」，
    不會用 0 或舊資料假裝成今天的價格。
    """
    out: dict[str, dict] = {}

    # --- 先把上一次的結果讀進來當底，行情失敗時才有東西可以沿用 ---
    path = ROOT / C.LATEST_PRICES_JSON
    try:
        if path.exists():
            old = json.loads(path.read_text(encoding="utf-8"))
            for k, v in (old.get("prices") or {}).items():
                if isinstance(v, dict) and v.get("price"):
                    out[k] = v
    except Exception as e:
        log.warning("舊的 latest_prices 讀取失敗（忽略）：%s", e)

    names = names or {}

    def put(code, name, price, prev, mdate):
        name = name or names.get(code) or out.get(code, {}).get("name", "")
        try:
            price = float(price)
        except Exception:
            return
        if not (price > 0):          # 0 或負數一律不寫入，寧可沿用舊價
            return
        try:
            prev = float(prev) if prev else None
        except Exception:
            prev = None
        rec = {"symbol": code, "name": name or "", "price": round(price, 2),
               "prev_close": round(prev, 2) if prev else None,
               "change_pct": round((price / prev - 1) * 100, 2) if prev else None,
               "market_date": mdate}
        out[code] = rec

    # --- 2) 歷史快取（先寫，之後會被當日快照覆蓋）---
    for code, hist in (hist_map or {}).items():
        try:
            if hist is None or len(hist) < 2:
                continue
            last = hist.iloc[-1]
            put(code, out.get(code, {}).get("name", ""), last["close"],
                hist["close"].iloc[-2], str(hist.index[-1])[:10])
        except Exception:
            continue

    # --- 1) 證交所當日快照（最準，覆蓋上面的）---
    n_snap = 0
    if snapshot is not None and len(snapshot):
        for _, r in snapshot.iterrows():
            put(r["code"], r.get("name", ""), r.get("close"),
                r.get("prev_close"), data_date)
            n_snap += 1

    payload = {
        "market_date": data_date,                       # 這次行情的交易日
        "updated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "count": len(out),
        "source": "證交所 OpenAPI 當日行情 ＋ data/history 快取",
        "prices": out,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                        encoding="utf-8")
        log.info("已寫出 %s：%d 檔（當日快照 %d、其餘沿用快取或上次結果）",
                 C.LATEST_PRICES_JSON, len(out), n_snap)
    except Exception as e:
        log.warning("latest_prices 寫入失敗：%s", e)


def _write_universe(core, extended: list, radar_rows: list,
                    stats: dict, trade_date: str) -> None:
    """輸出 data/universe.json 與 data/radar.json 給雷達頁與搜尋用。"""
    try:
        core_list = [{"symbol": r["code"], "name": r.get("name", ""),
                      "sector": r.get("sector", ""), "kind": r.get("kind", ""),
                      "market": r.get("market", "TWSE"), "is_core": True}
                     for _, r in core.iterrows()]
        (ROOT / C.UNIVERSE_JSON).parent.mkdir(parents=True, exist_ok=True)
        (ROOT / C.UNIVERSE_JSON).write_text(json.dumps(
            {"date": trade_date, "stats": stats,
             "core": core_list, "extended": extended},
            ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        (ROOT / C.RADAR_JSON).write_text(json.dumps(
            {"date": trade_date, "stats": stats, "rows": radar_rows},
            ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        log.info("已寫出 %s 與 %s（Core %d／Extended %d／雷達 %d）",
                 C.UNIVERSE_JSON, C.RADAR_JSON,
                 stats.get("core_total", 0), stats.get("extended_total", 0),
                 len(radar_rows))
    except Exception as e:
        log.warning("universe/radar 輸出失敗：%s", e)


def _scan_others(candidates: list, hist_map: dict, data_date: str) -> None:
    """
    非指定產業的股票用同一套評分排名，但結果只寫進 data/others.json，
    不會混進首頁排行。歷史資料一樣走共用快取，缺的才補抓。
    """
    if not candidates:
        return
    codes = [c["symbol"] for c in candidates]
    missing = [c for c in codes if c not in hist_map]
    if missing:
        log.info("其他產業：補抓 %d 檔缺少的歷史資料", len(missing))
        hist_map.update(fetch.fetch_history(missing))

    rows = []
    for c in candidates:
        hist = hist_map.get(c["symbol"])
        if hist is None:
            continue
        try:
            feats = indicators.compute_features(hist)
            if not feats:
                continue
            res = scoring.score_stock(feats)
            if res["score"] < C.OTHERS_MIN_SCORE:
                continue
            rows.append({
                "symbol": c["symbol"], "name": c.get("name", ""),
                "sector": c.get("sector", "其他"), "market": c.get("market", "TWSE"),
                "close": round(feats["close"], 2),
                "chg_pct": round(feats["chg_pct"], 2),
                "score": res["score"], "kind": res["kind"],
                "headline": res["headline"], "risk_level": res["risk_level"],
                "rsi": round(feats["rsi"], 1),
                "vol_ratio": round(feats["vol_ratio"], 2),
                "rr_ratio": res["rr_ratio"],
                "swing_low": res["swing_low"], "swing_high": res["swing_high"],
                "main_risk": res["main_risk"],
            })
        except Exception:
            continue

    rows.sort(key=lambda r: -r["score"])
    rows = rows[: C.OTHERS_TOP_N]
    try:
        (ROOT / C.OTHERS_JSON).parent.mkdir(parents=True, exist_ok=True)
        (ROOT / C.OTHERS_JSON).write_text(json.dumps(
            {"date": data_date, "scanned": len(candidates), "rows": rows},
            ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        log.info("其他產業：掃 %d 檔，%d 檔達標（>= %d 分）",
                 len(candidates), len(rows), C.OTHERS_MIN_SCORE)
    except Exception as e:
        log.warning("others.json 寫入失敗：%s", e)


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
