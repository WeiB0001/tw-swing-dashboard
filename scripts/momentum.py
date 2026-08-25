# -*- coding: utf-8 -*-
"""
momentum.py — 明日強勢預測（獨立於首頁排行）

**完全不碰 scoring、EV、backtest、walk-forward。** 這裡回答的是另一個問題：
「今天漲很多的股票，明天還會不會續強？」

做法（純規則 + 歷史統計，沒有 AI／sklearn）：
  1. 從當日全市場行情取出漲幅前段、且屬於科技／金融／ETF 的股票
  2. 把每天的強勢榜存進 data/strong_history.json
  3. 用既有的 data/history 快取回測：歷史上「符合同樣條件」的強勢股，
     隔日與未來 3 日的報酬如何
  4. 依統計結果 + 今日型態算出強勢分數，排出明日候選

**禁止 look-ahead**：特徵一律取自第 t 日收盤，報酬從 t+1 開盤才開始算
（t 日收盤價在收盤後已經買不到）。

**不額外下載資料**：只用 data/history 已經有的個股。
資料抓不到就用既有快取，不會讓流程中斷。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import indicators

log = logging.getLogger("momentum")
ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 分級（回測與預測共用同一套，否則查表會對不上）
# ---------------------------------------------------------------------------
def chg_bucket(x: float) -> str:
    if x >= C.MOM_EXTREME_CHG:
        return "極端"
    if x >= 7.0:
        return "7-9.5"
    if x >= 5.0:
        return "5-7"
    return "3-5"


def vol_bucket(x: float) -> str:
    if x >= 2.5:
        return "爆量"
    if x >= 1.5:
        return "放量"
    return "普通"


def pos_bucket(x: float) -> str:
    if x >= 0.8:
        return "收高"
    if x >= 0.5:
        return "收中"
    return "收低"


def _f(x, d=0.0):
    try:
        v = float(x)
        return v if np.isfinite(v) else d
    except Exception:
        return d


def _calib(wins: int, n: int) -> float:
    """跟回測同一套平滑：(勝場+10)/(樣本+20)，避免小樣本出現假高勝率。"""
    return (wins + C.SMOOTH_WINS) / (n + C.SMOOTH_N)


# ---------------------------------------------------------------------------
# 1) 歷史統計：強勢股隔日還強不強
# ---------------------------------------------------------------------------
def build_stats(hist_map: dict[str, pd.DataFrame]) -> dict:
    """
    掃過所有快取裡的個股，找出歷史上的「強勢日」，統計隔日與 3 日報酬。
    強勢日定義：當日漲幅 >= MOM_MIN_CHG 且成交金額達門檻。
    同一檔在 5 個交易日內只採樣一次，避免同一段行情被算成多個樣本。
    """
    samples = []
    for code, hist in hist_map.items():
        if hist is None or len(hist) < C.MIN_BARS + 5:
            continue
        try:
            df = indicators.compute_frame(hist)
        except Exception:
            continue

        last_i = -99
        n = len(df)
        for i in range(C.MIN_BARS, n - 4):        # 尾端留 4 根：t+1 進場、t+3 出場
            row = df.iloc[i]
            chg = _f(row.get("ret1"))
            if chg < C.MOM_MIN_CHG:
                continue
            if i - last_i < C.SIGNAL_COOLDOWN_DAYS:
                continue
            close = _f(row.get("close"))
            vol_ma = _f(row.get("vol_ma20"))
            if close <= 0 or close * vol_ma < C.MOM_MIN_TURNOVER:
                continue
            last_i = i

            # --- 報酬：t+1 開盤進場，禁止用 t 日收盤當進場價 ---
            entry = _f(df["open"].iloc[i + 1])
            if entry <= 0:
                continue
            r1 = (_f(df["close"].iloc[i + 1]) / entry - 1) * 100
            r3 = (_f(df["close"].iloc[i + 3]) / entry - 1) * 100

            samples.append({
                "chg": chg_bucket(chg),
                "vol": vol_bucket(_f(row.get("vol_ratio"))),
                "pos": pos_bucket(_f(row.get("close_pos_bar"))),
                "r1": r1, "r3": r3,
            })

    if not samples:
        return {"total": 0, "keys": {}, "by_chg": {}, "overall": None}

    def pack(sub):
        a1 = np.array([s["r1"] for s in sub])
        a3 = np.array([s["r3"] for s in sub])
        w = int((a1 > 0).sum())
        return {
            "samples": len(sub),
            "win_rate": round(float(w / len(sub) * 100), 1),
            "calibrated_win_rate": round(_calib(w, len(sub)) * 100, 1),
            "avg_r1": round(float(a1.mean()), 2),
            "median_r1": round(float(np.median(a1)), 2),
            "avg_r3": round(float(a3.mean()), 2),
            "worst_r1": round(float(a1.min()), 2),
        }

    keys, by_chg = {}, {}
    for s in samples:
        keys.setdefault("%s|%s|%s" % (s["chg"], s["vol"], s["pos"]), []).append(s)
        by_chg.setdefault(s["chg"], []).append(s)

    return {
        "total": len(samples),
        "keys": {k: pack(v) for k, v in keys.items()},
        "by_chg": {k: pack(v) for k, v in by_chg.items()},
        "overall": pack(samples),
        "entry_rule": "第 t 日收盤選股，t+1 開盤進場，t+1 / t+3 收盤計算報酬",
    }


def lookup(stats: dict, chg: float, vol: float, pos: float) -> tuple[dict | None, str]:
    """
    查歷史統計，樣本不足就往上退一層。
    退到最後仍不足就回 (None, "樣本不足")，**不會生一個假勝率出來**。
    """
    k = "%s|%s|%s" % (chg_bucket(chg), vol_bucket(vol), pos_bucket(pos))
    d = (stats.get("keys") or {}).get(k)
    if d and d["samples"] >= C.MOM_MIN_SAMPLES:
        return d, "漲幅＋量能＋收盤位置"
    d = (stats.get("by_chg") or {}).get(chg_bucket(chg))
    if d and d["samples"] >= C.MOM_MIN_SAMPLES:
        return d, "漲幅級距"
    d = stats.get("overall")
    if d and d["samples"] >= C.MOM_MIN_SAMPLES:
        return d, "全部強勢日"
    return None, "樣本不足"


# ---------------------------------------------------------------------------
# 2) 今日強勢股 → 明日候選
# ---------------------------------------------------------------------------
def score_one(f: dict) -> tuple[float, list[str], list[str]]:
    """強勢分數 0～100。加分項與扣分項都回傳，讓使用者看得到理由。"""
    chg = _f(f.get("chg_pct"))
    vol = _f(f.get("vol_ratio"))
    pos = _f(f.get("close_pos_bar"))
    upper = _f(f.get("upper_shadow"))
    close = _f(f.get("close"))
    ma5, ma20 = _f(f.get("ma5")), _f(f.get("ma20"))
    to_h20 = _f(f.get("pct_below_high20"), 99)
    to_h60 = _f(f.get("pct_below_high60"), 99)
    turnover = close * _f(f.get("volume"))

    pros, cons = [], []
    score = 0.0

    # 漲幅：強但不極端最好。極端漲停附近反而扣分
    if C.MOM_MIN_CHG <= chg < 5:
        score += 18; pros.append("今日 +%.1f%%，強度適中" % chg)
    elif 5 <= chg < 7:
        score += 22; pros.append("今日 +%.1f%%" % chg)
    elif 7 <= chg < C.MOM_EXTREME_CHG:
        score += 16; pros.append("今日 +%.1f%%，偏強" % chg)
    else:
        score += 4; cons.append("今日 +%.1f%% 過度極端，隔日容易開高走低" % chg)

    # 量能
    if vol >= 2.5:
        score += 18; pros.append("量能 %.1f 倍" % vol)
    elif vol >= 1.5:
        score += 20; pros.append("量能 %.1f 倍" % vol)
    elif vol >= 1.2:
        score += 12; pros.append("量能 %.1f 倍" % vol)
    else:
        cons.append("量能僅 %.1f 倍，缺乏認同" % vol)

    # 成交金額：太小的隔日容易買不到也賣不掉
    if turnover >= C.MOM_MIN_TURNOVER * 5:
        score += 10; pros.append("成交金額 %.1f 億" % (turnover / 1e8))
    elif turnover >= C.MOM_MIN_TURNOVER:
        score += 6
    else:
        cons.append("成交金額偏低，流動性風險高")

    # 收盤位置：收在最高附近代表買盤守到最後
    if pos >= 0.85:
        score += 18; pros.append("收盤在當日高點附近（%.0f%%）" % (pos * 100))
    elif pos >= 0.7:
        score += 12; pros.append("收盤位置 %.0f%%" % (pos * 100))
    elif pos >= 0.5:
        score += 5
    else:
        cons.append("收盤只在當日振幅 %.0f%% 位置，尾盤被壓" % (pos * 100))

    # 長上影 + 爆量 = 追價力道有疑慮
    if upper >= 0.35 and vol >= 2.0:
        score -= 12; cons.append("爆量且留 %.0f%% 上影線" % (upper * 100))
    elif upper >= 0.35:
        score -= 6; cons.append("留 %.0f%% 上影線" % (upper * 100))

    # 均線趨勢
    if close > ma5 > ma20 > 0:
        score += 12; pros.append("MA5 站上 MA20，趨勢向上")
    elif close > ma5 > 0:
        score += 6
    else:
        cons.append("尚未站上短均線")

    # 位階：接近或突破前高
    if to_h20 <= 0.5:
        score += 14; pros.append("已站上近 20 日高點")
    elif to_h20 <= 3:
        score += 9; pros.append("距 20 日高僅 %.1f%%" % to_h20)
    if to_h60 <= 0.5:
        score += 6; pros.append("同時突破 60 日高點")

    return max(0.0, min(100.0, score)), pros, cons


def build_forecast(snapshot: pd.DataFrame, hist_map: dict, universe: list[dict],
                   trade_date: str) -> dict:
    """
    今日強勢股 → 明日候選排名。
    只看 Extended 裡的科技／金融／ETF，且只用 data/history 已有的個股，
    不會為了這一頁多下載任何歷史資料。
    """
    meta = {u["symbol"]: u for u in (universe or [])}
    stats = build_stats(hist_map)
    log.info("強勢日歷史樣本：%d 筆", stats.get("total", 0))

    rows = []
    if snapshot is not None and len(snapshot):
        for _, s in snapshot.iterrows():
            code = s["code"]
            if code not in meta or code not in hist_map:
                continue                      # 不在關注範圍、或快取沒有 → 跳過
            chg = _f(s.get("chg_pct"))
            if chg < C.MOM_MIN_CHG:
                continue
            try:
                df = indicators.compute_frame(hist_map[code])
                f = indicators.features_at(df, -1)
            except Exception:
                continue
            if not f:
                continue
            f = dict(f)
            f["chg_pct"] = chg                # 以當日快照為準

            score, pros, cons = score_one(f)
            hit, source = lookup(stats, chg, _f(f.get("vol_ratio")),
                                 _f(f.get("close_pos_bar")))

            # 歷史統計只做微調，不主導分數：勝率每高於 50% 一個百分點加 0.4 分
            if hit:
                score = max(0.0, min(100.0, score +
                                     (hit["calibrated_win_rate"] - 50) * 0.4))

            rows.append({
                "symbol": code,
                "name": meta[code].get("name") or s.get("name", ""),
                "sector": meta[code].get("sector", ""),
                "kind": meta[code].get("kind", ""),
                "market": meta[code].get("market", "TWSE"),
                "close": round(_f(f.get("close")), 2),
                "chg_pct": round(chg, 2),
                "vol_ratio": round(_f(f.get("vol_ratio")), 2),
                "close_pos": round(_f(f.get("close_pos_bar")) * 100),
                "upper_shadow": round(_f(f.get("upper_shadow")) * 100),
                "turnover": int(_f(f.get("close")) * _f(f.get("volume"))),
                "rsi": round(_f(f.get("rsi")), 1),
                "ma5": round(_f(f.get("ma5")), 2),
                "ma20": round(_f(f.get("ma20")), 2),
                "to_high20": round(_f(f.get("pct_below_high20")), 1),
                "to_high60": round(_f(f.get("pct_below_high60")), 1),
                "score": round(score, 1),
                "pros": pros, "cons": cons,
                "hist": hit, "hist_source": source,
            })

    rows.sort(key=lambda r: -r["score"])
    rows = rows[: C.MOM_TOP_N]

    save_strong_history(rows, trade_date)
    return {
        "date": trade_date,
        "generated_at": trade_date,
        "stats": {"total_samples": stats.get("total", 0),
                  "overall": stats.get("overall"),
                  "entry_rule": stats.get("entry_rule", ""),
                  "min_samples": C.MOM_MIN_SAMPLES},
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# 3) 每日強勢榜存檔
# ---------------------------------------------------------------------------
def save_strong_history(rows: list[dict], trade_date: str) -> None:
    """同一天只留一筆；檔案壞掉就重新開始，不讓流程中斷。"""
    path = ROOT / C.STRONG_HISTORY_JSON
    try:
        old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        days = old.get("days", []) if isinstance(old, dict) else []
    except Exception:
        days = []
    days = [d for d in days if isinstance(d, dict) and d.get("date") != trade_date]
    days.append({
        "date": trade_date,
        "entries": [{
            "symbol": r["symbol"], "date": trade_date, "return_pct": r["chg_pct"],
            "volume_ratio": r["vol_ratio"], "turnover": r["turnover"],
            "rsi": r["rsi"], "ma5": r["ma5"], "ma20": r["ma20"],
            "to_high20": r["to_high20"], "to_high60": r["to_high60"],
            "score": r["score"],
        } for r in rows],
    })
    days = days[-C.STRONG_HISTORY_DAYS:]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"days": days}, ensure_ascii=False,
                                   separators=(",", ":")), encoding="utf-8")
    except Exception as e:
        log.warning("強勢榜歷史寫入失敗：%s", e)
