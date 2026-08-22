# -*- coding: utf-8 -*-
"""
scoring.py — 價差機會分數（0～100）

核心思路：超賣反彈 + 量能確認。
分數由五個成分加總，每個成分先算出 0～1 的「強度」，再乘上 config.WEIGHTS 的權重。
最後套用風險折扣（追高、異常爆量、完全沒反彈）。

要調整訊號邏輯，通常有兩種做法：
  A. 只改門檻／權重  → 改 config.py 就好
  B. 改判斷方式      → 改本檔案中對應的 _score_xxx 函式
每個成分都獨立回傳 (強度, 理由文字)，加新成分只要照同樣格式寫一個函式，
再加進 score_stock() 的 parts 清單即可。
"""

from __future__ import annotations

import config as C


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _lerp_down(x: float, x_full: float, x_zero: float) -> float:
    """x 越小分數越高：x <= x_full 給 1，x >= x_zero 給 0，中間線性內插。"""
    if x <= x_full:
        return 1.0
    if x >= x_zero:
        return 0.0
    return (x_zero - x) / (x_zero - x_full)


# ---------------------------------------------------------------------------
# 成分 1：超賣反彈（權重最高）
#   RSI 越低越好，但一定要「收盤站上 MA5 或 MA10」才算真的出現反彈跡象。
# ---------------------------------------------------------------------------
def _score_oversold(f: dict) -> tuple[float, str]:
    rsi = f["rsi"]
    base = _lerp_down(rsi, C.RSI_DEEP_OVERSOLD, C.RSI_MAX_INTEREST)
    if base <= 0:
        return 0.0, ""

    above_ma5 = f["close"] > f["ma5"]
    above_ma10 = f["close"] > f["ma10"]

    # 文案分級：真的超賣才講「超賣」，免得 RSI 46 也被寫成偏低
    level = "深度超賣" if rsi <= C.RSI_DEEP_OVERSOLD else ("超賣" if rsi < C.RSI_OVERSOLD else "中性偏弱")

    if above_ma5 or above_ma10:
        which = "MA5" if above_ma5 else "MA10"
        # RSI 由下往上翹（今天比昨天高）再加一點強度
        turning = f["rsi"] > f["rsi_prev"]
        strength = _clamp(base * (1.0 if turning else 0.88))
        tag = "、指標止跌翻揚" if turning else ""
        note = f"RSI {rsi:.0f} 位於{level}區{tag}，收盤已站回 {which}"
    else:
        # 還在均線下方＝只有超賣、沒有反彈訊號，只給 35% 強度
        strength = base * 0.35
        note = f"RSI {rsi:.0f} 位於{level}區，但收盤仍在 MA5／MA10 之下，尚未轉強"

    return strength, note


# ---------------------------------------------------------------------------
# 成分 2：量能確認
#   量增才代表有人接。1.3 倍開始加分、2.2 倍給滿分，超過 5 倍視為異常。
# ---------------------------------------------------------------------------
def _score_volume(f: dict) -> tuple[float, str]:
    r = f["vol_ratio"]
    if r < 1.0:
        return 0.0, f"量能 {r:.1f} 倍，低於 20 日均量，買盤不足"

    span = max(C.VOL_FULL_RATIO - C.VOL_SURGE_RATIO, 0.1)
    if r < C.VOL_SURGE_RATIO:
        strength = 0.25 * (r - 1.0) / max(C.VOL_SURGE_RATIO - 1.0, 0.1)
        note = f"量能 {r:.1f} 倍，略高於均量"
    else:
        strength = _clamp(0.25 + 0.75 * (r - C.VOL_SURGE_RATIO) / span)
        note = f"量能放大至 20 日均量 {r:.1f} 倍"

    if r >= C.VOL_BLOWOFF_RATIO:
        note = f"量能暴增 {r:.1f} 倍（異常放量，留意是否為出貨）"

    return strength, note


# ---------------------------------------------------------------------------
# 成分 3：低檔位階 + 已見反彈
#   收盤落在近 20 日區間下緣，且今天收紅（或收盤高於開盤）。
# ---------------------------------------------------------------------------
def _score_near_low(f: dict) -> tuple[float, str]:
    pos = f["pos_in_range20"]
    if pos > C.NEAR_LOW_ZONE:
        return 0.0, ""

    base = _clamp(1.0 - pos / max(C.NEAR_LOW_ZONE, 0.01))
    rebounding = (f["close"] > f["prev_close"]) or (f["close"] > f["open"])

    if rebounding:
        strength = base
        note = (f"股價位於近 20 日區間低檔（距低點 {f['pct_above_low20']:.1f}%），"
                f"今日已出現反彈")
    else:
        strength = base * 0.45
        note = f"股價貼近近 20 日低點（距低點 {f['pct_above_low20']:.1f}%），但今日尚未止跌"

    return strength, note


# ---------------------------------------------------------------------------
# 成分 4：均線轉強
#   優先序：MA5 金叉 MA10 > 今日剛站上 MA10 > 今日剛站上 MA5 > 已在均線上
# ---------------------------------------------------------------------------
def _score_ma_reclaim(f: dict) -> tuple[float, str]:
    cross_up = f["ma5_prev"] <= f["ma10_prev"] and f["ma5"] > f["ma10"]
    reclaim_ma10 = f["prev_close"] <= f["ma10_prev"] and f["close"] > f["ma10"]
    reclaim_ma5 = f["prev_close"] <= f["ma5_prev"] and f["close"] > f["ma5"]

    if cross_up:
        return 1.0, "MA5 向上金叉 MA10，短線轉強"
    if reclaim_ma10:
        return 0.9, "收盤今日一舉站回 MA10"
    if reclaim_ma5:
        return 0.8, "收盤今日重新站回 MA5"
    if f["close"] > f["ma5"] > f["ma10"]:
        return 0.6, "均線短多排列（收盤 > MA5 > MA10）"
    if f["close"] > f["ma5"]:
        return 0.4, "收盤位於 MA5 之上"
    if f["close"] > f["ma10"]:
        return 0.35, "收盤位於 MA10 之上"
    return 0.0, ""


# ---------------------------------------------------------------------------
# 成分 5：結構品質
#   中期趨勢（MA20 斜率）沒壞、回檔深度合理的，反彈成功率通常較高。
# ---------------------------------------------------------------------------
def _score_structure(f: dict) -> tuple[float, str]:
    strength, notes = 0.0, []

    if f["ma20_slope_pct"] >= 0:
        strength += 0.55
        notes.append("MA20 仍走平或向上，中期結構未破壞")
    elif f["ma20_slope_pct"] >= -1.5:
        strength += 0.25
        notes.append("MA20 微幅下彎，中期趨勢轉弱但幅度有限")

    # 距 MA20 的回檔深度：跌太深通常是有基本面問題，跌太淺則反彈空間小
    depth = (f["ma20"] / f["close"] - 1) * 100 if f["close"] else 0
    if 1.0 <= depth <= 12.0:
        strength += 0.45
        notes.append(f"距 MA20 約 {depth:.1f}%，反彈空間合理")
    elif depth > 12.0:
        strength += 0.15
        notes.append(f"距 MA20 達 {depth:.1f}%，乖離偏大")

    return _clamp(strength), "；".join(notes)


# ---------------------------------------------------------------------------
# 主函式
# ---------------------------------------------------------------------------
def score_stock(f: dict) -> dict:
    """
    輸入 indicators.compute_features() 的結果，
    回傳 {score, breakdown, reasons, headline, flags}。
    """
    parts = {
        "oversold":   _score_oversold(f),
        "volume":     _score_volume(f),
        "near_low":   _score_near_low(f),
        "ma_reclaim": _score_ma_reclaim(f),
        "structure":  _score_structure(f),
    }

    breakdown, reasons, raw = {}, [], 0.0
    for key, (strength, note) in parts.items():
        pts = strength * C.WEIGHTS[key]
        raw += pts
        breakdown[key] = {
            "points": round(pts, 1),
            "max": C.WEIGHTS[key],
            "ratio": round(strength, 3),
        }
        if note:
            reasons.append(note)

    # --- 風險折扣 ---
    flags, mult = [], 1.0
    if f["chg_pct"] >= C.CHASE_HIGH_PCT:
        mult *= C.CHASE_HIGH_PENALTY
        flags.append("當日漲幅過大，追高風險")
    if f["vol_ratio"] >= C.VOL_BLOWOFF_RATIO:
        mult *= C.BLOWOFF_PENALTY
        flags.append("異常爆量")
    if f["close"] < f["ma5"] and f["close"] < f["ma10"]:
        mult *= C.BELOW_ALL_MA_PENALTY
        flags.append("仍在 MA5／MA10 之下")

    score = round(_clamp(raw * mult, 0, 100), 1)

    return {
        "score": score,
        "breakdown": breakdown,
        "reasons": reasons,
        "headline": _headline(parts, f),
        "flags": flags,
    }


def _headline(parts: dict, f: dict) -> str:
    """挑出強度最高的成分，濃縮成一句 12 字內的短理由，給表格用。"""
    best = max(parts.items(), key=lambda kv: kv[1][0] * C.WEIGHTS[kv[0]])
    key, (strength, _) = best
    if strength <= 0:
        return "訊號偏弱"
    return {
        "oversold":   f"超賣反彈（RSI {f['rsi']:.0f}）",
        "volume":     f"量能放大 {f['vol_ratio']:.1f} 倍",
        "near_low":   "低檔止跌反彈",
        "ma_reclaim": "均線轉強",
        "structure":  "回檔至支撐區",
    }[key]
