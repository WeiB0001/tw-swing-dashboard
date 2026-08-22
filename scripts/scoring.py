# -*- coding: utf-8 -*-
"""
scoring.py — V2 排名模型

目標：
1) 降低「位置漂亮但沒有確認」的假高分。
2) 增加趨勢、轉強、量價確認的重要性。
3) RR 使用 ATR 下檔地板，避免支撐貼太近造成虛高 RR。
4) 保持原本 build.py / render.py 需要的輸出欄位完全相容。

注意：
- score_stock() 產生的是「rule_score」，仍不是統計機率。
- build.py 會在存在 data/backtest.json 時，把 rule_score 校準成歷史 5 日勝率後再排名。
"""

from __future__ import annotations
import config as C

# V2 權重：確認 > 便宜位置
V2_WEIGHTS = {
    "entry": 18,
    "trend": 22,
    "reversal": 22,
    "volume": 20,
    "rr": 18,
}

# 沒有轉強/量價確認時，不允許只靠位置和 RR 排太前面
CONFIRM_BASE = 0.58
CONFIRM_REVERSAL = 0.24
CONFIRM_VOLUME = 0.18

# RR 的下檔至少視為多少個 ATR，避免「支撐很近」讓 RR 虛高
RR_ATR_FLOOR = 1.35


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _ramp_up(x, x0, x1):
    if x1 == x0:
        return 0.0
    return _clamp((x - x0) / (x1 - x0))


def _ramp_down(x, x_full, x_zero):
    if x_zero == x_full:
        return 0.0
    if x <= x_full:
        return 1.0
    if x >= x_zero:
        return 0.0
    return (x_zero - x) / (x_zero - x_full)


def _band(x, lo, hi, soft=0.35):
    width = max(hi - lo, 1e-9)
    if lo <= x <= hi:
        return 1.0
    dist = (lo - x) if x < lo else (x - hi)
    return _clamp(1.0 - dist / (width * soft))


def _valid_breakout(f: dict) -> bool:
    return (
        f["is_breakout"] > 0
        and f["vol_ratio"] >= C.VOL_SURGE_RATIO
        and f["close_pos_bar"] >= C.CLOSE_STRONG_POS
        and f["upper_shadow"] <= C.UPPER_SHADOW_BAD
        and f["rsi"] < C.RSI_HOT
    )


def _score_entry(f: dict) -> tuple[float, str]:
    """位置只回答『現在買的位置是否合理』，不再讓它單獨主導排名。"""
    breakout = _valid_breakout(f)
    pos = f["pos20"]

    if pos >= C.POS_BAD:
        s_pos = 0.05
    else:
        # V2 稍微縮窄中低檔甜蜜區
        ideal_hi = min(C.POS_IDEAL_HIGH, 0.50)
        s_pos = _band(pos, C.POS_IDEAL_LOW, ideal_hi, soft=0.75)

    if breakout:
        s_pos = max(s_pos, C.BREAKOUT_POS_FLOOR)

    room20 = f["pct_below_high20"]
    room60 = f["pct_below_high60"]
    s_room = _ramp_up(room20, C.ROOM_TO_HIGH_MIN, C.ROOM_TO_HIGH_GOOD)
    if breakout:
        s_room = max(
            s_room,
            _ramp_up(room60, C.ROOM_TO_HIGH_MIN, C.ROOM_TO_HIGH_GOOD * 1.5),
        )

    s_bias = _band(
        f["bias20"],
        max(C.BIAS20_IDEAL_LOW, -7.0),
        min(C.BIAS20_IDEAL_HIGH, 3.0),
        soft=0.8,
    )
    s_rsi = _band(
        f["rsi"],
        max(C.RSI_ENTRY_IDEAL_LOW, 40.0),
        min(C.RSI_ENTRY_IDEAL_HIGH, 64.0),
        soft=0.75,
    )

    strength = _clamp(0.30*s_pos + 0.30*s_room + 0.22*s_bias + 0.18*s_rsi)

    if breakout:
        note = f"有效突破，距 60 日高仍有 {room60:.1f}% 空間"
    elif pos >= C.POS_BAD:
        note = f"位於近 20 日區間高檔（{pos*100:.0f}%），追高風險偏高"
    elif strength >= 0.65:
        note = f"位置合理：20 日位置 {pos*100:.0f}%、距前高 {room20:.1f}%、MA20 乖離 {f['bias20']:+.1f}%"
    else:
        note = f"位置普通：20 日位置 {pos*100:.0f}%、距前高 {room20:.1f}%"
    return strength, note


def _score_trend(f: dict) -> tuple[float, str]:
    parts = [
        (0.22, 1.0 if f["close"] > f["ma20"] else 0.0),
        (0.20, 1.0 if f["ma5"] > f["ma10"] > f["ma20"] else
               (0.55 if f["ma5"] > f["ma10"] else 0.0)),
        (0.20, _ramp_up(f["ma20_slope"], -1.2, 2.0)),
        (0.15, _ramp_up(f["ma60_slope"], -1.2, 2.5)),
        (0.13, 1.0 if f["close"] > f["ma60"] else 0.0),
        (0.10, _ramp_up(f["ma10_slope"], -0.8, 1.8)),
    ]
    strength = _clamp(sum(w*s for w, s in parts))

    if strength >= 0.75:
        note = f"趨勢偏多，MA20 斜率 {f['ma20_slope']:+.1f}%"
    elif strength >= 0.45:
        note = f"趨勢中性偏多，MA20 斜率 {f['ma20_slope']:+.1f}%"
    else:
        note = f"趨勢偏弱，MA20 斜率 {f['ma20_slope']:+.1f}%"
    return strength, note


def _score_reversal(f: dict) -> tuple[float, str]:
    """V2 提高『真的開始轉強』的辨識力。"""
    hits, labels = [], []

    def add(weight, cond, label):
        hits.append(weight if cond else 0.0)
        if cond:
            labels.append(label)

    add(0.18, f["rsi"] > f["rsi_prev"] and f["rsi_min5"] < 48, "RSI 回升")
    add(0.16, f["close"] > f["ma5"] and f["prev_close"] <= f["ma5_prev"], "重新站回 MA5")
    add(0.14, f["ma5_slope"] > 0, "MA5 上彎")
    add(0.14, f["ma5"] > f["ma10"] and f["ma5_prev"] <= f["ma10_prev"], "MA5 金叉 MA10")
    add(0.16,
        (f["macd_hist"] > f["macd_hist_prev"] > 0) or
        (f["macd_hist"] > f["macd_hist_prev"] and
         f["macd_hist"] > f["macd_hist_prev3"]),
        "MACD 動能改善")
    add(0.12, f["vol_ratio"] >= 1.15 and f["is_up_day"] > 0, "帶量收紅")
    add(0.10, f["held_low5"] > 0 and f["is_new_low"] <= 0, "低點未再破")

    strength = _clamp(sum(hits))

    # 空頭排列中只有一兩個弱訊號，不給過高分
    falling = f["ma5"] < f["ma10"] < f["ma20"] and f["close"] < f["ma20"]
    if falling and strength < 0.55:
        strength *= 0.65

    note = "、".join(labels[:4]) if labels else "尚未出現明確轉強訊號"
    return strength, note


def _score_volume(f: dict) -> tuple[float, str]:
    r = f["vol_ratio"]

    # 低量不是直接死刑，但只能給極低分
    if r < 0.75:
        return 0.0, f"量能僅 {r:.2f} 倍，明顯低於 20 日均量"
    if r < 1.0:
        return 0.10 * _ramp_up(r, 0.75, 1.0), f"量能 {r:.2f} 倍，買盤確認不足"

    base = _ramp_up(r, 1.0, C.VOL_FULL_RATIO)
    q, why = 1.0, []
    up = f["is_up_day"] > 0
    strong_close = f["close_pos_bar"] >= C.CLOSE_STRONG_POS
    long_upper = f["upper_shadow"] >= C.UPPER_SHADOW_BAD

    if up and strong_close and not long_upper:
        q *= 1.10
        why.append("量增收在高檔")
    elif up and long_upper:
        q *= 0.50
        why.append("帶量但長上影")
    elif not up:
        q *= 0.20
        why.append("量增價跌")
        if f["close"] < f["ma5"] and f["close"] < f["ma10"]:
            q *= 0.45
            why.append("跌破短均線")
    else:
        q *= 0.72
        why.append("收盤未守高")

    if f["pos20"] > 0.85 and r > 3.0:
        q *= 0.42
        why.append("高檔爆量")
    elif r >= C.VOL_BLOWOFF_RATIO:
        q *= 0.78
        why.append("異常放量")

    if _valid_breakout(f):
        q = min(1.30, q * 1.25)
        why.append("突破有量確認")

    ud = f["vol_ud_ratio"]
    if ud > 1.15:
        q = min(1.30, q * 1.10)
        why.append("上漲日量優勢")
    elif 0 < ud < 0.85:
        q *= 0.82
        why.append("下跌日量較大")

    strength = _clamp(base * q)
    note = f"量能 {r:.2f} 倍" + ("；" + "、".join(why) if why else "")
    return strength, note


def _score_rr(f: dict) -> tuple[float, str]:
    """
    V2 重點：
    原始 downside 若小於 1.35 ATR，視為至少 1.35 ATR 的實務波動風險。
    這能避免『支撐就在腳下』讓 RR 被灌到非常漂亮。
    """
    up = max(float(f["upside_pct"]), 0.0)
    raw_down = max(float(f["downside_pct"]), 0.0)
    atr_down = max(float(f.get("atr_pct", 0.0)) * RR_ATR_FLOOR, 0.0)
    effective_down = max(raw_down, atr_down, 0.01)
    rr = up / effective_down

    if up <= 0:
        return 0.0, "上檔空間不足"
    s_rr = _ramp_up(rr, C.RR_MIN, C.RR_FULL)
    s_room = _ramp_up(up, C.MIN_UPSIDE_PCT, C.MIN_UPSIDE_PCT * 2.5)
    strength = _clamp(0.62*s_rr + 0.38*s_room)

    note = (
        f"上檔約 +{up:.1f}%；ATR 調整下檔約 −{effective_down:.1f}%；"
        f"有效 RR {rr:.1f}:1"
    )
    return strength, note


def _risk_score(f: dict) -> tuple[float, list[dict]]:
    breakout = _valid_breakout(f)
    items = []

    def add(key, level, label):
        level = _clamp(level)
        if level > 0.02:
            items.append({
                "key": key,
                "level": round(level, 3),
                "weight": C.RISK_WEIGHTS[key],
                "label": label,
            })

    add("rsi_hot", _ramp_up(f["rsi"], C.RSI_HOT, C.RSI_HOT_FULL),
        f"RSI {f['rsi']:.0f} 過熱")

    near = _ramp_down(f["pct_below_high20"], 0.0, C.NEAR_HIGH_PCT)
    if breakout:
        near *= 0.5
    add("near_high", near, f"距 20 日高僅 {f['pct_below_high20']:.1f}%")

    add("far_from_low",
        _ramp_up(f["pct_above_low20"], C.FAR_FROM_LOW_PCT, C.FAR_FROM_LOW_FULL),
        f"已自 20 日低點上漲 {f['pct_above_low20']:.0f}%")

    add("ma20_bias",
        _ramp_up(f["bias20"], C.BIAS20_WARN, C.BIAS20_FULL),
        f"MA20 正乖離 {f['bias20']:.1f}%")

    runup = _ramp_up(f["ret5"], C.RUNUP5_WARN, C.RUNUP5_FULL)
    if f["up_streak"] >= 4:
        runup = max(runup, 0.45)
    add("run_up", runup,
        f"近 5 日 {f['ret5']:+.1f}%（連 {int(f['up_streak'])} 日紅）")

    if f["is_breakout"] > 0 and not breakout:
        reasons = []
        if f["vol_ratio"] < C.VOL_SURGE_RATIO:
            reasons.append(f"量僅 {f['vol_ratio']:.1f} 倍")
        if f["upper_shadow"] > C.UPPER_SHADOW_BAD:
            reasons.append("長上影")
        if f["close_pos_bar"] < C.CLOSE_STRONG_POS:
            reasons.append("收盤未守高")
        if f["rsi"] >= C.RSI_HOT:
            reasons.append("RSI 過熱")
        add("fake_breakout", 0.55 + 0.12*len(reasons),
            "突破未確認（" + "、".join(reasons) + "）")

    falling = f["ma5"] < f["ma10"] < f["ma20"] and f["close"] < f["ma5"]
    if falling or f["is_new_low"] > 0:
        level = 0.55 + (0.25 if f["is_new_low"] > 0 else 0) + (0.20 if falling else 0)
        add("no_bottom", min(level, 1.0), "空頭排列或仍在破底，尚未止跌")

    if not items:
        return 0.0, []

    total_w = sum(C.RISK_WEIGHTS.values())
    score = sum(it["level"] * it["weight"] for it in items) / total_w * 100
    items.sort(key=lambda it: it["level"]*it["weight"], reverse=True)
    return _clamp(score, 0, 100), items


def _classify(f: dict, comp: dict, risk: float) -> str:
    if _valid_breakout(f):
        return "帶量突破"
    if f["ma5"] < f["ma10"] < f["ma20"] and f["close"] < f["ma5"]:
        return "弱勢未止跌"
    if f["pos20"] <= 0.48 and comp["reversal"][0] >= 0.50:
        return "低檔止跌轉強"
    if f["rsi"] >= C.RSI_HOT or f["pos20"] >= C.POS_BAD:
        return "強勢但已漲多"
    if comp["trend"][0] >= 0.62 and comp["reversal"][0] >= 0.30:
        return "多頭回檔"
    return "區間整理"


def _headline(kind: str, f: dict, comp: dict) -> str:
    return {
        "帶量突破": f"帶量突破，量 {f['vol_ratio']:.1f} 倍",
        "低檔止跌轉強": f"低檔轉強（RSI {f['rsi']:.0f}）",
        "多頭回檔": f"多頭回檔且有確認",
        "強勢但已漲多": f"強勢但位置偏高",
        "弱勢未止跌": "跌深但尚未止跌",
        "區間整理": "區間整理，等待確認",
    }.get(kind, "訊號中性")


def _why(kind: str, f: dict, comp: dict, risk_items: list) -> str:
    label = {
        "entry": "進場位置",
        "trend": "趨勢",
        "reversal": "轉強",
        "volume": "量價",
        "rr": "有效RR",
    }
    ranked = sorted(
        comp.items(),
        key=lambda kv: kv[1][0] * V2_WEIGHTS[kv[0]],
        reverse=True,
    )
    tops = [label[k] for k, (s, _) in ranked[:2] if s >= 0.45]
    bits = ["、".join(tops) if tops else "整體訊號仍弱"]

    if comp["reversal"][0] < 0.25 and comp["volume"][0] < 0.20:
        bits.append("但缺乏轉強與量價確認")
    elif comp["reversal"][0] < 0.25:
        bits.append("轉強確認不足")
    elif comp["volume"][0] < 0.20:
        bits.append("量價確認不足")
    else:
        bits.append("確認訊號較完整")

    if risk_items and risk_items[0]["level"] >= 0.5:
        bits.append("主要風險：" + risk_items[0]["label"])
    return "；".join(bits) + "。"


def score_stock(f: dict) -> dict:
    comp = {
        "entry": _score_entry(f),
        "trend": _score_trend(f),
        "reversal": _score_reversal(f),
        "volume": _score_volume(f),
        "rr": _score_rr(f),
    }

    breakdown, reasons = {}, []
    raw = 0.0
    max_raw = float(sum(V2_WEIGHTS.values()))
    for key, (strength, note) in comp.items():
        w = V2_WEIGHTS[key]
        pts = strength * w
        raw += pts
        breakdown[key] = {
            "points": round(pts, 1),
            "max": w,
            "ratio": round(strength, 3),
        }
        if note:
            reasons.append(note)

    opportunity = raw / max_raw * 100 if max_raw else 0.0
    risk, risk_items = _risk_score(f)

    # 確認係數：沒有 reversal / volume 時，不能只靠「位置+RR」高居榜首
    confirmation = (
        CONFIRM_BASE
        + CONFIRM_REVERSAL * comp["reversal"][0]
        + CONFIRM_VOLUME * comp["volume"][0]
    )
    confirmation = _clamp(confirmation, 0.50, 1.0)

    rule_score = opportunity * confirmation
    rule_score *= (1 - C.RISK_MAX_CUT * risk / 100)
    rule_score = _clamp(rule_score, 0, 100)

    def stars(ratio):
        n = int(round(_clamp(ratio) * 5))
        return "★"*n + "☆"*(5-n)

    risk_level = (
        "低" if risk < 12 else
        "中低" if risk < 25 else
        "中" if risk < 40 else
        "偏高" if risk < 60 else "高"
    )

    up = max(float(f["upside_pct"]), 0.0)
    kind = _classify(f, comp, risk)
    main_risk = risk_items[0]["label"] if risk_items else "目前未偵測到明顯追高或破底風險"

    effective_down = max(
        float(f["downside_pct"]),
        float(f.get("atr_pct", 0.0)) * RR_ATR_FLOOR,
        0.01,
    )
    effective_rr = up / effective_down if effective_down else 0.0

    return {
        "score": round(rule_score, 1),       # build.py 有回測時會再校準
        "rule_score": round(rule_score, 1),
        "opportunity": round(opportunity, 1),
        "confirmation": round(confirmation * 100, 1),
        "risk": round(risk, 1),
        "breakdown": breakdown,
        "risk_items": risk_items,
        "reasons": reasons,
        "stars": {
            "trend": stars(comp["trend"][0]),
            "entry": stars(comp["entry"][0]),
            "volume": stars(comp["volume"][0]),
            "reversal": stars(comp["reversal"][0]),
            "risk": stars(risk / 100),
        },
        "risk_level": risk_level,
        "kind": kind,
        "headline": _headline(kind, f, comp),
        "why": _why(kind, f, comp, risk_items),
        "main_risk": main_risk,
        "swing_low": round(up * 0.45, 1),
        "swing_high": round(up, 1),
        "flags": [it["label"] for it in risk_items[:2] if it["level"] >= 0.5],
        "rr_ratio": round(effective_rr, 2),
        "downside_pct": round(effective_down, 2),
        "upside_pct": round(up, 2),
    }


def sort_key(row: dict):
    """
    V2：
    score 若已被 build.py 校準，就直接代表較接近歷史 5 日勝率的排序分數。
    同分時依平均報酬、確認度、風險、量價、轉強排序。
    """
    hist_avg = row.get("hist_avg_return")
    hist_avg = float(hist_avg) if hist_avg is not None else -999.0
    return (
        -row.get("score", 0),
        -hist_avg,
        -row.get("confirmation", 0),
        row.get("risk", 100),
        -row.get("breakdown", {}).get("volume", {}).get("ratio", 0),
        -row.get("breakdown", {}).get("reversal", {}).get("ratio", 0),
        -row.get("rule_score", row.get("score", 0)),
    )
