# -*- coding: utf-8 -*-
"""
scoring.py — 獲利可能性分數（Profit Probability Score, 0～100）

回答的問題不是「哪檔技術指標最漂亮」，而是：
    「以目前價格進場，在合理的 Swing 持有期間內，哪一檔最可能產生正報酬，
      而且風險報酬比最好？」

三層結構：
    1. 機會分數 Opportunity（0～100）
       進場位置 25 ＋ 趨勢品質 20 ＋ 轉強確認 15 ＋ 量價品質 15 ＋ 風險報酬 15
       （加權後正規化）
    2. 風險分數 Risk（0～100）
       追高、過熱、乖離過大、假突破、跌深未止跌
    3. 獲利可能性 PPS = Opportunity × (1 − RISK_MAX_CUT × Risk/100)

關鍵設計：**趨勢強度與進場位置是分開的兩件事。**
RSI 90、貼著 20 日高點的股票，趨勢分數會很高，但位置分數很低、風險分數很高，
最後不會自動排第一。這正是舊版模型最大的問題。

⚠️ 這是**技術分數**，不是獲利機率。它只描述「當下的價量型態長什麼樣」。
   首頁排名以 scripts/backtest.py 統計出的歷史平滑勝率為第一順位，
   技術分數只用來分層（同型態、同分數級距的訊號歸為一類去查歷史勝率）。

要調整邏輯：改門檻／權重 → config.py；改判斷方式 → 本檔的 _score_xxx 函式。
每個成分函式都獨立回傳 (強度 0~1, 說明文字)，格式一致，方便新增或替換。
"""

from __future__ import annotations

import config as C


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _ramp_up(x, x0, x1):
    """x 從 x0 升到 x1，分數 0 → 1。"""
    if x1 == x0:
        return 0.0
    return _clamp((x - x0) / (x1 - x0))


def _ramp_down(x, x_full, x_zero):
    """x 越小分數越高：x <= x_full 給 1，x >= x_zero 給 0。"""
    if x_zero == x_full:
        return 0.0
    if x <= x_full:
        return 1.0
    if x >= x_zero:
        return 0.0
    return (x_zero - x) / (x_zero - x_full)


def _band(x, lo, hi, soft=0.35):
    """
    落在 [lo, hi] 區間內給 1 分，超出區間依距離線性衰減。
    soft 是「衰減到 0 需要的區間寬度倍數」。
    """
    width = max(hi - lo, 1e-9)
    if lo <= x <= hi:
        return 1.0
    dist = (lo - x) if x < lo else (x - hi)
    return _clamp(1.0 - dist / (width * soft))


def _valid_breakout(f: dict) -> bool:
    """
    有效突破：突破前 20 日高 + 量能確認 + 收盤守在上緣 + 沒有過熱。
    這是「突破型賺錢機會」，不能用追高的標準去殺它（規格 Case 3）。
    """
    return (
        f["is_breakout"] > 0
        and f["vol_ratio"] >= C.VOL_SURGE_RATIO
        and f["close_pos_bar"] >= C.CLOSE_STRONG_POS
        and f["upper_shadow"] <= C.UPPER_SHADOW_BAD
        and f["rsi"] < C.RSI_HOT
    )


# ---------------------------------------------------------------------------
# ① 進場位置品質（權重 25，最高）
#    問的是：「還有上漲空間，而且現在不是追高的位置嗎？」
# ---------------------------------------------------------------------------
def _score_entry(f: dict) -> tuple[float, str]:
    breakout = _valid_breakout(f)

    # a) 在 20 日區間的位置：理想是中下緣
    pos = f["pos20"]
    if pos >= C.POS_BAD:
        s_pos = 0.05
    else:
        s_pos = _band(pos, C.POS_IDEAL_LOW, C.POS_IDEAL_HIGH, soft=0.9)
    if breakout:
        # 有效突破本來就會貼著高點，位置分數給保底
        s_pos = max(s_pos, C.BREAKOUT_POS_FLOOR)

    # b) 距近期高點還剩多少空間（沒空間就沒有價差可賺）
    room20 = f["pct_below_high20"]
    room60 = f["pct_below_high60"]
    s_room = _ramp_up(room20, C.ROOM_TO_HIGH_MIN, C.ROOM_TO_HIGH_GOOD)
    if breakout:
        # 突破 20 日高之後，看的是距 60 日高還有多少空間
        s_room = max(s_room, _ramp_up(room60, C.ROOM_TO_HIGH_MIN, C.ROOM_TO_HIGH_GOOD * 1.5))

    # c) 距 MA20 乖離：太遠＝追高，太深＝可能有基本面問題
    s_bias = _band(f["bias20"], C.BIAS20_IDEAL_LOW, C.BIAS20_IDEAL_HIGH, soft=0.9)

    # d) RSI 所在區帶：太熱不宜進場，太冷則要另外看有沒有止跌（交給轉強成分）
    s_rsi = _band(f["rsi"], C.RSI_ENTRY_IDEAL_LOW, C.RSI_ENTRY_IDEAL_HIGH, soft=0.8)

    strength = _clamp(0.34 * s_pos + 0.26 * s_room + 0.22 * s_bias + 0.18 * s_rsi)

    if breakout:
        note = f"突破前波高點且距 60 日高仍有 {room60:.1f}% 空間，屬突破型進場點"
    elif pos >= C.POS_BAD:
        note = f"收盤已在近 20 日區間頂端（位置 {pos*100:.0f}%），距前高僅 {room20:.1f}%，現在進場等於追高"
    elif s_pos > 0.6 and s_room > 0.5:
        note = (f"位於近 20 日區間 {pos*100:.0f}% 位置，距前高還有 {room20:.1f}% 空間，"
                f"距 MA20 乖離 {f['bias20']:+.1f}%")
    else:
        note = f"位置 {pos*100:.0f}%、距前高 {room20:.1f}%、乖離 {f['bias20']:+.1f}%，進場點普通"

    return strength, note


# ---------------------------------------------------------------------------
# ② 趨勢品質（權重 20）
#    只回答「這檔股票強不強」，不代表現在買賺錢機率高。
# ---------------------------------------------------------------------------
def _score_trend(f: dict) -> tuple[float, str]:
    parts = []
    parts.append((0.22, 1.0 if f["close"] > f["ma20"] else 0.0))                     # 站穩 MA20
    parts.append((0.20, 1.0 if f["ma5"] > f["ma10"] > f["ma20"] else
                        (0.5 if f["ma5"] > f["ma10"] else 0.0)))                     # 均線排列
    parts.append((0.20, _ramp_up(f["ma20_slope"], -1.5, 2.0)))                       # MA20 斜率
    parts.append((0.16, _ramp_up(f["ma60_slope"], -1.5, 3.0)))                       # MA60 斜率
    parts.append((0.12, 1.0 if f["close"] > f["ma60"] else 0.0))                     # 站上季線
    parts.append((0.10, _ramp_up(f["ma10_slope"], -1.0, 2.0)))                       # 近 5～10 日趨勢

    strength = _clamp(sum(w * s for w, s in parts))

    if strength >= 0.75:
        note = f"多頭排列且 MA20／MA60 同步向上（MA20 斜率 {f['ma20_slope']:+.1f}%）"
    elif strength >= 0.45:
        note = f"中期趨勢中性偏多，MA20 斜率 {f['ma20_slope']:+.1f}%"
    else:
        note = f"均線仍偏空排列，MA20 斜率 {f['ma20_slope']:+.1f}%"
    return strength, note


# ---------------------------------------------------------------------------
# ③ 轉強 / 動能確認（權重 15）
#    跌深不等於會反彈，要看到「真的開始轉向」的證據。
# ---------------------------------------------------------------------------
def _score_reversal(f: dict) -> tuple[float, str]:
    hits, labels = [], []

    def add(weight, cond, label):
        hits.append(weight if cond else 0.0)
        if cond:
            labels.append(label)

    add(0.18, f["rsi"] > f["rsi_prev"] and f["rsi_min5"] < 45, "RSI 自低檔回升")
    add(0.16, f["close"] > f["ma5"] and f["prev_close"] <= f["ma5_prev"], "收盤重新站回 MA5")
    add(0.14, f["ma5_slope"] > 0, "MA5 開始上彎")
    add(0.14, f["ma5"] > f["ma10"] and f["ma5_prev"] <= f["ma10_prev"], "MA5 金叉 MA10")
    add(0.14, f["macd_hist"] > f["macd_hist_prev"] > 0 or
              (f["macd_hist"] > f["macd_hist_prev"] and f["macd_hist"] > f["macd_hist_prev3"]),
        "MACD 柱狀體改善")
    add(0.12, f["vol_ratio"] > 1.1 and f["is_up_day"] > 0, "帶量收紅")
    add(0.12, f["held_low5"] > 0 and f["is_new_low"] <= 0, "近期低點未再破")

    strength = _clamp(sum(hits))
    if not labels:
        note = "尚未出現明確的轉強訊號"
    else:
        note = "、".join(labels[:4])
    return strength, note


# ---------------------------------------------------------------------------
# ④ 量價品質（權重 15）
#    量大不等於好。要分辨「好的爆量」與「危險的爆量」。
# ---------------------------------------------------------------------------
def _score_volume(f: dict) -> tuple[float, str]:
    r = f["vol_ratio"]
    base = _ramp_up(r, 1.0, C.VOL_FULL_RATIO)      # 量能規模
    if base <= 0:
        return 0.0, f"量能 {r:.1f} 倍，低於 20 日均量，買盤不足"

    # --- 品質係數：同樣的量，收成什麼樣子差很多 ---
    q, why = 1.0, []
    up = f["is_up_day"] > 0
    strong_close = f["close_pos_bar"] >= C.CLOSE_STRONG_POS
    long_upper = f["upper_shadow"] >= C.UPPER_SHADOW_BAD

    if up and strong_close and not long_upper:
        why.append("上漲帶量且收盤守在當日高檔")
    elif up and long_upper:
        q *= 0.55
        why.append(f"雖上漲帶量，但留下 {f['upper_shadow']*100:.0f}% 上影線，追價力道有疑慮")
    elif not up:
        q *= 0.25
        why.append("爆量收黑，量增價跌是賣壓訊號")
        if f["close"] < f["ma5"] and f["close"] < f["ma10"]:
            q *= 0.4
            why.append("且收盤跌破 MA5／MA10")
    elif up and not strong_close:
        q *= 0.7
        why.append("收盤未能守在當日高檔")

    # 高檔爆量：位置已在區間頂端又爆大量，出貨機率高
    if f["pos20"] > 0.85 and r > 3.0:
        q *= 0.45
        why.append("高檔爆量，需留意出貨")
    elif r >= C.VOL_BLOWOFF_RATIO:
        q *= 0.8
        why.append(f"量能暴增 {r:.1f} 倍，屬異常放量")

    # 有效突破額外加成
    if _valid_breakout(f):
        q = min(1.25, q * 1.25)
        why.append("量能有效確認突破")

    # 量價配合：上漲日均量 vs 下跌日均量
    ud = f["vol_ud_ratio"]
    if ud > 1.15:
        q = min(1.25, q * 1.1)
        why.append(f"近月上漲日均量為下跌日的 {ud:.1f} 倍")
    elif 0 < ud < 0.85:
        q *= 0.85
        why.append("近月下跌日的量比上漲日大")

    strength = _clamp(base * q)
    note = f"量能 {r:.1f} 倍；" + "，".join(why) if why else f"量能 {r:.1f} 倍"
    return strength, note


# ---------------------------------------------------------------------------
# ⑤ 風險報酬（權重 15）
#    「潛在上漲空間值不值得承擔目前的下跌風險？」
# ---------------------------------------------------------------------------
def _score_rr(f: dict) -> tuple[float, str]:
    up, down, rr = f["upside_pct"], f["downside_pct"], f["rr_ratio"]
    if down <= 0 or rr <= 0:
        return 0.0, "無法估計有效支撐，風險難以衡量"

    s_rr = _ramp_up(rr, C.RR_MIN, C.RR_FULL)
    s_room = _ramp_up(up, C.MIN_UPSIDE_PCT, C.MIN_UPSIDE_PCT * 2.5)   # 空間太小沒肉
    strength = _clamp(0.7 * s_rr + 0.3 * s_room)

    note = (f"上檔至 {f['target']:.1f} 約 +{up:.1f}%，下檔支撐 {f['support']:.1f} 約 −{down:.1f}%，"
            f"風險報酬比 {rr:.1f}:1")
    return strength, note


# ---------------------------------------------------------------------------
# 風險分數（0～100，越高越危險）
# ---------------------------------------------------------------------------
def _risk_score(f: dict) -> tuple[float, list[dict]]:
    breakout = _valid_breakout(f)
    items = []

    def add(key, level, label):
        level = _clamp(level)
        if level > 0.02:
            items.append({"key": key, "level": round(level, 3),
                          "weight": C.RISK_WEIGHTS[key], "label": label})

    # RSI 過熱
    add("rsi_hot", _ramp_up(f["rsi"], C.RSI_HOT, C.RSI_HOT_FULL),
        f"RSI {f['rsi']:.0f} 已進入過熱區")

    # 距近期高點太近＝追高（有效突破時風險減半，因為突破本來就會貼高）
    near = _ramp_down(f["pct_below_high20"], 0.0, C.NEAR_HIGH_PCT)
    if breakout:
        near *= 0.5
    add("near_high", near, f"距 20 日高點僅 {f['pct_below_high20']:.1f}%，追高空間有限")

    # 距近期低點已經漲很多
    add("far_from_low", _ramp_up(f["pct_above_low20"], C.FAR_FROM_LOW_PCT, C.FAR_FROM_LOW_FULL),
        f"已自 20 日低點上漲 {f['pct_above_low20']:.0f}%")

    # 乖離 MA20 太遠
    add("ma20_bias", _ramp_up(f["bias20"], C.BIAS20_WARN, C.BIAS20_FULL),
        f"距 MA20 正乖離 {f['bias20']:.1f}%，回測均線的空間大")

    # 短期漲幅過大／連續大漲
    runup = _ramp_up(f["ret5"], C.RUNUP5_WARN, C.RUNUP5_FULL)
    if f["up_streak"] >= 4:
        runup = max(runup, 0.45)
    add("run_up", runup, f"近 5 日已漲 {f['ret5']:+.1f}%（連 {int(f['up_streak'])} 日紅）")

    # 假突破風險：看起來突破，但沒有量、留長上影、或已經過熱
    if f["is_breakout"] > 0 and not breakout:
        reasons = []
        if f["vol_ratio"] < C.VOL_SURGE_RATIO:
            reasons.append(f"量能僅 {f['vol_ratio']:.1f} 倍")
        if f["upper_shadow"] > C.UPPER_SHADOW_BAD:
            reasons.append(f"上影線 {f['upper_shadow']*100:.0f}%")
        if f["close_pos_bar"] < C.CLOSE_STRONG_POS:
            reasons.append("收盤未守在高檔")
        if f["rsi"] >= C.RSI_HOT:
            reasons.append("RSI 過熱")
        add("fake_breakout", 0.55 + 0.15 * len(reasons),
            "突破但未獲確認（" + "、".join(reasons) + "），有假突破風險")

    # 跌深但完全沒止跌：這是「便宜」，不是「機會」
    falling = f["ma5"] < f["ma10"] < f["ma20"] and f["close"] < f["ma5"]
    if falling or f["is_new_low"] > 0:
        level = 0.5
        if f["is_new_low"] > 0:
            level += 0.3
        if falling:
            level += 0.2
        if f["rsi"] < 30:
            level = min(1.0, level + 0.1)
        add("no_bottom", level, "均線空頭排列且仍在破底，屬跌深但尚未止跌")

    if not items:
        return 0.0, []

    total_w = sum(C.RISK_WEIGHTS.values())
    score = sum(it["level"] * it["weight"] for it in items) / total_w * 100
    items.sort(key=lambda it: it["level"] * it["weight"], reverse=True)
    return _clamp(score, 0, 100), items


# ---------------------------------------------------------------------------
# 型態分類（給 UI 用一個詞說明這是哪一種機會）
# ---------------------------------------------------------------------------
def _classify(f: dict, comp: dict, risk: float) -> str:
    if _valid_breakout(f):
        return "帶量突破"
    if f["ma5"] < f["ma10"] < f["ma20"] and f["close"] < f["ma5"]:
        return "弱勢未止跌"
    if f["pos20"] <= 0.45 and comp["reversal"][0] >= 0.45:
        return "低檔止跌轉強"
    if f["rsi"] >= C.RSI_HOT or f["pos20"] >= C.POS_BAD:
        return "強勢但已漲多"
    if comp["trend"][0] >= 0.6:
        return "多頭回檔"
    return "區間整理"


# ---------------------------------------------------------------------------
# 主函式
# ---------------------------------------------------------------------------
def score_stock(f: dict) -> dict:
    """
    輸入 indicators.features_at() 的特徵 dict，
    回傳完整評分結果（含分項、星等、預期空間、主要風險、排序用的 tie-breaker）。
    """
    comp = {
        "entry":    _score_entry(f),
        "trend":    _score_trend(f),
        "reversal": _score_reversal(f),
        "volume":   _score_volume(f),
        "rr":       _score_rr(f),
    }

    breakdown, reasons, raw, max_raw = {}, [], 0.0, 0.0
    for key, (strength, note) in comp.items():
        w = C.WEIGHTS[key]
        pts = strength * w
        raw += pts
        max_raw += w
        breakdown[key] = {"points": round(pts, 1), "max": w, "ratio": round(strength, 3)}
        if note:
            reasons.append(note)

    opportunity = raw / max_raw * 100 if max_raw else 0.0
    risk, risk_items = _risk_score(f)
    pps = _clamp(opportunity * (1 - C.RISK_MAX_CUT * risk / 100), 0, 100)

    # --- UI 用的星等（風險是「越多星越危險」） ---
    def stars(ratio):
        n = int(round(_clamp(ratio) * 5))
        return "★" * n + "☆" * (5 - n)

    risk_level = ("低" if risk < 12 else "中低" if risk < 25 else
                  "中" if risk < 40 else "偏高" if risk < 60 else "高")

    star_map = {
        "trend": stars(comp["trend"][0]),
        "entry": stars(comp["entry"][0]),
        "volume": stars(comp["volume"][0]),
        "reversal": stars(comp["reversal"][0]),
        "risk": stars(risk / 100),
    }

    # --- 預期 Swing 空間：保守到樂觀 ---
    up = max(f["upside_pct"], 0.0)
    swing_low = round(up * 0.45, 1)
    swing_high = round(up, 1)

    kind = _classify(f, comp, risk)
    main_risk = risk_items[0]["label"] if risk_items else "目前未偵測到明顯的追高或破底風險"

    return {
        "score": round(pps, 1),              # 獲利可能性分數（排名主鍵）
        "opportunity": round(opportunity, 1),
        "risk": round(risk, 1),
        "breakdown": breakdown,
        "risk_items": risk_items,
        "reasons": reasons,
        "stars": star_map,
        "risk_level": risk_level,
        "kind": kind,
        "headline": _headline(kind, f, comp),
        "why": _why(kind, f, comp, risk_items),
        "main_risk": main_risk,
        "swing_low": swing_low,
        "swing_high": swing_high,
        "flags": [it["label"] for it in risk_items[:2] if it["level"] >= 0.5],
        # tie-breaker 用：分數接近時的次要排序依據
        "rr_ratio": round(f["rr_ratio"], 2) if f["rr_ratio"] else 0.0,
        "downside_pct": round(f["downside_pct"], 2),
        "upside_pct": round(f["upside_pct"], 2),
    }


def _headline(kind: str, f: dict, comp: dict) -> str:
    """表格上那一行 12 字內的短理由。"""
    return {
        "帶量突破": f"帶量突破前高，量增 {f['vol_ratio']:.1f} 倍",
        "低檔止跌轉強": f"低檔轉強（RSI {f['rsi']:.0f}）",
        "多頭回檔": f"多頭回檔至支撐（乖離 {f['bias20']:+.1f}%）",
        "強勢但已漲多": f"趨勢強但位置偏高（RSI {f['rsi']:.0f}）",
        "弱勢未止跌": "跌深但尚未止跌",
        "區間整理": "區間整理，訊號中性",
    }.get(kind, "訊號中性")


def _why(kind: str, f: dict, comp: dict, risk_items: list) -> str:
    """一句話說明「為什麼排這個名次」，取強度最高的兩項成分 + 風險狀態。"""
    label = {
        "entry": "進場位置好",
        "trend": "趨勢結構強",
        "reversal": "轉強訊號明確",
        "volume": "量價配合",
        "rr": "風險報酬比佳",
    }
    ranked = sorted(comp.items(), key=lambda kv: kv[1][0] * C.WEIGHTS[kv[0]], reverse=True)
    tops = [label[k] for k, (s, _) in ranked[:2] if s >= 0.5]

    bits = []
    if tops:
        bits.append("、".join(tops))
    if kind == "帶量突破":
        # 已經突破 20 日高的，該講的是「距下一個壓力（60 日高）還有多少」
        bits.append(f"突破後距 60 日高仍有 {f['pct_below_high60']:.1f}% 空間"
                    if f["pct_below_high60"] >= 2
                    else f"已逼近 60 日高（僅剩 {f['pct_below_high60']:.1f}%）")
    else:
        bits.append(f"距前高仍有 {f['pct_below_high20']:.1f}% 空間" if f["pct_below_high20"] >= 3
                    else f"距前高僅剩 {f['pct_below_high20']:.1f}%")
    if not risk_items:
        bits.append("目前追高風險低")
    elif risk_items[0]["level"] >= 0.5:
        bits.append(f"但{risk_items[0]['label']}")
    else:
        bits.append("風險項目輕微")
    return "；".join(bits) + "。"


# ---------------------------------------------------------------------------
# 排序
# ---------------------------------------------------------------------------
def sort_key(row: dict):
    """
    排名順位（有足夠歷史樣本時）：
        期望值 EV → 平滑勝率 → Profit Factor → 平均回撤 → 技術分數

    兩個原則：
      - **EV <= 0 的一律降到有正 EV 的之後**，但不刪除。看得到、但排在後面。
      - 沒有歷史樣本的走 fallback：EV 視為 0（與負 EV 同層），純用技術分數決定先後。

    技術分數只是型態描述，不是機率，所以它排在最後。
    """
    ev = row.get("hist_expectancy")
    has = ev is not None
    ev = float(ev) if has else 0.0

    wr = row.get("hist_calibrated")
    wr = float(wr) if wr is not None else 50.0
    pf = row.get("hist_pf")
    pf = float(pf) if pf is not None else 1.0
    mdd = row.get("hist_mdd")
    mdd = float(mdd) if mdd is not None else -99.0

    return (
        0 if ev > 0 else 1,      # 第一層：正期望值優先
        -round(ev, 3),
        -round(wr, 1),
        -round(pf, 2),
        -round(mdd, 1),
        -row["score"],           # 技術分數（fallback 時才真正起作用）
        -row.get("rr_ratio", 0),
        row.get("downside_pct", 99),
    )
