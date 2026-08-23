# -*- coding: utf-8 -*-
"""
plan.py — 明日交易計畫（純規則，沒有模型、沒有 AI）

全部沿用 indicators already 算好的欄位：ATR、MA、support、target、pos20、bias20。
只是把它們翻譯成「明天要看什麼價位」。

三種狀態：
  可觀察進場 ── 型態成立、風險不高，明天過觸發價可以留意
  等待確認   ── 型態還沒完成，先觀察不動作
  禁止追價   ── 過熱／貼前高／乖離過大，明天不論怎麼漲都不該追

⚠️ 這是把技術位置換算成價位的算術，不是獲利保證。
"""

from __future__ import annotations

import math

import config as C


def _ok(x) -> bool:
    """擋掉 NaN 與 inf，避免除零或壞資料把頁面弄掛。"""
    try:
        return x is not None and math.isfinite(float(x))
    except Exception:
        return False


def build_plan(f: dict, res: dict) -> dict:
    """f = indicators.features_at() 的結果，res = scoring.score_stock() 的結果。"""
    close = float(f.get("close") or 0)
    atr = float(f.get("atr") or 0)
    if not _ok(close) or close <= 0:
        return {"available": False, "status": "資料不足", "note": "價格資料異常，不產生計畫"}
    if not _ok(atr) or atr <= 0:
        atr = close * 0.02          # ATR 壞掉時用 2% 當保底，免得後面全是除零

    high = float(f.get("high") or close)
    low = float(f.get("low") or close)
    support = float(f.get("support") or (close - 2 * atr))
    target = float(f.get("target") or (close + 2 * atr))
    risk = float(res.get("risk") or 0)

    # --- 狀態 ---
    overheat = (f.get("rsi", 0) >= C.RSI_HOT
                or f.get("pos20", 0) >= C.POS_BAD
                or f.get("bias20", 0) >= C.BIAS20_WARN)
    weak = f.get("ma5", 0) < f.get("ma10", 0) < f.get("ma20", 0) and close < f.get("ma5", 0)

    if overheat or risk >= 55:
        status, note = "禁止追價", "位置過高或風險偏大，明天不論怎麼漲都不建議追價"
    elif weak:
        status, note = "等待確認", "均線仍空頭排列，等出現止跌訊號再說"
    elif res.get("kind") in ("帶量突破", "低檔止跌轉強", "多頭回檔"):
        status, note = "可觀察進場", "型態成立，明天站上觸發價可留意"
    else:
        status, note = "等待確認", "型態尚未完成，先觀察"

    # --- 價位 ---
    trigger = high + 0.05 * atr                      # 過今日高點才算轉強
    invalidation = low - 0.05 * atr                  # 跌破今日低點，這個型態就算失效
    stop = min(support, invalidation - 0.2 * atr)    # 停損取兩者較低，留一點緩衝
    stop = max(stop, close - 3 * atr)                # 但不要離譜地遠
    t1 = min(target, trigger + 1.5 * atr)            # 第一目標：近壓或 1.5 ATR
    t1 = max(t1, trigger + 0.8 * atr)                # 目標一定要在觸發價之上，否則 RR 會變負數
    t2 = max(target, t1 + 1.0 * atr, trigger + 2.5 * atr)

    # ----------------------------------------------------------------
    # 進場判定：只回答「現在適不適合進場」，不影響選股模型與排名
    # ----------------------------------------------------------------
    kind = res.get("kind") or ""
    vol = float(f.get("vol_ratio") or 0)
    breakout_kind = kind == "帶量突破"

    # 判定基準：突破型看有沒有站上前 20 日高，回檔／轉強型看有沒有站回 MA5
    if breakout_kind:
        ref = float(f.get("high20_prev") or 0)
        ref_label = "前 20 日高"
    else:
        ref = float(f.get("ma5") or 0)
        ref_label = "MA5"
    confirmed = ref > 0 and close >= ref

    # 量能分級
    if vol >= 1.0:
        vol_tier, vol_text = "ok", "量能確認（%.2f×）" % vol
    elif vol >= 0.7:
        vol_tier, vol_text = "weak", "量能普通（%.2f×）" % vol
    else:
        vol_tier, vol_text = "bad", "⚠ 量能不足（%.2f×），不追價" % vol

    if status == "禁止追價":
        entry_icon, entry_status = "⛔", "禁止追價"
        entry_note = note
    elif not confirmed:
        entry_icon, entry_status = "🟡", "等待確認，不建議立即進場"
        entry_note = "收盤 %.2f 尚未站上%s %.2f" % (close, ref_label, ref) if ref > 0 \
            else "缺少判定基準價，無法確認"
    elif vol_tier == "bad":
        entry_icon, entry_status = "⚠", "量能不足，不追價"
        entry_note = "收盤已站上%s，但%s" % (ref_label, vol_text)
    elif breakout_kind:
        # 突破型：收盤站上前高且量能 >= 1.0 才算收盤確認
        entry_icon, entry_status = "🟢", "收盤確認，可觀察進場"
        entry_note = "收盤 %.2f 站上%s %.2f，%s" % (close, ref_label, ref, vol_text)
    elif vol_tier == "ok":
        entry_icon, entry_status = "🟢", "收盤確認，可觀察進場"
        entry_note = "收盤站回%s %.2f，%s" % (ref_label, ref, vol_text)
    else:
        entry_icon, entry_status = "🟡", "等待確認，不建議立即進場"
        entry_note = "收盤站回%s %.2f，但%s" % (ref_label, ref, vol_text)

    risk_amt = trigger - stop
    reward_amt = t1 - trigger
    rr = (reward_amt / risk_amt) if risk_amt > 0 else None

    # RR 警示：只提醒，不改任何歷史 EV 或排名
    if rr is None or not _ok(rr):
        rr_tier, rr_tag = "na", "風報無法估算"
    elif rr >= 1.5:
        rr_tier, rr_tag = "good", "✓ 風報佳（%.2f）" % rr
    elif rr >= 1.0:
        rr_tier, rr_tag = "fair", "⚠ 風報普通（%.2f）" % rr
    else:
        rr_tier, rr_tag = "poor", "⚠ 風報偏差（%.2f），不追價" % rr

    dist_trigger = (trigger / close - 1) * 100 if close > 0 else None

    return {
        "available": True,
        "status": status,
        # --- 進場判定（只回答現在適不適合進場）---
        "entry_icon": entry_icon,
        "entry_status": entry_status,
        "entry_note": entry_note,
        "ref_price": round(ref, 2) if ref > 0 else None,
        "ref_label": ref_label,
        "vol_ratio": round(vol, 2),
        "vol_tier": vol_tier,
        "rr_tier": rr_tier,
        "rr_tag": rr_tag,
        "dist_trigger_pct": round(dist_trigger, 2) if dist_trigger is not None else None,
        # 盤中狀態要有真實報價才能判斷，這裡一律 False，不偽造
        "intraday_checked": False,
        "note": note,
        "trigger": round(trigger, 2),
        "invalidation": round(invalidation, 2),
        "stop": round(stop, 2),
        "target1": round(t1, 2),
        "target2": round(t2, 2),
        "rr": round(rr, 2) if (_ok(rr) and rr > 0) else None,
        "risk_pct": round(risk_amt / trigger * 100, 1) if trigger > 0 else None,
        "atr_pct": round(atr / close * 100, 1),
    }
