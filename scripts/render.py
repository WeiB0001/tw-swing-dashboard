# -*- coding: utf-8 -*-
"""
render.py — 把計算結果渲染成 index.html

版面在 templates/dashboard.html.j2。改配色／排版只要動樣板。
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

import config as C
import strategy

log = logging.getLogger("render")

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = ROOT / "templates"

# 機會分數能量條的五段：(breakdown 的 key, 顯示文字)
# 順序 = 條上的左到右順序，段寬由 config.WEIGHTS 決定
SEGMENTS = [
    ("entry", "位置"),
    ("trend", "趨勢"),
    ("reversal", "轉強"),
    ("volume", "量價"),
    ("rr", "風報"),
]

STAR_ROWS = [
    ("entry", "進場位置"),
    ("trend", "趨勢"),
    ("volume", "量價"),
    ("reversal", "轉強"),
]


def _format_twd(amount: int) -> str:
    """把金額轉成台灣人習慣的說法：1.2 億 / 3,500 萬 / 8,000 元。"""
    if amount >= 100_000_000:
        return f"{amount / 100_000_000:g} 億元"
    if amount >= 10_000:
        return f"{amount / 10_000:,.0f} 萬元"
    return f"{amount:,} 元"


def _price_map_json(rows: list) -> str:
    """
    給「我的交易記帳」用的精簡價格表：{代號: [名稱, 收盤, 昨收]}。
    直接重用本次已經算好的資料，不會為了記帳多抓任何 API。
    """
    m = {}
    for r in rows:
        try:
            m[r["code"]] = [r.get("name", ""), round(float(r["close"]), 2),
                            round(float(r.get("prev_close") or 0), 2) or None]
        except Exception:
            continue
    return json.dumps(m, ensure_ascii=False, separators=(",", ":"))


def _groups(rows: list) -> list:
    """
    這次排行裡實際出現的產業大類與檔數，用來產生 Tab。
    只列出真的有標的的分類，不會出現點了沒東西的空 Tab。
    """
    from collections import Counter
    c = Counter(r.get("group") or "其他" for r in rows)
    # 常看的排前面，其餘依檔數多寡
    order = ["半導體", "電腦週邊", "光電", "網通", "電子零件", "其他電子",
             "電子通路", "資訊服務", "金融", "ETF"]
    head = [(g, c[g]) for g in order if c.get(g)]
    tail = sorted([(g, n) for g, n in c.items() if g not in order],
                  key=lambda x: -x[1])
    return [{"name": g, "count": n} for g, n in head + tail]


def reference_rows(rows: list[dict], data_date: str | None = None) -> list[dict]:
    """Prepare an unfiltered reference view without changing strategy records."""
    result = [dict(row) for row in rows]

    def technical_key(row):
        value = row.get("score")
        known = isinstance(value, (int, float)) and math.isfinite(value)
        return (not known, -value if known else 0, row.get("code", ""))

    for rank, row in enumerate(sorted(result, key=technical_key), 1):
        row["technical_rank"] = rank
    for row in result:
        warnings = []
        if data_date and row.get("quote_date") != data_date:
            warnings.append("行情日期不一致，價格需更新確認")
        if row.get("hist_risk_reward") is None:
            warnings.append("歷史統計不足，達標與虧損機率未知")
        else:
            if row["hist_risk_reward"] <= 0:
                warnings.append("歷史風險報酬分不高於 0")
            if row.get("hist_expectancy") is not None and row["hist_expectancy"] <= 0:
                warnings.append("歷史平均淨報酬不高於 0")
            if any(row.get(k) is None or row[k] <= 0 for k in ("hist_ev_lower", "hist_risk_reward_lower")):
                warnings.append("統計保守估計未轉正")
        if row.get("main_risk"):
            warnings.append(row["main_risk"])
        elif row.get("momentum_tier", 2) >= 2:
            warnings.append("技術轉弱或追高風險")
        # Keep one statistical warning and the current technical risk visible.
        brief = warnings[:1]
        if row.get("main_risk") and row["main_risk"] not in brief:
            brief.append(row["main_risk"])
        row["reference_warnings"] = warnings
        row["reference_risk"] = "；".join(brief) or "仍有價格波動與跳空風險"
    return result


def render_html(payload: dict) -> str:
    import ranking
    policy = strategy.capital_policy()
    if not policy["blocked"]:
        policy = payload.get("capital_policy") or {"blocked": True, "reason": "尚未取得交易資格驗證"}
        if not strategy.compatible(payload.get("meta")) or not ranking.research_validated(payload.get("backtest")):
            policy = {"blocked": True, "reason": "風險報酬綜合排名尚未通過樣本外驗證，暫不配置資金"}
            quality = (payload.get("backtest") or {}).get("data_quality") or {}
            if quality.get("issue_count"):
                policy["reason"] += "；行情異常或公司行動待核對"
            if quality.get("unresolved_exits"):
                policy["reason"] += "；仍有未結算出場"
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    tpl = env.get_template("dashboard.html.j2")
    rows = reference_rows(payload["rows"], payload["meta"].get("data_date"))
    return tpl.render(
        meta=payload["meta"],
        index=payload.get("index") or None,
        backtest=payload.get("backtest") or None,
        livecheck=payload.get("livecheck") or None,
        us=payload.get("us") or None,
        signals=payload.get("signals") or None,
        portfolio=payload.get("portfolio") or None,
        research_forward=payload.get("research_forward") or None,
        risk_settings={"per_trade": C.RISK_PER_TRADE_PCT, "total": C.MAX_TOTAL_RISK_PCT,
                       "sector": C.MAX_SECTOR_POSITION_PCT, "buffer": C.RISK_GAP_BUFFER_PCT,
                       "fee": C.BROKER_FEE_PCT, "minimum_fee": C.MIN_BROKER_FEE_TWD,
                       "slippage": C.SLIPPAGE_PCT, "stock_tax": C.STOCK_SELL_TAX_PCT,
                       "etf_tax": C.ETF_SELL_TAX_PCT, "reference": C.REFERENCE_NOTIONAL_TWD},
        entry_settings={"premium": C.ENTRY_MAX_PREMIUM_PCT, "bias": C.ENTRY_MAX_MA20_BIAS_PCT},
        capital_policy=policy,
        no_loss_required=C.REQUIRE_NO_LOSS,
        loss_aversion=C.LOSS_AVERSION,
        stop_loss_pct=C.STOP_LOSS_NET_PCT,
        min_calibration_dates=C.MIN_CALIBRATION_DATES,
        research_count=sum(bool(r.get("research_eligible")) for r in payload["rows"]),
        pattern_min_samples=C.PATTERN_MIN_SAMPLES,
        min_samples=C.MIN_SAMPLES_SCORE,
        w_evidence=C.FINAL_W_EVIDENCE,
        w_position=C.FINAL_W_POSITION,
        w_momentum=C.FINAL_W_MOMENTUM,
        in_sample_discount=C.IN_SAMPLE_DISCOUNT,
        top_slots=C.TOP_SLOTS,
        groups=_groups(payload["rows"]),
        hold_days=C.HOLD_DAYS,
        exit_days=C.EXIT_MAX_DAYS,
        min_exit_days=C.EXIT_MIN_DAYS,
        profit_target=C.EXIT_MIN_PROFIT,
        cost_pct=C.TOTAL_COST_PCT,
        eligible_count=0 if policy["blocked"] else sum(bool(r.get("trade_eligible")) for r in payload["rows"]),
        shrink_k=C.SHRINK_K,
        ov_min_r2=C.OVERSEAS_MIN_R2,
        plan_t1=C.PLAN_T1_ATR,
        plan_stop=C.PLAN_STOP_MAX_ATR,
        mom_high=C.MOM_GOOD_CHG_HIGH,
        mom_low=C.MOM_GOOD_CHG_LOW,
        mom_green=C.MOM_GREEN_MIN,
        mom_bias=C.MOM_BIAS_MAX,
        price_map_json=_price_map_json(payload["rows"]),
        paper_max_positions=C.PAPER_MAX_POSITIONS,
        paper_max_hold=C.PAPER_MAX_HOLD_DAYS,
        smooth_wins=C.SMOOTH_WINS,
        smooth_n=C.SMOOTH_N,
        rows=rows,
        positive_risk_reward_count=sum(r.get("hist_risk_reward") is not None and r["hist_risk_reward"] > 0 for r in rows),
        segments=SEGMENTS,
        star_rows=STAR_ROWS,
        hide_unaffordable=C.HIDE_UNAFFORDABLE,
        initial_visible=C.INITIAL_VISIBLE,
        budget_presets=C.BUDGET_PRESETS,
        split_options=C.SPLIT_OPTIONS,
        default_splits=C.DEFAULT_SPLITS,
        max_position_pct=C.MAX_POSITION_PCT,
        allow_odd_lot=C.ALLOW_ODD_LOT,
        weights=C.WEIGHTS,
        thresholds={
            "vol_surge": C.VOL_SURGE_RATIO,
            "vol_full": C.VOL_FULL_RATIO,
            "rsi_hot": int(C.RSI_HOT),
            "near_high": C.NEAR_HIGH_PCT,
            "risk_cut": int(C.RISK_MAX_CUT * 100),
            "min_price": int(C.MIN_CLOSE_PRICE),
            "min_turnover": _format_twd(C.MIN_TURNOVER_TWD),
            "max_price": int(C.MAX_CLOSE_PRICE) if C.MAX_CLOSE_PRICE else 0,
        },
    )


def render_static(name: str) -> str:
    """雷達頁等純靜態頁：資料從 data/*.json 動態載入，樣板不需要 payload。"""
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)),
                      autoescape=select_autoescape(["html"]),
                      trim_blocks=True, lstrip_blocks=True)
    return env.get_template(name).render()


def write_outputs(payload: dict) -> None:
    """寫出 index.html、portfolio.html、data/latest.json，並存一份當日封存檔。"""
    html = render_html(payload)

    (ROOT / C.OUTPUT_HTML).write_text(html, encoding="utf-8")
    log.info("已寫出 %s（%d bytes）", C.OUTPUT_HTML, len(html.encode()))

    for tpl, out in [("portfolio.html.j2", "portfolio.html"),
                     ("radar.html.j2", "radar.html"),
                     ("others.html.j2", "others.html"),
                     ("alerts.html.j2", "alerts.html"),
                     ("momentum.html.j2", "momentum.html")]:
        try:
            page = render_static(tpl)
            (ROOT / out).write_text(page, encoding="utf-8")
            log.info("已寫出 %s（%d bytes）", out, len(page.encode()))
        except Exception as e:
            log.warning("%s 產生失敗（不影響首頁）：%s", out, e)

    json_path = ROOT / C.OUTPUT_JSON
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    archive = ROOT / C.ARCHIVE_DIR
    archive.mkdir(parents=True, exist_ok=True)
    (archive / f"{payload['meta']['trade_date']}.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    # 每日排名存檔是 livecheck 的資料來源，保留久一點但不要無限膨脹
    try:
        files = sorted(archive.glob("20*-*-*.json"))
        for old in files[:-C.ARCHIVE_KEEP_DAYS]:
            old.unlink()
    except Exception:
        pass
    log.info("已寫出 %s 與當日封存（存檔 %d 份）",
             C.OUTPUT_JSON, len(list(archive.glob("20*-*-*.json"))))
