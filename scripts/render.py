# -*- coding: utf-8 -*-
"""
render.py — 把計算結果渲染成 index.html

版面在 templates/dashboard.html.j2。改配色／排版只要動樣板。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

import config as C

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


def render_html(payload: dict) -> str:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    tpl = env.get_template("dashboard.html.j2")
    return tpl.render(
        meta=payload["meta"],
        index=payload.get("index") or None,
        backtest=payload.get("backtest") or None,
        us=payload.get("us") or None,
        signals=payload.get("signals") or None,
        portfolio=payload.get("portfolio") or None,
        pattern_min_samples=C.PATTERN_MIN_SAMPLES,
        price_map_json=_price_map_json(payload["rows"]),
        paper_max_positions=C.PAPER_MAX_POSITIONS,
        paper_max_hold=C.PAPER_MAX_HOLD_DAYS,
        smooth_wins=C.SMOOTH_WINS,
        smooth_n=C.SMOOTH_N,
        rows=payload["rows"],
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


def render_portfolio() -> str:
    """
    我的交易記帳頁。純靜態，不需要任何 payload——
    交易資料在使用者的 LocalStorage，最新價由首頁寫進 LocalStorage 共用。
    """
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)),
                      autoescape=select_autoescape(["html"]),
                      trim_blocks=True, lstrip_blocks=True)
    return env.get_template("portfolio.html.j2").render()


def write_outputs(payload: dict) -> None:
    """寫出 index.html、portfolio.html、data/latest.json，並存一份當日封存檔。"""
    html = render_html(payload)

    (ROOT / C.OUTPUT_HTML).write_text(html, encoding="utf-8")
    log.info("已寫出 %s（%d bytes）", C.OUTPUT_HTML, len(html.encode()))

    try:
        pf = render_portfolio()
        (ROOT / "portfolio.html").write_text(pf, encoding="utf-8")
        log.info("已寫出 portfolio.html（%d bytes）", len(pf.encode()))
    except Exception as e:
        log.warning("portfolio.html 產生失敗（不影響首頁）：%s", e)

    json_path = ROOT / C.OUTPUT_JSON
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    archive = ROOT / C.ARCHIVE_DIR
    archive.mkdir(parents=True, exist_ok=True)
    (archive / f"{payload['meta']['trade_date']}.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    log.info("已寫出 %s 與當日封存", C.OUTPUT_JSON)
