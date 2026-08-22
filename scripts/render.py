# -*- coding: utf-8 -*-
"""
render.py — 把計算結果渲染成 index.html

樣板在 templates/dashboard.html.j2。要改版面／配色只要改樣板，
Python 這邊不用動。
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

# 能量條的五段：(breakdown 的 key, 顯示文字)。順序＝條上的左到右順序。
SEGMENTS = [
    ("oversold", "超賣"),
    ("volume", "量能"),
    ("near_low", "低檔"),
    ("ma_reclaim", "均線"),
    ("structure", "結構"),
]


def _format_twd(amount: int) -> str:
    """把金額轉成台灣人習慣的說法：1.2 億 / 3,500 萬 / 8,000 元。"""
    if amount >= 100_000_000:
        return f"{amount / 100_000_000:g} 億元"
    if amount >= 10_000:
        return f"{amount / 10_000:,.0f} 萬元"
    return f"{amount:,} 元"


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
        rows=payload["rows"],
        segments=SEGMENTS,
        weights=C.WEIGHTS,
        thresholds={
            "vol_surge": C.VOL_SURGE_RATIO,
            "vol_full": C.VOL_FULL_RATIO,
            "near_low_zone": C.NEAR_LOW_ZONE,
            "min_price": int(C.MIN_CLOSE_PRICE),
            "min_turnover": _format_twd(C.MIN_TURNOVER_TWD),
        },
    )


def write_outputs(payload: dict) -> None:
    """寫出 index.html、data/latest.json，並存一份當日封存檔。"""
    html = render_html(payload)

    (ROOT / C.OUTPUT_HTML).write_text(html, encoding="utf-8")
    log.info("已寫出 %s（%d bytes）", C.OUTPUT_HTML, len(html.encode()))

    json_path = ROOT / C.OUTPUT_JSON
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    archive = ROOT / C.ARCHIVE_DIR
    archive.mkdir(parents=True, exist_ok=True)
    (archive / f"{payload['meta']['trade_date']}.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    log.info("已寫出 %s 與當日封存", C.OUTPUT_JSON)
