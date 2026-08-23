# -*- coding: utf-8 -*-
"""
intraday.py — 盤中觸發提醒（獨立於正式流程）

**這支完全不碰排名、勝率、EV、PF、回測。** 它只做一件事：
拿昨天收盤後算好的「明日交易計畫」價位（data/latest.json 裡的 plan），
對照盤中現價，判斷觸發到哪一段，寫成 data/intraday.json 給提醒頁輪詢。

資料來源：證交所 MIS 即時行情（免金鑰）。這是**延遲報價**，
不是逐筆即時，頁面上會標示延遲與最後更新時間。

非交易時段（收盤後、假日）一律寫 market_open=false，頁面顯示「--」，
不會拿收盤價假裝成盤中狀態。

用法（GitHub Actions 盤中每 5 分鐘跑一次）：
    python scripts/intraday.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, time as dtime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests

import config as C

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("intraday")

ROOT = Path(__file__).resolve().parent.parent
MIS_URL = "https://mis.twse.com.tw/stock/api/getStockInfo.jsp"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; tw-swing-dashboard/1.0)",
    "Referer": "https://mis.twse.com.tw/stock/index.jsp",
}


# ---------------------------------------------------------------------------
def is_market_open(now: datetime) -> bool:
    """台股交易時段：週一～週五 09:00–13:30。收盤後與假日都算休市。"""
    if now.weekday() >= 5:
        return False
    return dtime(9, 0) <= now.time() <= dtime(13, 35)


def _num(x):
    try:
        v = float(x)
        return v if v == v else None
    except Exception:
        return None


def fetch_mis(codes: list[str]) -> dict[str, dict]:
    """
    向證交所 MIS 要即時報價。上市用 tse_、上櫃用 otc_，
    第一輪拿不到的代號用另一個前綴再試一次。
    """
    out: dict[str, dict] = {}

    def ask(prefix: str, syms: list[str]) -> None:
        for i in range(0, len(syms), 45):          # 一次問太多會被擋
            chunk = syms[i:i + 45]
            ex = "|".join(f"{prefix}_{c}.tw" for c in chunk)
            try:
                r = requests.get(MIS_URL, params={"ex_ch": ex, "json": "1",
                                                  "delay": "0", "_": int(time.time() * 1000)},
                                 headers=HEADERS, timeout=C.HTTP_TIMEOUT)
                if r.status_code != 200:
                    log.warning("MIS 回應 %s", r.status_code)
                    continue
                for it in (r.json().get("msgArray") or []):
                    code = str(it.get("c", "")).strip()
                    if code:
                        out[code] = it
            except Exception as e:
                log.warning("MIS 取得失敗（%s）：%s", prefix, e)
            time.sleep(0.3)

    ask("tse", codes)
    missing = [c for c in codes if c not in out]
    if missing:
        ask("otc", missing)
    return out


# ---------------------------------------------------------------------------
def status_of(price: float, plan: dict, vol_ratio: float | None) -> tuple[str, str]:
    """
    判斷目前走到哪一段。回傳 (狀態, 說明)。
    順序很重要：先看有沒有觸及停損，再看目標，最後才是觸發價。
    """
    trig = _num(plan.get("trigger"))
    stop = _num(plan.get("stop"))
    t1 = _num(plan.get("target1"))

    if stop and price <= stop:
        return "stopped", "已觸及停損價"
    if t1 and price >= t1:
        return "target", "已達目標一"
    if stop and trig and price <= stop + (trig - stop) * 0.25:
        return "near_stop", "接近停損（距停損不到區間的四分之一）"
    if trig and price >= trig:
        note = "已突破觸發價"
        if vol_ratio is not None:
            note += f"，量能 {vol_ratio:.1f} 倍"
        return "triggered", note
    if trig:
        gap = (trig / price - 1) * 100 if price else 0
        return "waiting", f"未觸發，距觸發價還有 {gap:.1f}%"
    return "waiting", "未觸發"


def build() -> dict:
    now = datetime.now(C.TZ)
    open_now = is_market_open(now)

    payload = {
        "generated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "market_open": open_now,
        "trade_date": now.strftime("%Y-%m-%d"),
        "source": "證交所 MIS 即時行情（延遲報價）",
        "rows": [],
    }

    if not open_now:
        payload["note"] = "目前非交易時段，盤中狀態顯示為 --。收盤後請看首頁的正式排行。"
        log.info("非交易時段，只寫出休市狀態")
        return payload

    # 交易計畫來自昨天收盤後算好的結果，這裡不重算任何分數
    try:
        latest = json.loads((ROOT / C.OUTPUT_JSON).read_text(encoding="utf-8"))
    except Exception as e:
        payload["note"] = "讀不到 data/latest.json，無法取得交易計畫。"
        log.warning("latest.json 讀取失敗：%s", e)
        return payload

    rows = [r for r in latest.get("rows", [])
            if (r.get("plan") or {}).get("available")][: C.INTRADAY_MAX]
    if not rows:
        payload["note"] = "昨天的排行裡沒有可用的交易計畫。"
        return payload

    quotes = fetch_mis([r["code"] for r in rows])
    if not quotes:
        payload["note"] = "即時報價暫時取不到，顯示的是最後一次成功的資料。"
        log.warning("MIS 完全沒有回應")
        return payload

    data_times = []
    for r in rows:
        q = quotes.get(r["code"])
        if not q:
            continue
        price = _num(q.get("z"))                       # 成交價
        if price is None or price <= 0:                # 尚未成交就用最佳買價
            price = _num((q.get("b") or "").split("_")[0])
        if price is None or price <= 0:
            continue

        prev = _num(q.get("y")) or _num(r.get("prev_close"))
        vol = _num(q.get("v"))                          # 累積成交量（張）
        vol_ma = _num(r.get("vol_ma20"))
        vol_ratio = (vol * 1000 / vol_ma) if (vol and vol_ma) else None

        ts = q.get("tlong")
        if ts:
            try:
                data_times.append(datetime.fromtimestamp(int(ts) / 1000, C.TZ))
            except Exception:
                pass

        plan = r["plan"]
        st, desc = status_of(price, plan, vol_ratio)
        ma20 = _num(r.get("ma20"))

        payload["rows"].append({
            "symbol": r["code"], "name": r.get("name", ""),
            "rank": r.get("rank"), "kind": r.get("kind"),
            "price": round(price, 2),
            "chg_pct": round((price / prev - 1) * 100, 2) if prev else None,
            "prev_close": round(prev, 2) if prev else None,
            "open": _num(q.get("o")), "high": _num(q.get("h")), "low": _num(q.get("l")),
            "volume_lots": int(vol) if vol else None,
            "vol_ratio": round(vol_ratio, 2) if vol_ratio else None,
            "trigger": plan.get("trigger"), "stop": plan.get("stop"),
            "target1": plan.get("target1"), "target2": plan.get("target2"),
            "rr": plan.get("rr"),
            "plan_status": plan.get("status"),
            "status": st, "status_text": desc,
            # 依據直接寫出來，不要讓使用者猜這些價位怎麼來的
            "basis": {
                "trigger": "今日前高 + 0.05×ATR（收盤後算出，盤中不變）",
                "stop": "支撐與失效價取較低，並限制在 3×ATR 內",
                "target1": "近壓與觸發價 +1.5×ATR 取較近者",
                "volume": (f"量能門檻＝20 日均量 × {C.VOL_SURGE_RATIO}"
                           f"（20 日均量 {int(vol_ma/1000):,} 張）" if vol_ma else "無均量資料"),
                "ma20": f"MA20 = {ma20:.2f}（收盤後計算）" if ma20 else "",
            },
        })

    if data_times:
        newest = max(data_times)
        payload["data_time"] = newest.strftime("%Y-%m-%d %H:%M:%S")
        payload["delay_seconds"] = max(0, int((now - newest).total_seconds()))
    log.info("盤中提醒：%d 檔，延遲約 %s 秒",
             len(payload["rows"]), payload.get("delay_seconds", "?"))
    return payload


def main() -> int:
    try:
        payload = build()
    except Exception as e:
        log.error("盤中提醒產生失敗：%s", e)
        return 0                                # 失敗也不要讓 Actions 變紅
    try:
        p = ROOT / C.INTRADAY_JSON
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                     encoding="utf-8")
        log.info("已寫出 %s", C.INTRADAY_JSON)
    except Exception as e:
        log.warning("寫入失敗：%s", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
