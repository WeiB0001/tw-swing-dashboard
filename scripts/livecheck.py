# -*- coding: utf-8 -*-
"""
livecheck.py — 用「你的網站實際發布過的排名」驗證，不是重放回測

回測是拿歷史資料重新模擬一遍，難免有各種假設偏差。這支不一樣：
它讀 **data/history/*.json**（每天收盤後真的產出過的那份排名），
再用 data/history/*.csv 的實際後續價格，算出那些排名後來到底賺不賺。

這是最貼近真實的驗證——當天發布了什麼，就用什麼來算，沒有事後諸葛的空間。

成功定義與首頁一致：
    發布後下一個可交易日開盤買 → 第 2～10 日收盤檢查淨利目標
    未達目標第 EXIT_MAX_DAYS 日出場；完整持有期間成熟後才納入統計

另外可以帶入你自己的交易紀錄（從「我的交易」頁匯出的 JSON），
一起比較「模型的排名」與「你實際的成交」差在哪。

用法：
    python scripts/livecheck.py                       # 用全部存檔
    python scripts/livecheck.py --days 30             # 只看最近 30 天
    python scripts/livecheck.py --journal 我的交易.json  # 一併分析自己的交易
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import config as C
import strategy
import risk_stats

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("livecheck")

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / C.ARCHIVE_DIR


# ---------------------------------------------------------------------------
def load_archives(days: int) -> list[dict]:
    """Only this strategy version; retain the earliest publication per signal day."""
    by_date = {}
    for f in sorted(ARCHIVE.glob("20*-*-*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            meta, rows = d.get("meta") or {}, d.get("rows") or []
            if not rows or not strategy.compatible(meta) or meta.get("mode") != "live":
                continue
            date = meta.get("data_date")
            published = meta.get("ranked_at") or meta.get("generated_iso")
            if not date or not published:
                continue
            item = {"date": date, "published_at": published, "rows": rows, "file": f.name}
            if date not in by_date or published < by_date[date]["published_at"]:
                by_date[date] = item
        except (OSError, ValueError, TypeError) as e:
            log.warning("%s 讀取失敗：%s", f.name, e)
    out = [by_date[k] for k in sorted(by_date)]
    return out[-days:] if days > 0 else out


def load_prices() -> dict[str, pd.DataFrame]:
    """讀價格快取，用來算排名發布之後的實際走勢。"""
    d = ROOT / C.HISTORY_CACHE_DIR
    out = {}
    for f in d.glob("*.csv"):
        try:
            df = pd.read_csv(f)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).drop_duplicates("date").sort_values("date")
            out[f.stem] = df.set_index("date")
        except Exception:
            continue
    log.info("讀到 %d 檔價格快取", len(out))
    return out


def outcome(df: pd.DataFrame, sig_date: str, cost: float,
            published_at: str | None = None) -> dict | None:
    idx = int(df.index.searchsorted(pd.Timestamp(sig_date), side="right"))
    if published_at:
        published = pd.Timestamp(published_at)
        published = published.tz_localize(C.TZ) if published.tzinfo is None else published.tz_convert(C.TZ)
        # Do not pretend a delayed publication existed before the next open.
        while idx < len(df):
            market_open = pd.Timestamp(str(df.index[idx])[:10] + " 09:00", tz=C.TZ)
            if market_open > published:
                break
            idx += 1
    return strategy.outcome(df, idx, cost)


def pack(items: list[dict]) -> dict:
    if not items:
        return {"n": 0}
    nets = np.array([x["net"] for x in items])
    wins = sum(1 for x in items if x["success"])
    gw = float(nets[nets > 0].sum())
    gl = float(-nets[nets <= 0].sum())
    return {
        **risk_stats.losses(nets.tolist()),
        **risk_stats.risk_reward(nets.tolist()),
        "n": len(items),
        "success_rate": round(wins / len(items) * 100, 1),
        "win_rate": round(float((nets > 0).mean()) * 100, 1),
        "avg_days": round(float(np.mean([x["days"] for x in items])), 1),
        "ev": round(float(nets.mean()), 2),
        "median": round(float(np.median(nets)), 2),
        "pf": round(gw / gl, 2) if gl > 0 else None,
        "avg_mdd": round(float(np.mean([x["mdd"] for x in items])), 2),
        "worst": round(float(nets.min()), 2),
    }


GROUPS = [("Top 3", 0, 3), ("Top 10", 0, 10), ("11–30", 10, 30), ("31 名以後", 30, 10**6)]


def run(days: int) -> dict:
    archives = load_archives(days)
    prices = load_prices()
    if not archives:
        return {"strategy": strategy.contract(), "days": 0, "pending": 0,
                "period": "", "all": {"n": 0}, "groups": {},
                "reason": "新策略尚無已發布且完整成熟的樣本"}
    if not prices:
        raise RuntimeError("data/history 裡沒有價格快取（*.csv）。")

    buckets = {label: [] for label, _, _ in GROUPS}
    every, pending = [], 0

    for a in archives:
        rows = sorted(a["rows"], key=lambda r: r.get("final_rank") or r.get("rank") or 999)
        for i, r in enumerate(rows):
            df = prices.get(r.get("code"))
            if df is None:
                continue
            o = outcome(df, a["date"], C.TOTAL_COST_PCT, a["published_at"])
            if o is None:
                continue
            if not o["closed"]:
                pending += 1
                continue          # 還沒走完 10 天的不算，避免結果被截斷偏差影響
            every.append(o)
            for label, lo, hi in GROUPS:
                if lo <= i < hi:
                    buckets[label].append(o)

    return {
        "strategy": strategy.contract(),
        "days": len(archives),
        "period": f"{archives[0]['date']} ～ {archives[-1]['date']}",
        "pending": pending,
        "cost_pct": C.TOTAL_COST_PCT,
        "exit_max_days": C.EXIT_MAX_DAYS,
        "groups": {k: pack(v) for k, v in buckets.items()},
        "all": pack(every),
    }


# ---------------------------------------------------------------------------
def analyze_journal(path: str) -> dict | None:
    """分析你從「我的交易」匯出的 JSON，看實際成交跟模型排名差多少。"""
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("交易紀錄讀取失敗：%s", e)
        return None
    trades = d.get("trades") if isinstance(d, dict) else d
    if not isinstance(trades, list) or not trades:
        return None

    closed = [t for t in trades if t.get("status") == "CLOSED"
              and t.get("sell_price") and t.get("buy_price")]
    if not closed:
        return {"total": len(trades), "closed": 0}

    nets, days = [], []
    for t in closed:
        try:
            shares = float(t["shares"])
            cost = float(t["buy_price"]) * shares + float(t.get("buy_fee") or 0)
            proceeds = (float(t["sell_price"]) * shares
                        - float(t.get("sell_fee") or 0) - float(t.get("sell_tax") or 0))
            if cost > 0:
                nets.append((proceeds - cost) / cost * 100)
            if t.get("buy_date") and t.get("sell_date"):
                days.append((pd.Timestamp(t["sell_date"]) - pd.Timestamp(t["buy_date"])).days)
        except Exception:
            continue
    if not nets:
        return {"total": len(trades), "closed": len(closed)}

    a = np.array(nets)
    gw, gl = float(a[a > 0].sum()), float(-a[a <= 0].sum())
    return {
        "total": len(trades), "closed": len(closed),
        "win_rate": round(float((a > 0).mean() * 100), 1),
        "avg": round(float(a.mean()), 2),
        "median": round(float(np.median(a)), 2),
        "pf": round(gw / gl, 2) if gl > 0 else None,
        "best": round(float(a.max()), 2), "worst": round(float(a.min()), 2),
        "avg_days": round(float(np.mean(days)), 1) if days else None,
    }


def report(res: dict, jr: dict | None) -> None:
    print("\n" + "=" * 72)
    print(f"實際發布過的排名追蹤：{res['days']} 天（{res['period']}）")
    print(f"成功定義：隔日開盤買，之後每天收盤檢查，扣 {res['cost_pct']}% 成本後"
          f"第 {C.EXIT_MIN_DAYS}～{C.EXIT_MAX_DAYS} 日淨利至少 {C.EXIT_MIN_PROFIT:g}% 才達標")
    if res["pending"]:
        print(f"（{res['pending']} 筆還沒走完 {res['exit_max_days']} 天，未列入統計）")
    print("=" * 72)
    print(f"{'名次分組':<12}{'N':>6}{'淨利達標率':>11}{'平均天數':>9}"
          f"{'EV':>9}{'PF':>7}{'平均回撤':>10}{'最差':>9}")
    for label, _, _ in GROUPS:
        d = res["groups"].get(label) or {}
        if not d.get("n"):
            continue
        print(f"{label:<12}{d['n']:>6}{d['success_rate']:>10.1f}%{d['avg_days']:>8.1f}"
              f"{d['ev']:>8.2f}%{(d['pf'] if d['pf'] is not None else 0):>7.2f}"
              f"{d['avg_mdd']:>9.2f}%{d['worst']:>8.2f}%")
    a = res["all"]
    if a.get("n"):
        print(f"{'全部':<11}{a['n']:>6}{a['success_rate']:>10.1f}%{a['avg_days']:>8.1f}"
              f"{a['ev']:>8.2f}%{(a['pf'] if a['pf'] is not None else 0):>7.2f}"
              f"{a['avg_mdd']:>9.2f}%{a['worst']:>8.2f}%")

    t3 = (res["groups"].get("Top 3") or {}).get("ev")
    rest = (res["groups"].get("31 名以後") or {}).get("ev")
    if t3 is not None and rest is not None:
        print("\n→ " + (f"✅ 排名有鑑別力：Top 3 的 EV {t3:+.2f}% 高於 31 名以後的 {rest:+.2f}%"
                        if t3 > rest else
                        f"❌ 排名沒有鑑別力：Top 3 的 EV {t3:+.2f}% 沒有優於 31 名以後的 {rest:+.2f}%"))

    if jr:
        print("\n" + "-" * 72)
        print(f"你的實際交易：{jr['total']} 筆，已平倉 {jr['closed']} 筆")
        if jr.get("win_rate") is not None:
            print(f"  勝率 {jr['win_rate']}%｜平均 {jr['avg']:+.2f}%｜中位 {jr['median']:+.2f}%"
                  f"｜PF {jr['pf']}｜平均持有 {jr['avg_days']} 天")
            print(f"  最好 {jr['best']:+.2f}%｜最差 {jr['worst']:+.2f}%")
            if a.get("n"):
                print(f"  對照模型全部訊號：EV {a['ev']:+.2f}%、獲利出場率 {a['success_rate']}%")


def main() -> int:
    ap = argparse.ArgumentParser(description="用實際發布過的排名做驗證")
    ap.add_argument("--days", type=int, default=0, help="只看最近幾天的存檔（0＝全部）")
    ap.add_argument("--journal", default="", help="我的交易匯出的 JSON 檔路徑")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    try:
        res = run(args.days)
    except RuntimeError as e:
        log.error("%s", e)
        return 0

    jr = analyze_journal(args.journal) if args.journal else None
    if jr:
        res["journal"] = jr
    if res.get("days"):
        report(res, jr)

    if not args.no_save:
        p = ROOT / C.LIVECHECK_JSON
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("已寫入 %s", C.LIVECHECK_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
