# -*- coding: utf-8 -*-
"""
backtest.py — V2 歷史回測 / 校準

重要修正：
1) 訊號於 t 日收盤後產生，實際進場改為 t+1 日開盤。
2) 加入滑價 + 買進手續費 + 賣出手續費 + 證交稅的近似交易成本。
3) 每個歷史日用當日成交金額建立 point-in-time 流動性條件，
   不再單純拿「今天的成交熱門股」回看歷史。
4) 產生 score_buckets，供 build.py 用歷史 5 日勝率校準 live ranking。

限制：
- yfinance 的 volume * close 是成交金額近似，不是 TWSE 官方逐日成交金額。
- 若要做到完全 point-in-time 的上市/下市成分，需要另外保存每日全市場 snapshot。
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import config as C
import indicators
import scoring

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backtest_v2")
ROOT = Path(__file__).resolve().parent.parent

# 可依券商自行微調。這裡採保守近似，單位為百分比。
BUY_FEE_PCT = 0.1425 * 0.60      # 手續費假設 6 折
SELL_FEE_PCT = 0.1425 * 0.60
STOCK_TAX_PCT = 0.30             # 股票證交稅；ETF 真實稅率不同
SLIPPAGE_EACH_SIDE_PCT = 0.05

# Point-in-time universe 參數
MIN_TURNOVER_TWD = getattr(C, "MIN_TURNOVER_TWD", 50_000_000)
MAX_DAILY_UNIVERSE = getattr(C, "MAX_UNIVERSE", 220)


def load_universe_history(demo: bool) -> dict[str, pd.DataFrame]:
    if demo:
        import demo_data
        return {code: hist for code, _, _, hist in demo_data.build_dataset()}

    import fetch
    snapshot = fetch.fetch_twse_snapshot()
    if snapshot.empty:
        raise RuntimeError("證交所行情取得失敗，無法取得候選代號。")

    # 仍需一份候選代號來抓歷史資料；但每一天是否納入 ranking，
    # 會在 run_backtest() 用該日成交金額重新判斷。
    universe = fetch.build_universe(snapshot)
    codes = universe["code"].tolist()

    # 核心/科技/ETF 補入候選池，避免只限今天 top turnover
    extra = set(getattr(C, "CORE_WEIGHTED_STOCKS", []))
    extra.update(getattr(C, "TECH_STOCKS", []))
    extra.update(getattr(C, "ETF_STOCKS", []))
    codes = list(dict.fromkeys(codes + sorted(extra)))

    return fetch.fetch_history_yf(codes)


def _is_forced_code(code: str) -> bool:
    return (
        code in set(getattr(C, "CORE_WEIGHTED_STOCKS", []))
        or code in set(getattr(C, "TECH_STOCKS", []))
        or code in set(getattr(C, "ETF_STOCKS", []))
    )


def _cost_adjusted_return(entry_open: float, exit_close: float, is_etf: bool) -> float:
    """
    以百分比近似成本：
    買：手續費 + 滑價
    賣：手續費 + 滑價 + 證交稅
    ETF 稅率用 0.1% 近似。
    """
    tax = 0.10 if is_etf else STOCK_TAX_PCT
    gross = (exit_close / entry_open - 1.0) * 100.0
    costs = BUY_FEE_PCT + SELL_FEE_PCT + tax + 2*SLIPPAGE_EACH_SIDE_PCT
    return gross - costs


def _prepare_frames(hist_map: dict[str, pd.DataFrame], max_hold: int):
    frames = {}
    for code, hist in hist_map.items():
        if hist is None or len(hist) < C.MIN_BARS + max_hold + 15:
            continue
        try:
            df = indicators.compute_frame(hist)
            df.index = pd.to_datetime(df.index)
            # 日成交金額近似：收盤價 * 成交股數
            df["turnover_est"] = (
                pd.to_numeric(df["close"], errors="coerce") *
                pd.to_numeric(df["volume"], errors="coerce")
            )
            frames[code] = df
        except Exception as e:
            log.warning("%s 特徵計算失敗：%s", code, e)
    return frames


def run_backtest(hist_map: dict[str, pd.DataFrame], lookback_days: int) -> dict:
    max_hold = max(C.BACKTEST_HOLD_DAYS)
    frames = _prepare_frames(hist_map, max_hold)
    if not frames:
        raise RuntimeError("沒有足夠長的歷史資料可以回測。")

    log.info("候選歷史標的：%d 檔", len(frames))

    all_dates = sorted(set().union(*[set(df.index) for df in frames.values()]))
    # 需多留一天給 next-day open entry，再留 max_hold 給 exit
    usable = all_dates[C.MIN_BARS: len(all_dates) - max_hold - 1]
    if lookback_days > 0:
        usable = usable[-lookback_days:]
    if not usable:
        raise RuntimeError("可回測交易日不足。")

    log.info("回測期間：%s ～ %s（%d 個交易日）",
             str(usable[0])[:10], str(usable[-1])[:10], len(usable))

    signals = []

    for n, day in enumerate(usable, 1):
        candidates = []

        # 先依當日成交金額建立 point-in-time 候選
        liquidity_rows = []
        for code, df in frames.items():
            if day not in df.index:
                continue
            pos = df.index.get_loc(day)
            if not isinstance(pos, (int, np.integer)):
                continue
            if pos < C.MIN_BARS - 1 or pos + max_hold + 1 >= len(df):
                continue
            turn = float(df["turnover_est"].iloc[pos])
            if np.isfinite(turn):
                liquidity_rows.append((turn, code, pos))

        liquidity_rows.sort(reverse=True)
        top_turn_codes = {code for _, code, _ in liquidity_rows[:getattr(C, "TOP_N_BY_TURNOVER", 100)]}

        daily_allowed = []
        for turn, code, pos in liquidity_rows:
            if turn >= MIN_TURNOVER_TWD and (code in top_turn_codes or _is_forced_code(code)):
                daily_allowed.append((turn, code, pos))
            if len(daily_allowed) >= MAX_DAILY_UNIVERSE:
                break

        for turn, code, pos in daily_allowed:
            df = frames[code]
            f = indicators.features_at(df, pos)
            if not f:
                continue

            res = scoring.score_stock(f)
            rule_score = float(res.get("rule_score", res["score"]))
            if rule_score < C.BACKTEST_MIN_SCORE:
                continue

            # 訊號在 t 日收盤後才知道，因此 t+1 開盤進場
            entry_pos = pos + 1
            entry = float(df["open"].iloc[entry_pos])
            if not np.isfinite(entry) or entry <= 0:
                continue

            fwd = {}
            is_etf = getattr(C, "is_etf", lambda x: x.startswith("00"))(code)

            for h in C.BACKTEST_HOLD_DAYS:
                # 持有 h 個交易日：entry day 算第 1 天，因此 exit 為 entry_pos+h-1 收盤
                exit_pos = entry_pos + h - 1
                if exit_pos >= len(df):
                    continue
                exit_px = float(df["close"].iloc[exit_pos])
                trough = float(df["low"].iloc[entry_pos:exit_pos+1].min())

                gross_ret = (exit_px / entry - 1.0) * 100.0
                net_ret = _cost_adjusted_return(entry, exit_px, is_etf)
                mdd = (trough / entry - 1.0) * 100.0

                fwd[h] = {
                    "ret": net_ret,
                    "gross_ret": gross_ret,
                    "mdd": mdd,
                }

            if len(fwd) != len(C.BACKTEST_HOLD_DAYS):
                continue

            candidates.append({
                "date": str(day)[:10],
                "entry_date": str(df.index[entry_pos])[:10],
                "code": code,
                "score": rule_score,
                "rule_score": rule_score,
                "risk": res["risk"],
                "confirmation": res.get("confirmation", 0),
                "rr_ratio": res["rr_ratio"],
                "downside_pct": res["downside_pct"],
                "breakdown": res["breakdown"],
                "hist_winrate": None,
                "hist_avg_return": None,
                "turnover_est": round(turn),
                "fwd": fwd,
            })

        if not candidates:
            continue

        candidates.sort(key=scoring.sort_key)
        for rank, row in enumerate(candidates, 1):
            row["rank"] = rank
            signals.append(row)

        if n % 20 == 0:
            log.info("進度 %d/%d（累積 %d 筆訊號）", n, len(usable), len(signals))

    if not signals:
        raise RuntimeError("回測期間沒有達標訊號。")

    return {
        "version": 2,
        "generated_at": datetime.now(C.TZ).strftime("%Y-%m-%d %H:%M"),
        "period": f"{str(usable[0])[:10]} ～ {str(usable[-1])[:10]}",
        "trading_days": len(usable),
        "universe_size": len(frames),
        "total_signals": len(signals),
        "primary_hold_days": 5,
        "entry_rule": "signal day close -> next trading day open",
        "cost_assumptions": {
            "buy_fee_pct": BUY_FEE_PCT,
            "sell_fee_pct": SELL_FEE_PCT,
            "stock_tax_pct": STOCK_TAX_PCT,
            "etf_tax_pct": 0.10,
            "slippage_each_side_pct": SLIPPAGE_EACH_SIDE_PCT,
        },
        "topk": _stats_by_topk(signals),
        "score_buckets": _stats_by_bucket(signals),
        "note": (
            "V2：次日開盤進場、已扣近似交易成本/滑價，並使用歷史當日成交金額近似建立 point-in-time universe。"
            "候選代號本身仍受目前可抓到的股票歷史資料限制，尚非完整上市/下市成分資料庫。"
        ),
    }


def _describe(rets: list[float], mdds: list[float]) -> dict:
    if not rets:
        return {
            "samples": 0, "win_rate": None, "avg_return": None,
            "median_return": None, "max_gain": None, "max_loss": None,
            "avg_win": None, "avg_loss": None, "payoff": None,
            "profit_factor": None, "avg_mdd": None, "worst_mdd": None,
        }

    a = np.array(rets, dtype=float)
    m = np.array(mdds, dtype=float)
    wins, losses = a[a > 0], a[a <= 0]
    gross_win = wins.sum()
    gross_loss = -losses.sum()

    return {
        "samples": int(len(a)),
        "win_rate": round(float((a > 0).mean()*100), 1),
        "avg_return": round(float(a.mean()), 2),
        "median_return": round(float(np.median(a)), 2),
        "max_gain": round(float(a.max()), 2),
        "max_loss": round(float(a.min()), 2),
        "avg_win": round(float(wins.mean()), 2) if len(wins) else 0.0,
        "avg_loss": round(float(losses.mean()), 2) if len(losses) else 0.0,
        "payoff": round(float(wins.mean()/abs(losses.mean())), 2)
                  if len(wins) and len(losses) and losses.mean() != 0 else None,
        "profit_factor": round(float(gross_win/gross_loss), 2) if gross_loss > 0 else None,
        "avg_mdd": round(float(m.mean()), 2),
        "worst_mdd": round(float(m.min()), 2),
    }


def _stats_by_topk(signals: list[dict]) -> dict:
    out = {}
    for k in C.BACKTEST_TOP_K:
        sub = [s for s in signals if s["rank"] <= k]
        out[f"top{k}"] = {
            str(h): _describe(
                [s["fwd"][h]["ret"] for s in sub],
                [s["fwd"][h]["mdd"] for s in sub],
            )
            for h in C.BACKTEST_HOLD_DAYS
        }

    out["all"] = {
        str(h): _describe(
            [s["fwd"][h]["ret"] for s in signals],
            [s["fwd"][h]["mdd"] for s in signals],
        )
        for h in C.BACKTEST_HOLD_DAYS
    }
    return out


def _stats_by_bucket(signals: list[dict]) -> list[dict]:
    """
    依 rule_score 分桶，供 live build 做 empirical calibration。
    額外存 gross/net，讓你能看到交易成本後是否還有 edge。
    """
    h = 5
    out = []
    for lo, hi in C.BACKTEST_SCORE_BUCKETS:
        sub = [s for s in signals if lo <= s["rule_score"] < hi]
        if not sub:
            out.append({
                "lo": lo, "hi": hi, "samples": 0,
                "win_rate": None, "avg_return": None, "gross_avg_return": None,
                "hold_days": h,
            })
            continue

        net = [s["fwd"][h]["ret"] for s in sub]
        gross = [s["fwd"][h]["gross_ret"] for s in sub]
        mdd = [s["fwd"][h]["mdd"] for s in sub]
        d = _describe(net, mdd)

        out.append({
            "lo": lo,
            "hi": hi,
            "samples": d["samples"],
            "win_rate": d["win_rate"],
            "avg_return": d["avg_return"],
            "gross_avg_return": round(float(np.mean(gross)), 2),
            "median_return": d["median_return"],
            "profit_factor": d["profit_factor"],
            "hold_days": h,
        })
    return out


def print_report(bt: dict, demo: bool) -> None:
    print("\n" + "="*72)
    print(f"V2 回測期間 {bt['period']}｜{bt['universe_size']} 檔｜{bt['total_signals']} 筆")
    print("進場：訊號隔日開盤｜報酬：已扣近似成本與滑價")
    if demo:
        print("⚠️ DEMO 模擬資料只驗證流程，沒有預測意義。")
    print("="*72)

    for k in ["top1", "top3", "top5", "top10", "all"]:
        if k not in bt["topk"]:
            continue
        print(f"\n■ {k.upper()}")
        for h in C.BACKTEST_HOLD_DAYS:
            d = bt["topk"][k][str(h)]
            if not d["samples"]:
                continue
            print(
                f"{h:>2}日｜n={d['samples']:>5}｜勝率 {d['win_rate']:>5.1f}%｜"
                f"平均 {d['avg_return']:>+6.2f}%｜中位 {d['median_return']:>+6.2f}%｜"
                f"PF {(d['profit_factor'] or 0):>4.2f}"
            )

    print("\n■ Rule score → 實際 5 日勝率")
    for b in bt["score_buckets"]:
        if b["samples"]:
            print(
                f"{b['lo']:>3}–{b['hi']:<3}｜n={b['samples']:>5}｜"
                f"勝率 {b['win_rate']:>5.1f}%｜淨平均 {b['avg_return']:>+6.2f}%"
            )
        else:
            print(f"{b['lo']:>3}–{b['hi']:<3}｜無樣本")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description="V2 歷史回測與勝率校準")
    ap.add_argument("--demo", action="store_true", help="用模擬資料驗證流程")
    ap.add_argument("--days", type=int, default=0, help="只回測最近 N 個交易日（0=全部）")
    ap.add_argument("--no-save", action="store_true", help="不要寫 data/backtest.json")
    args = ap.parse_args()

    hist_map = load_universe_history(args.demo)
    bt = run_backtest(hist_map, args.days)
    bt["mode"] = "demo" if args.demo else "live"
    print_report(bt, args.demo)

    if args.no_save or args.demo:
        log.info("未寫入 %s", C.BACKTEST_JSON)
        return 0

    path = ROOT / C.BACKTEST_JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bt, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("已寫入 %s；下次 build 會用它校準排名", C.BACKTEST_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
