# -*- coding: utf-8 -*-
"""
build.py — V3 live ranking
- 保留 V2 rule_score
- 加入美股 context
- 若 data/model_v3.json 存在，用 walk-forward 訓練出的 Logistic + Platt calibration
  產生 ml_win_prob，並作為主要 ranking score。
"""

from __future__ import annotations
import argparse, json, logging, sys
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import config as C
import fetch, indicators, render, scoring
import market_context as MC
import ml_model as MM

ROOT=Path(__file__).resolve().parent.parent
log=logging.getLogger("build_v3")
logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s")

def _row_feature_dict(code,f,res,ctx):
    sector=C.sector_of(code)
    c=MC.sector_adjusted_context(ctx,sector)
    return {
        "rule_score":float(res.get("rule_score",res["score"])),
        "risk":float(res["risk"]),
        "confirmation":float(res.get("confirmation",0)),
        "rsi":float(f["rsi"]),"vol_ratio":float(f["vol_ratio"]),
        "bias20":float(f["bias20"]),"pos20":float(f["pos20"]),
        "ma20_slope":float(f["ma20_slope"]),"ma60_slope":float(f["ma60_slope"]),
        "atr_pct":float(f["atr_pct"]),"rr_ratio":float(res["rr_ratio"]),
        "ret5":float(f["ret5"]),"macd_hist":float(f["macd_hist"]),
        **{k:float(c.get(k,0.0)) for k in MM.MODEL_FEATURES if k in c},
    }

def build_row(code,name,f,res,ctx,model):
    feat=_row_feature_dict(code,f,res,ctx)
    rule=float(res.get("rule_score",res["score"]))
    if model:
        p=MM.predict_probability(model,feat)*100
        # 保留少量規則分數避免模型在極端 extrapolation 時失控
        final=0.85*p+0.15*rule
        calibrated=True
    else:
        p=None; final=rule; calibrated=False
    return {
        "code":code,"name":name,"sector":C.sector_of(code),"asset_type":C.asset_type(code),
        "close":f["close"],"lot_cost":round(f["close"]*1000),"chg_pct":f["chg_pct"],
        "rsi":f["rsi"],"vol_ratio":f["vol_ratio"],"volume":f["volume"],"vol_ma20":f["vol_ma20"],
        "ma5":f["ma5"],"ma10":f["ma10"],"ma20":f["ma20"],"ma60":f["ma60"],
        "ma20_slope":f["ma20_slope"],"ma60_slope":f["ma60_slope"],"bias20":f["bias20"],
        "pos20":f["pos20"],"pct_above_low20":f["pct_above_low20"],"pct_below_high20":f["pct_below_high20"],
        "pct_below_high5":f["pct_below_high5"],"pct_below_high60":f["pct_below_high60"],
        "atr":f["atr"],"atr_pct":f["atr_pct"],"macd_hist":f["macd_hist"],"target":f["target"],"support":f["support"],
        "score":round(final,1),"rule_score":round(rule,1),"ml_win_prob":round(p,1) if p is not None else None,
        "calibrated":calibrated,"confirmation":res.get("confirmation",0),"opportunity":res["opportunity"],
        "risk":res["risk"],"breakdown":res["breakdown"],"risk_items":res["risk_items"],"stars":res["stars"],
        "risk_level":res["risk_level"],"kind":res["kind"],"headline":res["headline"],"why":res["why"],
        "main_risk":res["main_risk"],"swing_low":res["swing_low"],"swing_high":res["swing_high"],
        "upside_pct":res["upside_pct"],"downside_pct":res["downside_pct"],"rr_ratio":res["rr_ratio"],
        "reasons":res["reasons"],"flags":res["flags"],
        "hist_winrate":None,"hist_samples":None,"hist_avg_return":None,
        "us_context":{k:round(float(ctx.get(k,0)),3) for k in ["sox_ret1","nasdaq_ret1","tsm_ret1","nvda_ret1","vix","risk_on"]},
    }

def attach_backtest(rows):
    path=ROOT/C.BACKTEST_JSON
    if not path.exists(): return None
    try: bt=json.loads(path.read_text(encoding="utf-8"))
    except Exception: return None
    buckets=bt.get("score_buckets",[])
    for r in rows:
        raw=r.get("rule_score",r["score"])
        for b in buckets:
            if b["lo"]<=raw<b["hi"] and b.get("samples",0)>=C.BACKTEST_MIN_SAMPLES:
                r["hist_winrate"]=b.get("win_rate");r["hist_samples"]=b.get("samples");r["hist_avg_return"]=b.get("avg_return");break
    return bt

def _backtest_summary(bt):
    if not bt:return None
    return {"version":bt.get("version",1),"generated_at":bt.get("generated_at"),"period":bt.get("period"),
            "hold_days":bt.get("primary_hold_days"),"topk":bt.get("topk",{}),"samples":bt.get("total_signals")}

def run_live(context_mode="after_tw_close"):
    now=datetime.now(C.TZ)
    snapshot=fetch.fetch_twse_snapshot()
    if snapshot.empty: raise RuntimeError("證交所當日行情取得失敗。")
    universe=fetch.build_universe(snapshot)
    codes=universe["code"].tolist(); name_map=dict(zip(universe["code"],universe["name"]))
    hist_map=fetch.fetch_history_yf(codes)
    missing=[c for c in codes if c not in hist_map]
    if missing: hist_map.update(fetch.fetch_history_finmind(missing[:40]))
    us=MC.fetch_us_history(period="3mo")
    ctx=MC.context_for_tw_date(us,pd.Timestamp(now.date()),mode=context_mode)
    model=MM.load_model(ROOT/"data"/"model_v3.json")
    rows=[]
    trade_date=pd.Timestamp(now.date())
    for _,srow in universe.iterrows():
        code=srow["code"]; hist=hist_map.get(code)
        if hist is None: continue
        try:
            hist=fetch.merge_today_bar(hist,srow,trade_date)
            f=indicators.compute_features(hist)
            if not f: continue
            res=scoring.score_stock(f)
            rows.append(build_row(code,name_map.get(code,code),f,res,ctx,model))
        except Exception as e:
            log.warning("%s failed: %s",code,e)
    bt=attach_backtest(rows)
    rows.sort(key=lambda r:(-r["score"],-float(r.get("hist_avg_return") or -999),r["risk"],-r.get("confirmation",0)))
    top=[r for r in rows if r["score"]>=C.MIN_SCORE_TO_SHOW][:C.TOP_N_DISPLAY]
    return {
        "meta":{"generated_at":now.strftime("%Y-%m-%d %H:%M"),"generated_iso":now.isoformat(timespec="seconds"),
                "trade_date":now.strftime("%Y-%m-%d"),"scanned_count":len(rows),"universe_count":len(universe),
                "top_n":C.TOP_N_BY_TURNOVER,"min_score":C.MIN_SCORE_TO_SHOW,
                "source_note":"TWSE + yfinance 台股 + 美股市場 context","mode":"live",
                "ranking_version":"v3-us-context-ml","context_mode":context_mode,
                "model_loaded":bool(model),"model_metrics":model.get("metrics") if model else None},
        "index":fetch.fetch_market_index(),"backtest":_backtest_summary(bt),"rows":top
    }

def run_demo():
    import demo_data
    return demo_data.build_demo_payload()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--demo",action="store_true")
    ap.add_argument("--preopen",action="store_true",help="隔日台股開盤前刷新，可使用最新完成的美股交易日")
    args=ap.parse_args()
    try:
        payload=run_demo() if args.demo else run_live("preopen" if args.preopen else "after_tw_close")
    except Exception as e:
        log.error("建置失敗：%s",e);return 1
    render.write_outputs(payload);return 0
if __name__=="__main__":
    raise SystemExit(main())
