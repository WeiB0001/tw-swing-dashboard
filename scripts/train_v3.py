# -*- coding: utf-8 -*-
"""
train_v3.py — V3 walk-forward 勝率模型訓練

目標：
P(隔日開盤進場後，5 個交易日收盤淨報酬 > 0)

流程：
- 台股技術特徵 + V2 rule_score
- 美股 context（嚴格使用訊號當時已知資料）
- 依日期 walk-forward 產生 out-of-sample predictions
- 用 OOF predictions 做 Platt calibration
- 最後用全部歷史資料 fit live model
- 輸出 data/model_v3.json
"""

from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

import config as C
import indicators, scoring
import market_context as MC
import ml_model as MM
import backtest as BT

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("train_v3")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def row_features(code, f, res, ctx):
    sector = C.sector_of(code)
    c = MC.sector_adjusted_context(ctx, sector)
    return {
        "rule_score": float(res.get("rule_score", res["score"])),
        "risk": float(res["risk"]),
        "confirmation": float(res.get("confirmation",0)),
        "rsi": float(f["rsi"]), "vol_ratio": float(f["vol_ratio"]),
        "bias20": float(f["bias20"]), "pos20": float(f["pos20"]),
        "ma20_slope": float(f["ma20_slope"]), "ma60_slope": float(f["ma60_slope"]),
        "atr_pct": float(f["atr_pct"]), "rr_ratio": float(res["rr_ratio"]),
        "ret5": float(f["ret5"]), "macd_hist": float(f["macd_hist"]),
        **{k: float(c.get(k,0.0)) for k in MM.MODEL_FEATURES if k in c},
    }

def build_dataset(hist_map, days=0, context_mode="after_tw_close"):
    max_hold = 5
    frames = {}
    for code,hist in hist_map.items():
        if hist is None or len(hist) < C.MIN_BARS + 20:
            continue
        try:
            df = indicators.compute_frame(hist)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df["turnover_est"] = df["close"]*df["volume"]
            frames[code]=df
        except Exception:
            pass

    us = MC.fetch_us_history(period="5y")
    dates = sorted(set().union(*[set(df.index) for df in frames.values()]))
    usable = dates[C.MIN_BARS:len(dates)-max_hold-1]
    if days>0:
        usable=usable[-days:]

    rows=[]
    for di,day in enumerate(usable,1):
        liquidity=[]
        for code,df in frames.items():
            if day not in df.index: continue
            pos=df.index.get_loc(day)
            if not isinstance(pos,(int,np.integer)) or pos+max_hold+1>=len(df): continue
            turn=float(df["turnover_est"].iloc[pos])
            if np.isfinite(turn): liquidity.append((turn,code,pos))
        liquidity.sort(reverse=True)
        top={code for _,code,_ in liquidity[:getattr(C,"TOP_N_BY_TURNOVER",100)]}
        allowed=[x for x in liquidity if x[0]>=getattr(C,"MIN_TURNOVER_TWD",50_000_000) and (x[1] in top or BT._is_forced_code(x[1]))]
        allowed=allowed[:getattr(C,"MAX_UNIVERSE",220)]
        ctx=MC.context_for_tw_date(us,day,mode=context_mode)

        for _,code,pos in allowed:
            df=frames[code]
            f=indicators.features_at(df,pos)
            if not f: continue
            res=scoring.score_stock(f)
            if float(res.get("rule_score",res["score"])) < 25: continue

            entry_pos=pos+1
            exit_pos=entry_pos+4
            entry=float(df["open"].iloc[entry_pos])
            exit_px=float(df["close"].iloc[exit_pos])
            if not np.isfinite(entry) or entry<=0 or not np.isfinite(exit_px): continue
            is_etf=C.is_etf(code)
            ret=BT._cost_adjusted_return(entry,exit_px,is_etf)
            feat=row_features(code,f,res,ctx)
            rows.append({
                "date":str(day)[:10],"code":code,"y":1 if ret>0 else 0,"ret5d":ret,**feat
            })
        if di%30==0: log.info("dataset %d/%d rows=%d",di,len(usable),len(rows))
    return pd.DataFrame(rows)

def train_walk_forward(df):
    if len(df)<500:
        raise RuntimeError(f"樣本只有 {len(df)}，至少建議 500 筆。")
    df=df.sort_values("date").reset_index(drop=True)
    X=df[MM.MODEL_FEATURES].replace([np.inf,-np.inf],np.nan).fillna(0.0).astype(float)
    y=df["y"].astype(int).values
    dates=pd.to_datetime(df["date"])
    unique=sorted(dates.unique())
    if len(unique)<120:
        raise RuntimeError("交易日太少，無法做可靠 walk-forward。")

    # 5 folds by date, expanding train
    cuts=np.linspace(int(len(unique)*0.45),len(unique),6,dtype=int)
    oof=np.full(len(df),np.nan)
    fold_metrics=[]
    for i in range(5):
        train_end=cuts[i]
        test_end=cuts[i+1]
        train_dates=set(unique[:train_end])
        test_dates=set(unique[train_end:test_end])
        tr=dates.isin(train_dates).values
        te=dates.isin(test_dates).values
        if tr.sum()<300 or te.sum()<50: continue
        scaler=StandardScaler().fit(X[tr])
        clf=LogisticRegression(C=0.35,max_iter=2000,class_weight="balanced")
        clf.fit(scaler.transform(X[tr]),y[tr])
        p=clf.predict_proba(scaler.transform(X[te]))[:,1]
        oof[te]=p
        if len(np.unique(y[te]))>1:
            fold_metrics.append({
                "auc":float(roc_auc_score(y[te],p)),
                "brier":float(brier_score_loss(y[te],p)),
                "n":int(te.sum())
            })

    mask=np.isfinite(oof)
    if mask.sum()<200:
        raise RuntimeError("OOF 樣本不足。")

    # Platt calibration using OOF raw logit
    raw=np.clip(oof[mask],1e-6,1-1e-6)
    logits=np.log(raw/(1-raw)).reshape(-1,1)
    cal=LogisticRegression(C=1e6,max_iter=2000)
    cal.fit(logits,y[mask])
    cal_p=cal.predict_proba(logits)[:,1]

    # final model on all data
    scaler=StandardScaler().fit(X)
    clf=LogisticRegression(C=0.35,max_iter=2000,class_weight="balanced")
    clf.fit(scaler.transform(X),y)

    metrics={
        "oof_n":int(mask.sum()),
        "oof_auc":float(roc_auc_score(y[mask],cal_p)) if len(np.unique(y[mask]))>1 else None,
        "oof_brier":float(brier_score_loss(y[mask],cal_p)),
        "oof_logloss":float(log_loss(y[mask],cal_p)),
        "base_win_rate":float(y[mask].mean()),
        "avg_fold_auc":float(np.mean([x["auc"] for x in fold_metrics])) if fold_metrics else None,
        "folds":fold_metrics,
    }
    model={
        "version":3,
        "target":"next_open_to_5th_close_net_positive",
        "features":MM.MODEL_FEATURES,
        "scaler_mean":scaler.mean_.tolist(),
        "scaler_scale":scaler.scale_.tolist(),
        "coef":clf.coef_[0].tolist(),
        "intercept":float(clf.intercept_[0]),
        "cal_a":float(cal.coef_[0][0]),
        "cal_b":float(cal.intercept_[0]),
        "metrics":metrics,
        "samples":int(len(df)),
        "start_date":str(df["date"].min()),
        "end_date":str(df["date"].max()),
    }
    return model

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--days",type=int,default=750,help="訓練最近 N 個台股交易日")
    ap.add_argument("--demo",action="store_true")
    args=ap.parse_args()
    hist=BT.load_universe_history(args.demo)
    df=build_dataset(hist,args.days)
    out_csv=ROOT/"data"/"training_v3.csv"
    out_csv.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(out_csv,index=False)
    model=train_walk_forward(df)
    p=ROOT/"data"/"model_v3.json"
    p.write_text(json.dumps(model,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(model["metrics"],ensure_ascii=False,indent=2))
    print("saved",p)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
