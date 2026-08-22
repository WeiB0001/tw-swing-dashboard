# -*- coding: utf-8 -*-
"""
ml_model.py — V3 線性機率模型

模型採 Logistic Regression + walk-forward OOF calibration。
存成 JSON（非 pickle），GitHub Actions / 靜態 repo 比較穩定。
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np

MODEL_FEATURES = [
    "rule_score","risk","confirmation","rsi","vol_ratio","bias20","pos20",
    "ma20_slope","ma60_slope","atr_pct","rr_ratio","ret5","macd_hist",
    "spx_ret1","spx_ret5","nasdaq_ret1","nasdaq_ret5","sox_ret1","sox_ret5",
    "vix","vix_chg","dxy_ret1","us10y_chg","tsm_ret1","nvda_ret1",
    "risk_on","tech_context","sector_x_sox","sector_x_tsm","sector_x_nvda","sector_is_tech",
]

def sigmoid(z):
    z = np.clip(z, -35, 35)
    return 1/(1+np.exp(-z))

def load_model(path) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def predict_probability(model: dict, feat: dict) -> float:
    mu = model["scaler_mean"]
    sd = model["scaler_scale"]
    coef = model["coef"]
    z = float(model["intercept"])
    for i, name in enumerate(model["features"]):
        x = float(feat.get(name, 0.0) or 0.0)
        s = float(sd[i]) if float(sd[i]) != 0 else 1.0
        z += float(coef[i]) * ((x - float(mu[i])) / s)
    raw = float(sigmoid(z))
    # Platt calibration on OOF logits
    a = float(model.get("cal_a", 1.0))
    b = float(model.get("cal_b", 0.0))
    logit = np.log(np.clip(raw,1e-6,1-1e-6)/np.clip(1-raw,1e-6,1))
    cal = float(sigmoid(a*logit+b))
    return max(0.0, min(1.0, cal))
