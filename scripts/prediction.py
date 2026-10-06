"""Versioned, shadow-only hierarchical estimates and chronological diagnostics.

Only completed, purged training labels enter the model. Peers exclude the
security being predicted. Small individual samples borrow strength from peers;
these estimates are not calibrated investment probabilities or trade approval.
"""
from __future__ import annotations

import math
from collections import defaultdict

import config as C
import strategy

VERSION = 1
MIN_PEERS = 100
OWN_PRIOR = 50
MAX_OWN_WEIGHT = .25
LEVELS = ("asset_regime_pattern_vol", "asset_regime_pattern", "asset_regime", "asset")


def keys(row):
    asset = "etf" if str(row.get("code", "")).startswith("00") else "stock"
    regime = row.get("regime") or "unknown"
    pattern = row.get("kind", row.get("pattern", "unknown"))
    atr = row.get("atr_pct")
    vol = "unknown" if atr is None else "low" if atr < 2 else "mid" if atr < 4 else "high"
    return ["|".join(parts) for parts in [(asset, regime, pattern, vol),
            (asset, regime, pattern), (asset, regime), (asset,)]]


def _empty():
    return dict(n=0, wins=0, losses=0, gain=0., loss=0., net=0.)


def _add(stats, ex):
    net = ex["net"]
    target = strategy.outcome_success(ex)
    stats["n"] += 1
    stats["wins"] += int(target)
    stats["losses"] += int(net < -1e-9)
    stats["gain"] += net if target else 0.
    stats["loss"] += max(-net, 0.)
    stats["net"] += net


def fit(signals):
    dates = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    model = {"version": VERSION, "active_in_main_rank": False, "groups": {}, "own": {},
             "last_outcome": None, "samples": 0,
             "settings": {"min_peers": MIN_PEERS, "own_prior": OWN_PRIOR,
                          "max_own_weight": MAX_OWN_WEIGHT}}
    for s in signals:
        ex = s.get("exit") or {}
        if (not ex.get("closed") or not ex.get("entered", True) or not ex.get("label_end")
                or ex.get("days") is None or not math.isfinite(ex.get("net", float("nan")))):
            continue
        row = dict(s.get("replay") or {}, code=s["code"], regime=s.get("regime", "unknown"))
        row.setdefault("kind", s.get("pattern", "unknown"))
        _add(model["own"].setdefault(s["code"], _empty()), ex)
        for key in keys(row):
            group = model["groups"].setdefault(key, {"total": _empty(), "by_code": {}})
            _add(group["total"], ex)
            _add(group["by_code"].setdefault(s["code"], _empty()), ex)
            dates[key][s["date"]][s["code"]] += 1
        model["last_outcome"] = max(model["last_outcome"] or "", ex["label_end"])
        model["samples"] += 1
    for key, group in model["groups"].items():
        group["signal_dates"] = len(dates[key])
        solo = defaultdict(int)
        for securities in dates[key].values():
            if len(securities) == 1:
                solo[next(iter(securities))] += 1
        group["peer_dates"] = {code: len(dates[key])-solo[code] for code in group["by_code"]}
    return model


def predict(row, model):
    if not model or model.get("version") != VERSION or not row.get("feature_quality_ok", True):
        return None
    # A cached model may never leak a later outcome into an earlier quotation.
    day = row.get("quote_date")
    if day and model.get("last_outcome") and model["last_outcome"] > day:
        return None
    code = row["code"]
    own = model["own"].get(code, _empty())
    for level, key in zip(LEVELS, keys(row)):
        group = model["groups"].get(key)
        if not group:
            continue
        excluded = group["by_code"].get(code, _empty())
        peer = {k: group["total"][k] - excluded[k] for k in _empty()}
        peer_dates = group["peer_dates"].get(code, group["signal_dates"])
        if peer["n"] < MIN_PEERS or peer_dates < C.MIN_CALIBRATION_DATES:
            continue
        weight = min(MAX_OWN_WEIGHT, own["n"] / (own["n"] + OWN_PRIOR))
        mix = {k: (1-weight) * peer[k] / peer["n"] +
               (weight * own[k] / own["n"] if own["n"] else 0.)
               for k in ("wins", "losses", "gain", "loss", "net")}
        # Mutually exclusive target/loss/other outcomes remain a valid mixture.
        return {"target_pct": round(100*mix["wins"], 2),
                "loss_pct": round(100*mix["losses"], 2),
                "avg_target_gain": round(mix["gain"]/mix["wins"], 3) if mix["wins"] else None,
                "avg_loss": round(mix["loss"]/mix["losses"], 3) if mix["losses"] else None,
                "expectancy": round(mix["net"], 4),
                "score": round(mix["gain"] - C.LOSS_AVERSION*mix["loss"], 4),
                "own_samples": own["n"], "peer_samples": peer["n"], "peer_dates": peer_dates,
                "own_weight_pct": round(weight*100, 1), "peer_level": level,
                "research_only": True, "calibrated": False}
    return None


def attach(rows, model, regime=None):
    for row in rows:
        features = dict(row)
        if regime:
            features["regime"] = regime
        row["prediction"] = predict(features, model)
    ordered = sorted(rows, key=lambda r: (r.get("prediction") is None,
                     -(r.get("prediction") or {}).get("score", -1e9),
                     (r.get("prediction") or {}).get("loss_pct", 100), r.get("code", "")))
    for rank, row in enumerate(ordered, 1):
        row["prediction_rank"] = rank
    return ordered


def calibration_report(records, field, outcome):
    """Out-of-time diagnostics, never fit on evaluation labels.

    Brier baseline is each fold's training prevalence. Dates, not just trade
    counts, are exposed because overlapping daily signals are dependent.
    """
    if not records:
        return {"available": False, "samples": 0}
    bins = defaultdict(list)
    errors, base_errors = [], []
    for r in records:
        p, y = r["prediction"][field] / 100, float(r[outcome])
        errors.append((p-y)**2)
        base_errors.append((r["baseline"][field]/100-y)**2)
        bins[min(4, int(p*5))].append((p, y, r["date"]))
    return {"available": True, "samples": len(records),
            "signal_dates": len({r["date"] for r in records}),
            "brier": round(sum(errors)/len(errors), 5),
            "baseline_brier": round(sum(base_errors)/len(base_errors), 5),
            "bins": [{"lo": b*20, "hi": (b+1)*20, "samples": len(items),
                      "signal_dates": len({x[2] for x in items}),
                      "predicted_pct": round(100*sum(x[0] for x in items)/len(items), 2),
                      "observed_pct": round(100*sum(x[1] for x in items)/len(items), 2)}
                     for b, items in sorted(bins.items())]}


def evaluate_day(state, ranked, lookup, baseline, fold):
    """Compare on the same full reference pool; qualification never hides it."""
    shared = [r for r in ranked if r.get("prediction") and r.get("hist_risk_reward") is not None]
    alternatives = {"current": shared,
                    "individual": sorted(shared, key=lambda r: r["prediction_rank"]),
                    "technical": sorted(shared, key=lambda r: (-r["score"], r["code"]))}
    for name, ordered in alternatives.items():
        for top in (5, 10):
            picked = [dict(lookup[r["code"]], fold=fold) for r in ordered[:top]]
            state.setdefault("picks", {}).setdefault(f"{name}_top{top}", []).extend(picked)
    for r in shared:
        source = lookup[r["code"]]
        ex = source["exit"]
        if ex.get("closed"):
            state.setdefault("predictions", []).append({"date": source["date"], "prediction": r["prediction"],
                "baseline": baseline, "target": strategy.outcome_success(ex), "loss": ex["net"] < -1e-9})


def summarize(state, pack):
    comparisons = {}
    for key, picks in state.get("picks", {}).items():
        comparisons[key] = {**pack(picks), "selected_signals": len(picks),
            "selected_dates": len({s["date"] for s in picks}),
            "unfilled": sum(s["exit"].get("entered") is False for s in picks),
            "unresolved": sum(not s["exit"].get("closed") and s["exit"].get("entered") is not False for s in picks),
            "folds": [{"fold": f, **pack([s for s in picks if s["fold"] == f])}
                      for f in sorted({s["fold"] for s in picks})]}
    records = state.get("predictions", [])
    return {"version": VERSION, "available": bool(records), "active_in_main_rank": False,
            "basis": "same_dates_same_known_reference_pool_no_eligibility_filter",
            "comparisons": comparisons,
            "target_calibration": calibration_report(records, "target_pct", "target"),
            "loss_calibration": calibration_report(records, "loss_pct", "loss"),
            "note": "時間分段、完整排除重疊標籤；每日候選可能重複，不是獨立交易或資金組合績效。機率僅研究估計，尚未進行獨立驗證的機率校正。"}
