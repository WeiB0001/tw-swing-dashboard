"""One ranking implementation shared by the website and chronological replay."""
from __future__ import annotations

import config as C
import strategy


def attach(rows: list[dict], tables: dict, regime: str = "sideways", use_overseas=False) -> None:
    baseline = (tables.get("overall") or {}).get("success_rate", 0) or 0
    for r in rows:
        for key in list(r):
            if key.startswith("hist_"):
                r[key] = None
        r["hist_confidence"] = 0
        r["hist_samples"] = 0
        r["oos_reliability"] = None
        candidates = [
            ("regime", tables.get("regime_buckets", [])),
            ("pattern", tables.get("pattern_buckets", [])),
            ("bucket", tables.get("score_buckets", [])),
        ]
        if use_overseas:
            candidates.insert(0, ("overseas", tables.get("overseas_buckets", [])))
        for source, buckets in candidates:
            hit = next((b for b in buckets
                        if b["lo"] <= r["score"] < b["hi"]
                        and b.get("samples", 0) >= C.MIN_SAMPLES_SCORE
                        and b.get("signal_dates", 0) >= C.MIN_CALIBRATION_DATES
                        and b.get("ev_lower") is not None
                        and b.get("risk_reward_lower") is not None
                        and b.get("risk_reward_score") is not None
                        and (source in ("bucket", "overseas") or b.get("pattern") == r["kind"])
                        and (source != "overseas" or (b.get("overseas_group") == r.get("overseas_group")
                             and b.get("overseas_state") == r.get("overseas_state")
                             and r.get("overseas_context", {}).get("available")))
                        and (source != "regime" or b.get("regime") == r.get("regime", regime))), None)
            if hit is None:
                continue
            n = int(hit["samples"])
            successes = int(hit.get("successes", 0))
            # Shrink once toward the training-pool target rate, never a fixed 50%.
            rate = 100 * (successes + C.SHRINK_K * baseline / 100) / (n + C.SHRINK_K)
            ev = float(hit.get("expectancy") or 0)
            r.update(hist_success=round(rate, 1), hist_success_raw=hit.get("success_rate"),
                     hist_successes=successes, hist_samples=n,
                     hist_success_lower=round(strategy.wilson_lower(successes, n), 2),
                     hist_calibrated=hit.get("calibrated_win_rate"), hist_raw=hit.get("win_rate"),
                     hist_expectancy=round(ev * n / (n + C.SHRINK_K), 3),
                     hist_ev_lower=round(float(hit["ev_lower"]) * n / (n + C.SHRINK_K), 3),
                     hist_risk_reward=round(float(hit["risk_reward_score"]) * n / (n + C.SHRINK_K), 3),
                     hist_risk_reward_raw=hit["risk_reward_score"],
                     hist_risk_reward_lower=round(float(hit["risk_reward_lower"]) * n / (n + C.SHRINK_K), 3),
                     hist_avg_target_gain=hit.get("avg_target_gain"),
                     hist_avg_loss_magnitude=hit.get("avg_loss_magnitude"),
                     hist_target_gain_component=hit.get("target_gain_component"),
                     hist_loss_component=hit.get("loss_component"),
                     hist_signal_dates=hit.get("signal_dates"),
                     hist_losses=hit.get("losses"), hist_loss_rate=hit.get("loss_rate"),
                     hist_loss_rate_upper=hit.get("loss_rate_upper"),
                     hist_worst_net=hit.get("worst_net"), hist_tail_mean=hit.get("tail_mean_5pct"),
                     hist_expectancy_raw=ev, hist_avg_return=hit.get("avg_return"),
                     hist_pf=hit.get("profit_factor"), hist_mdd=hit.get("avg_mdd"),
                     hist_avg_days=hit.get("avg_days"), hist_avg_days_win=hit.get("avg_days_win"),
                     hist_avg_win=hit.get("avg_win"), hist_avg_loss=hit.get("avg_loss"),
                     hist_source=source, hist_basis="HISTORICAL",
                     hist_confidence=min(3, next((stars for need, stars in C.CONFIDENCE_TIERS if n >= need), 0)))
            break


def sort(rows: list[dict]) -> list[dict]:
    def key(r):
        n = int(r.get("hist_samples") or 0)
        known = n >= C.MIN_SAMPLES_SCORE and r.get("hist_risk_reward") is not None
        score = float(r.get("hist_risk_reward") or 0)
        ev = float(r.get("hist_expectancy") or 0)
        pf = r.get("hist_pf")
        pf_ok = (pf is not None and pf > 1) or (r.get("hist_losses") == 0 and ev > 0)
        quality = (known and score > 0 and (r.get("hist_risk_reward_lower") or 0) > 0
                   and ev > 0 and (r.get("hist_ev_lower") or 0) > 0
                   and (r.get("hist_signal_dates") or 0) >= C.MIN_CALIBRATION_DATES
                   and r.get("hist_loss_rate_upper") is not None and pf_ok)
        safe = r.get("momentum_tier", 2) < 2
        eligible = bool(quality and safe)
        r["research_eligible"] = eligible
        r["rank_eligible"] = False  # Entry eligibility is assigned only by apply_policy.
        r["trade_eligible"] = False
        r["research_status"] = ("通過研究門檻" if eligible else
                            "樣本不足" if not known else
                            "風險報酬或不確定性未過門檻" if not quality else "技術風險偏高")
        r["rank_status"] = r["research_status"] + " · 僅供研究"
        r["success_used"] = r.get("hist_success_lower")
        r["rank_score"] = r.get("hist_risk_reward")
        r["top_fill"] = False  # Never fill the actionable list with failed candidates.
        # Order by the published composite score. Eligibility remains separate.
        return (0 if known else 1, -score,
                -float(r.get("hist_risk_reward_lower") or 0),
                float(r.get("hist_loss_rate_upper") if r.get("hist_loss_rate_upper") is not None else 100),
                -ev,
                -float(r.get("hist_success_lower") or 0),
                -float(r.get("final_score") or 0), r.get("code", ""))
    out = sorted(rows, key=key)
    for i, r in enumerate(out, 1):
        r["final_rank"] = i
    return out


def research_validated(bt: dict | None) -> bool:
    if not strategy.compatible(bt) or bt.get("mode") != "live":
        return False
    wf = bt.get("walk_forward") or {}
    quality = bt.get("data_quality") or {}
    if quality.get("issue_count", 0) or quality.get("unresolved_exits", 0) or quality.get("market_regime_available") is False:
        return False
    return bool(wf.get("available") and (wf.get("samples") or 0) >= 100
                and (wf.get("ev_lower") or 0) > 0
                and (wf.get("risk_reward_lower") or 0) > 0
                and (wf.get("signal_dates") or 0) >= C.MIN_CALIBRATION_DATES
                and (wf.get("profit_factor") or 0) > 1
                and (wf.get("lift_ev") or 0) > 0
                and (wf.get("lift_risk_reward") or 0) > 0
                and (wf.get("positive_folds") or 0) >= 2
                and (wf.get("rank_order") or {}).get("monotonic"))


def apply_policy(rows: list[dict], bt: dict | None, data_date: str | None = None) -> dict:
    policy = strategy.capital_policy()
    policy["strategy"] = strategy.contract()
    policy["research_validated"] = research_validated(bt)
    if not policy["blocked"] and not policy["research_validated"]:
        policy.update(blocked=True, reason="風險報酬綜合排名尚未通過樣本外驗證，暫不配置資金")
        if (bt or {}).get("data_quality", {}).get("issue_count"):
            policy["reason"] += "；行情異常或公司行動待核對"
    for r in rows:
        stale = bool(data_date and r.get("quote_date") != data_date)
        allowed = bool(not policy["blocked"] and r.get("research_eligible") and not stale)
        r["rank_eligible"] = r["trade_eligible"] = allowed
        r["rank_status"] = ("符合模擬候選條件" if allowed else
                            "行情日期不一致，禁止配置" if stale else
                            r["research_status"] + " · " + ("零虧損要求：不配置" if C.REQUIRE_NO_LOSS else "僅供研究"))
    return policy
