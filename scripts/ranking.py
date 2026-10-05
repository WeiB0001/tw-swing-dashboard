"""One ranking implementation shared by the website and chronological replay."""
from __future__ import annotations

import config as C
import strategy


def attach(rows: list[dict], tables: dict, regime: str = "sideways") -> None:
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
        for source, buckets in candidates:
            hit = next((b for b in buckets
                        if b["lo"] <= r["score"] < b["hi"]
                        and b.get("samples", 0) >= C.MIN_SAMPLES_SCORE
                        and (source == "bucket" or b.get("pattern") == r["kind"])
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
        known = n >= C.MIN_SAMPLES_SCORE and r.get("hist_success") is not None
        ev = float(r.get("hist_expectancy") or 0)
        pf = r.get("hist_pf")
        quality = known and ev > 0 and pf is not None and pf > 1
        safe = r.get("momentum_tier", 2) < 2
        eligible = bool(quality and safe)
        r["rank_eligible"] = eligible
        r["rank_status"] = ("符合候選條件" if eligible else
                            "樣本不足" if not known else
                            "期望值或獲利因子未過門檻" if not quality else "技術風險偏高")
        r["success_used"] = r.get("hist_success_lower")
        r["rank_score"] = r.get("final_score", 0)
        r["top_fill"] = False  # Never fill the actionable list with failed candidates.
        return (0 if eligible else 1, 0 if quality else 1, 0 if known else 1,
                -float(r.get("hist_success_lower") or 0), -ev,
                -float(r.get("final_score") or 0), r.get("code", ""))
    out = sorted(rows, key=key)
    for i, r in enumerate(out, 1):
        r["final_rank"] = i
    return out
