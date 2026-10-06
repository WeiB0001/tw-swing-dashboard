import copy
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import backtest
import build
import config as C
import execution
import fetch
import indicators
import livecheck
import prediction
import ranking
import render
import risk_stats
import stock_history
import strategy
from test_target_strategy import mature


def sample(code, net, day, days=10, **extra):
    date = pd.Timestamp("2025-01-01") + pd.Timedelta(days=day)
    return {"code": code, "date": str(date)[:10], "score": 55., "pattern": "x", "regime": "bull",
            "replay": {"code": code, "score": 55., "kind": "x", "regime": "bull", "atr_pct": 3.},
            "exit": {"closed": True, "entered": True, "net": net, "days": days,
                     "success": True, "mdd": min(net, -1.),
                     "label_end": str(date + pd.Timedelta(days=21))[:10], **extra}}


class StrictHorizonTests(unittest.TestCase):
    def test_day_eleven_gain_is_real_profit_but_not_target_credit(self):
        h = mature([100.] * 10 + [105.] * 5)
        h["volume"] = 1000
        h.iloc[9, h.columns.get_loc("volume")] = 0
        h.iloc[10, h.columns.get_loc("open")] = 105.
        ex = execution.simulate(h, 0)
        self.assertEqual(ex["days"], 11)
        self.assertGreater(ex["net"], 3)
        self.assertFalse(ex["success"])
        self.assertTrue(ex["late_exit"])
        s = sample("A", ex["net"], 1, days=11)
        for report in [backtest._pack([s]), stock_history.build([s])["by_code"]["A"], livecheck.pack([s["exit"]])]:
            self.assertEqual(report["success_rate"], 0)
            self.assertEqual(report["risk_reward_score"], 0)
            self.assertIsNone(report["avg_target_gain"])
            self.assertEqual(report["late_exits"], 1)
        self.assertGreater(backtest._pack([s])["expectancy"], 3)
        self.assertGreater(livecheck.pack([s["exit"]])["ev"], 3)

    def test_exact_target_day_two_and_ten_and_missing_days(self):
        for days in [2, 10]:
            self.assertTrue(strategy.success(3., days))
        for days in [None, 1, 11]:
            self.assertFalse(strategy.success(3., days))
        self.assertFalse(strategy.success(2.999, 10))
        self.assertFalse(strategy.outcome_success({"net": 8., "success": True}))

    def test_late_loss_and_genuine_large_gap_are_not_erased(self):
        signals = [sample("A", 4., i, days=11) for i in range(30)]
        signals += [sample("B", -20., i, days=12) for i in range(30)]
        result = backtest._pack(signals)
        self.assertEqual(result["expectancy"], -8.)
        self.assertEqual(result["loss_rate"], 50.)
        self.assertEqual(result["success_rate"], 0.)
        self.assertEqual(result["risk_reward_lower"], -15.)
        self.assertEqual(result["worst_net"], -20.)


class PredictionTests(unittest.TestCase):
    def training(self):
        return [sample("A", 6., i) for i in range(10)] + [sample("B", -2., i) for i in range(120)]

    def test_excludes_own_stock_from_peers_and_shrinks_small_samples(self):
        model = prediction.fit(self.training())
        row = self.training()[0]["replay"]
        result = prediction.predict(row, model)
        self.assertEqual(result["peer_samples"], 120)
        self.assertEqual(result["own_samples"], 10)
        self.assertAlmostEqual(result["target_pct"], 100/6, places=2)
        self.assertAlmostEqual(result["loss_pct"], 500/6, places=2)
        self.assertLessEqual(result["own_weight_pct"], 25)
        self.assertEqual(prediction.predict(dict(row, code="NEW"), model)["own_samples"], 0)

    def test_asset_separation_date_diversity_and_missing_model(self):
        model = prediction.fit(self.training())
        self.assertIsNone(prediction.predict({"code": "0050"}, model))
        self.assertIsNone(prediction.predict({"code": "A", "feature_quality_ok": False}, model))
        clustered = prediction.fit([sample("B", 5., 1) for _ in range(500)])
        self.assertIsNone(prediction.predict({"code": "A"}, clustered))
        rows = [{"code": "A"}, {"code": "B"}]
        prediction.attach(rows, None)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r["prediction"] is None for r in rows))

    def test_future_labels_and_overlapping_windows_cannot_change_model(self):
        train = self.training()
        cutoff = "2025-05-01"
        late = sample("A", 100., 119)
        test = sample("A", 1000., 150)
        prior = backtest.training_before(train + [late, test], cutoff)
        self.assertTrue(all(s["exit"]["label_end"] < cutoff for s in prior))
        a = prediction.fit(prior)
        late["exit"]["net"] = -999.
        test["exit"]["net"] = -9999.
        self.assertEqual(a, prediction.fit(backtest.training_before(train + [late, test], cutoff)))
        self.assertIsNone(prediction.predict({"code": "A", "quote_date": "2025-01-01"}, a))

    def test_top_five_ten_are_measured_when_all_fail_eligibility(self):
        rows, lookup = [], {}
        for i in range(12):
            code = str(2000+i)
            s = sample(code, -2. if i % 2 else 4., 150)
            lookup[code] = s
            rows.append({"code": code, "score": i, "research_eligible": False,
                         "prediction_rank": 12-i, "hist_risk_reward": -1.,
                         "prediction": {"target_pct": 40., "loss_pct": 60.}})
        lookup["2011"]["exit"] = {"closed": False, "entered": False}
        lookup["2010"]["exit"] = {"closed": False, "entered": True}
        state = {}
        prediction.evaluate_day(state, rows, lookup, {"target_pct": 50., "loss_pct": 50.}, 1)
        report = prediction.summarize(state, backtest._pack)
        self.assertTrue(report["available"])
        self.assertFalse(report["active_in_main_rank"])
        d = report["comparisons"]["individual_top5"]
        self.assertEqual((d["selected_signals"], d["samples"], d["unfilled"], d["unresolved"]), (5, 3, 1, 1))
        self.assertEqual(report["comparisons"]["current_top10"]["selected_signals"], 10)
        self.assertEqual(len(rows), 12)

    def test_brier_uses_frozen_training_baseline_and_reports_reliability(self):
        records = [{"date": "2025-06-01", "prediction": {"target_pct": p}, "target": y,
                    "baseline": {"target_pct": 50.}} for p, y in [(80., True), (20., False)]]
        r = prediction.calibration_report(records, "target_pct", "target")
        self.assertEqual(r["brier"], .04)
        self.assertEqual(r["baseline_brier"], .25)
        self.assertEqual(r["signal_dates"], 1)
        self.assertEqual(sum(b["samples"] for b in r["bins"]), 2)


class HistoryQualityTests(unittest.TestCase):
    def test_invalid_past_bars_disable_estimates_without_removing_reference_row(self):
        h = pd.DataFrame({"open": 100., "high": 101., "low": 99., "close": 100., "volume": 1000.},
                         index=pd.bdate_range("2025-01-01", periods=110))
        h.iloc[50, h.columns.get_loc("low")] = 105.
        f = indicators.compute_features(h)
        self.assertFalse(f["feature_quality_ok"])
        row = {"code": "A", "score": 60., "feature_quality_ok": False}
        ranking.attach([row], {})
        self.assertIsNone(row.get("hist_risk_reward"))
        self.assertEqual(len(render.reference_rows([row])), 1)
        self.assertGreater(execution.audit({"A": h})["invalid_bars"], 0)

    def test_extended_history_backfills_even_current_cache_and_does_not_repeat_for_new_listing(self):
        now = datetime(2026, 10, 6, 17, tzinfo=C.TZ)
        h = pd.DataFrame({"open": 100., "high": 101., "low": 99., "close": 100., "volume": 1000.},
                         index=pd.bdate_range("2026-01-01", "2026-10-06"))
        with tempfile.TemporaryDirectory() as folder, patch.object(fetch, "_CACHE_DIR", Path(folder)), \
                patch.object(fetch, "load_cache", return_value=h), patch.object(fetch, "save_cache"), \
                patch.object(fetch, "datetime") as clock, patch.object(fetch.time, "sleep"), \
                patch.object(fetch, "_finmind_get", return_value=[]) as api:
            clock.now.return_value = now
            fetch.fetch_history(["A"])
            self.assertLess(api.call_args.args[0]["start_date"], "2024-01-01")
            fetch.fetch_history(["A"])
            self.assertEqual(api.call_count, 1)

    def test_api_body_error_is_not_successful_empty_history(self):
        with patch.object(fetch.requests, "get") as get:
            get.return_value.status_code = 200
            get.return_value.json.return_value = {"status": 402, "msg": "request failed"}
            self.assertIsNone(fetch._finmind_get({"dataset": "TaiwanStockPrice"}))


if __name__ == "__main__":
    unittest.main()
