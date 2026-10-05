import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pandas as pd
import config as C
import strategy
import ranking
import backtest
import build
import livecheck
import paper_target
import render


def bars(closes, start="2026-01-05"):
    return pd.DataFrame({"open": [100.] * len(closes), "close": closes,
                         "low": [min(99., c) for c in closes],
                         "high": [max(101., c) for c in closes]},
                        index=pd.bdate_range(start, periods=len(closes)))


def bucket(successes=60, n=100, ev=1.0, pf=2.):
    return {"lo": 45, "hi": 65, "samples": n, "successes": successes,
            "success_rate": successes / n * 100, "expectancy": ev,
            "profit_factor": pf, "avg_mdd": -2., "win_rate": 80.}


class TargetOutcomeTests(unittest.TestCase):
    def test_tiny_profit_is_not_target_success(self):
        result = strategy.outcome(bars([100.51] * 10), 0)
        self.assertGreater(result["net"], 0)
        self.assertFalse(result["success"])
        self.assertEqual(result["days"], 10)

    def test_target_is_net_of_cost(self):
        self.assertFalse(strategy.outcome(bars([103.] * 10), 0)["success"])

    def test_exact_three_percent_is_inclusive(self):
        result = strategy.outcome(bars([103.5] * 10), 0)
        self.assertTrue(result["success"])
        self.assertEqual(result["days"], 5)
        self.assertAlmostEqual(result["net"], 3.)

    def test_pre_fifth_day_hit_does_not_qualify(self):
        result = strategy.outcome(bars([104.] * 4 + [101.] * 6), 0)
        self.assertFalse(result["success"])

    def test_day_ten_success_counts(self):
        result = strategy.outcome(bars([100.] * 9 + [103.5]), 0)
        self.assertTrue(result["success"])
        self.assertEqual(result["days"], 10)

    def test_incomplete_winners_and_losers_are_both_pending(self):
        for px in [90., 110.]:
            self.assertEqual(strategy.outcome(bars([px] * 8), 0), {"closed": False})

    def test_five_percent_setting_and_version_invalidation(self):
        old = {"strategy": strategy.contract()}
        with patch.object(C, "EXIT_MIN_PROFIT", 5.):
            self.assertFalse(strategy.compatible(old))
            self.assertFalse(strategy.outcome(bars([103.5] * 10), 0)["success"])
            self.assertTrue(strategy.outcome(bars([105.5] * 10), 0)["success"])

    def test_invalid_prices_are_not_scored(self):
        h = bars([103.5] * 10)
        h.iloc[2, h.columns.get_loc("close")] = float("nan")
        self.assertIsNone(strategy.outcome(h, 0))

    def test_backtest_and_published_result_agree(self):
        h = bars([100.] + [103.5] * 10)
        a = backtest.first_profitable_exit(h, 0, C.TOTAL_COST_PCT)
        b = livecheck.outcome(h, str(h.index[0])[:10], C.TOTAL_COST_PCT)
        self.assertEqual(a, b)

    def test_small_positive_return_kept_in_ev_but_not_success_rate(self):
        a = strategy.outcome(bars([101.5] * 10), 0)
        b = strategy.outcome(bars([104.5] * 10), 0)
        stats = backtest._pack([{"exit": a}, {"exit": b}])
        self.assertEqual(stats["win_rate"], 100)
        self.assertEqual(stats["success_rate"], 50)
        self.assertAlmostEqual(stats["expectancy"], 2.5)


class RankingTests(unittest.TestCase):
    def test_negative_ev_cannot_outrank_eligible_candidate(self):
        good = {"code": "A", "score": 50., "kind": "x", "momentum_tier": 1}
        bad = {"code": "B", "score": 50., "kind": "x", "momentum_tier": 0}
        ranking.attach([good], {"overall": {"success_rate": 40}, "score_buckets": [bucket(50)]})
        ranking.attach([bad], {"overall": {"success_rate": 40}, "score_buckets": [bucket(90, ev=-1., pf=.8)]})
        ranked = ranking.sort([bad, good])
        self.assertEqual(ranked[0]["code"], "A")
        self.assertFalse(bad["rank_eligible"])

    def test_stale_statistics_are_removed(self):
        row = {"code": "A", "score": 50., "kind": "x", "hist_success": 90., "hist_samples": 500}
        ranking.attach([row], {})
        self.assertIsNone(row["hist_success"])
        self.assertFalse(ranking.sort([row])[0]["rank_eligible"])

    def test_small_samples_do_not_enter_auto_candidates(self):
        row = {"code": "A", "score": 50., "kind": "x", "momentum_tier": 0}
        ranking.attach([row], {"score_buckets": [bucket(29, 29)]})
        self.assertFalse(ranking.sort([row])[0]["rank_eligible"])

    def test_empirical_prior_replaces_fifty_percent_prior(self):
        row = {"code": "A", "score": 50., "kind": "x"}
        ranking.attach([row], {"overall": {"success_rate": 10}, "score_buckets": [bucket(50)]})
        self.assertAlmostEqual(row["hist_success"], 36.7)

    def test_wilson_penalizes_small_samples_at_same_hit_rate(self):
        self.assertLess(strategy.wilson_lower(18, 30), strategy.wilson_lower(180, 300))

    def test_walk_forward_purges_future_labels(self):
        inputs = [{"date": "2026-01-02", "exit": {"label_end": "2026-01-19"}},
                  {"date": "2026-01-01", "exit": {"label_end": "2026-01-15"}}]
        self.assertEqual(backtest.training_before(inputs, "2026-01-19"), inputs[1:])

    def test_decile_boundaries_cover_all_ranks_once(self):
        signal = {"exit": {"success": True, "days": 5, "net": 3., "mdd": -1.}}
        rows = [{**signal, "pct": i / 100} for i in range(100)]
        self.assertEqual([x["samples"] for x in backtest.stats_by_decile(rows)], [10, 20, 30, 40])


class PublicationAndPaperTests(unittest.TestCase):
    def test_duplicate_archives_count_once_and_old_version_is_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            for i, hour in enumerate([16, 17, 18]):
                meta = {"strategy": strategy.contract(), "mode": "live", "data_date": "2026-01-05",
                        "ranked_at": f"2026-01-05 {hour}:00:00"}
                Path(folder, f"2026-01-0{i+5}.json").write_text(json.dumps({"meta": meta, "rows": [{"code": "A"}]}))
            Path(folder, "2026-01-04.json").write_text(json.dumps({"meta": {}, "rows": [{"code": "A"}]}))
            with patch.object(livecheck, "ARCHIVE", Path(folder)):
                records = livecheck.load_archives(0)
            self.assertEqual(len(records), 1)
            self.assertIn("16:00", records[0]["published_at"])

    def test_late_publication_does_not_buy_an_already_passed_open(self):
        h = bars([100., 103.5] + [103.5] * 9)
        result = livecheck.outcome(h, "2026-01-04", .5, "2026-01-05T10:00:00+08:00")
        self.assertEqual(result["exit_date"], "2026-01-12")

    def test_paper_matches_outcome_and_cost_ledger(self):
        h = bars([103.5] * 10)
        pf = {"strategy": strategy.contract(), "cash": 100000., "positions": [],
              "pending": [{"code": "A", "name": "A", "signal_date": "2026-01-02",
                           "published_at": "2026-01-02T16:00:00+08:00", "budget": 100000.}],
              "trades": [], "equity": [], "start_index": None, "last_date": None}
        state, summary = paper_target.update(pf, [], "2026-01-16", 100., {"A": h},
                                             "2026-01-16T16:00:00+08:00")
        expected = strategy.outcome(h, 0)
        trade = state["trades"][0]
        self.assertAlmostEqual(trade["net_pct"], expected["net"])
        self.assertEqual(trade["held"], expected["days"])
        self.assertEqual(trade["success"], expected["success"])
        self.assertAlmostEqual(state["cash"] - 100000., trade["pnl"])
        unchanged, _ = paper_target.update(copy.deepcopy(state), [], "2026-01-16", 100., {"A": h})
        self.assertEqual(unchanged, state)

    def test_average_return_alone_does_not_claim_target_accuracy(self):
        wf = {"available": True, "samples": 500, "expectancy": 1.2,
              "profit_factor": 1.5, "lift_ev": .4, "positive_folds": 3,
              "lift_success_pp": -.2}
        self.assertFalse(build._oos_proven({"walk_forward": wf}))
        wf["lift_success_pp"] = .2
        self.assertTrue(build._oos_proven({"walk_forward": wf}))

    def test_missing_statistics_render_without_old_success_rates(self):
        payload = json.loads((Path(__file__).resolve().parents[1] / "data/latest.json").read_text())
        payload["rows"] = payload["rows"][:1]
        build.ranking.attach(payload["rows"], {})
        build.add_momentum(payload["rows"])
        build.add_final_score(payload["rows"])
        payload["rows"] = build.sort_by_final(payload["rows"])
        payload["backtest"] = None
        payload["portfolio"] = None
        payload["livecheck"] = None
        payload["meta"]["has_winrate"] = False
        html = render.render_html(payload)
        self.assertIn("淨利至少 +3%", html)
        self.assertIn("今日沒有符合候選條件的標的", html)
        self.assertNotIn("<em>歷史勝率</em>", html)

    def test_template_compiles(self):
        from jinja2 import Environment, FileSystemLoader
        Environment(loader=FileSystemLoader(str(render.TEMPLATE_DIR))).get_template("dashboard.html.j2")

    def test_old_backtest_file_never_supplies_new_target_rate(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "data").mkdir()
            Path(folder, "data/backtest.json").write_text(json.dumps({"mode": "live", "oos_overall": {"samples": 100}}))
            row = {"score": 50., "kind": "x", "hist_success": 99.}
            with patch.object(build, "ROOT", Path(folder)):
                self.assertIsNone(build.attach_backtest([row]))
            self.assertIsNone(row["hist_success"])


if __name__ == "__main__":
    unittest.main()
