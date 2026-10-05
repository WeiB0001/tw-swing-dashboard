import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from test_target_strategy import bars, bucket
import config as C
import strategy
import ranking
import risk_stats
import backtest
import paper_target
import tracking
import render


def row(code, ev, success=60.):
    return {"code": code, "name": code, "hist_samples": 200, "hist_expectancy": ev,
            "hist_success": success, "hist_success_lower": success - 10,
            "hist_pf": 2., "hist_ev_lower": ev - .5, "hist_signal_dates": 30,
            "hist_risk_reward": ev, "hist_risk_reward_lower": ev - .5,
            "hist_losses": 40, "hist_loss_rate_upper": 30., "momentum_tier": 0}


def valid_research():
    return {"strategy": strategy.contract(), "mode": "live",
            "walk_forward": {"available": True, "samples": 300, "ev_lower": .5,
                             "risk_reward_lower": .3, "lift_risk_reward": .2,
                             "signal_dates": 50, "profit_factor": 1.5, "lift_ev": .4,
                             "positive_folds": 3, "rank_order": {"monotonic": True}}}


class ExpectedReturnTests(unittest.TestCase):
    def test_higher_return_outranks_higher_target_rate(self):
        ranked = ranking.sort([row("high_hit", 1., 90.), row("high_ev", 4., 60.)])
        self.assertEqual([r["code"] for r in ranked], ["high_ev", "high_hit"])

    def test_risk_status_does_not_break_displayed_ev_order(self):
        risky = row("risky", 5.)
        risky["momentum_tier"] = 2
        ranked = ranking.sort([row("safer", 2.), risky])
        self.assertEqual(ranked[0]["code"], "risky")
        self.assertFalse(ranked[0]["research_eligible"])
        self.assertFalse(ranked[0]["trade_eligible"])

    def test_negative_uncertainty_estimate_fails_research_gate(self):
        r = row("A", 3.)
        r["hist_ev_lower"] = -.1
        self.assertFalse(ranking.sort([r])[0]["research_eligible"])

    def test_one_date_with_many_stocks_does_not_count_as_many_dates(self):
        signals = [{"date": "2026-01-01", "exit": {"net": 5.}} for _ in range(1000)]
        self.assertEqual(risk_stats.expected_return_lower(signals), {"signal_dates": 1, "ev_lower": None, "risk_reward_lower": None})

    def test_constant_returns_preserved_by_block_bootstrap(self):
        signals = [{"date": f"2026-01-{i:02}", "exit": {"net": 3.}} for i in range(1, 31)]
        a = risk_stats.expected_return_lower(signals)
        self.assertEqual(a, risk_stats.expected_return_lower(signals))
        self.assertAlmostEqual(a["ev_lower"], 3.)

    def test_ninety_nine_samples_are_insufficient(self):
        r = {"code": "A", "score": 50, "kind": "x"}
        ranking.attach([r], {"score_buckets": [bucket(90, 99)]})
        self.assertIsNone(r.get("hist_expectancy"))

    def test_negative_and_zero_returns_are_not_hidden(self):
        d = backtest.describe([-10., 0., 3.], [-10., -2., -1.])
        self.assertEqual(d["samples"], 3)
        self.assertEqual(d["losses"], 1)
        self.assertAlmostEqual(d["expectancy"], -7 / 3, places=3)
        self.assertEqual(d["worst_net"], -10.)
        self.assertEqual(d["tail_mean_5pct"], -10.)

    def test_zero_past_losses_is_not_zero_uncertainty(self):
        d = risk_stats.losses([3.] * 100)
        self.assertEqual(d["loss_rate"], 0.)
        self.assertGreater(d["loss_rate_upper"], 0.)

    def test_rank_monotonicity_must_be_observed_out_of_sample(self):
        bad = [{"samples": 100, "expectancy": v} for v in [1.36, 2.01, 2.35, 1.62]]
        self.assertFalse(risk_stats.rank_order_check(bad)["monotonic"])
        good = [{"samples": 100, "expectancy": v} for v in [3., 2., 1., -.5]]
        self.assertTrue(risk_stats.rank_order_check(good)["monotonic"])

    def test_old_contract_and_demo_cannot_validate_research(self):
        bt = valid_research()
        bt["strategy"]["version"] = "net-target-v1"
        self.assertFalse(ranking.research_validated(bt))
        bt = valid_research()
        bt["mode"] = "demo"
        self.assertFalse(ranking.research_validated(bt))


class CapitalPolicyTests(unittest.TestCase):
    @patch.object(C, "REQUIRE_NO_LOSS", True)
    def test_zero_loss_requirement_blocks_even_validated_research(self):
        rows = ranking.sort([row("A", 4.)])
        policy = ranking.apply_policy(rows, valid_research())
        self.assertTrue(policy["research_validated"])
        self.assertTrue(policy["blocked"])
        self.assertTrue(rows[0]["research_eligible"])
        self.assertFalse(rows[0]["trade_eligible"])
        self.assertFalse(rows[0]["rank_eligible"])

    @patch.object(C, "REQUIRE_NO_LOSS", False)
    def test_new_risk_policy_permits_validated_candidates_without_changing_exits(self):
        self.assertFalse(strategy.capital_policy()["blocked"])
        rows = ranking.sort([row("A", 4.)])
        self.assertFalse(ranking.apply_policy(rows, valid_research())["blocked"])
        self.assertTrue(rows[0]["trade_eligible"])
        self.assertIsNone(C.STOP_LOSS_NET_PCT)

    @patch.object(C, "REQUIRE_NO_LOSS", False)
    @patch.object(C, "STOP_LOSS_NET_PCT", 2.)
    def test_unvalidated_or_stale_data_cannot_get_entry_permission(self):
        rows = ranking.sort([row("A", 4.)])
        self.assertTrue(ranking.apply_policy(rows, None)["blocked"])
        rows[0]["quote_date"] = "2026-01-01"
        policy = ranking.apply_policy(rows, valid_research(), "2026-01-02")
        self.assertFalse(policy["blocked"])
        self.assertFalse(rows[0]["trade_eligible"])

    def test_paper_cancels_pending_orders_even_on_same_date(self):
        pf = tracking._blank_portfolio()
        pf.update(strategy=strategy.contract(), last_date="2026-01-05")
        pf["pending"] = [{"code": "A", "entry_approved": True, "budget": 10000.}]
        state, summary = paper_target.update(pf, [], "2026-01-05", None, {})
        self.assertEqual(state["pending"], [])
        self.assertEqual(summary["trades"], 0)
        self.assertEqual(len(state["cancelled_pending"]), 1)

    def test_paper_validation_gate_cannot_be_bypassed_by_row_flags(self):
        rows = [dict(row("A", 4.), trade_eligible=True, rank_eligible=True)]
        state, _ = paper_target.update({}, rows, "2026-01-05", None, {})
        self.assertEqual(state["pending"], [])
        self.assertEqual(state["positions"], [])

    def test_version_change_records_cancellation_of_old_pending_order(self):
        old = {"strategy": {"version": "old"}, "pending": [{"code": "A"}]}
        state, _ = paper_target.update(old, [], "2026-01-05", None, {})
        self.assertEqual(state["pending"], [])
        self.assertIn("策略版本更換", state["cancelled_pending"][0]["cancel_reason"])

    def test_render_cannot_reenable_allocation_from_stale_payload(self):
        p = json.loads((Path(__file__).resolve().parents[1] / "data/latest.json").read_text())
        p["rows"] = p["rows"][:1]
        r = p["rows"][0]
        r["trade_eligible"] = True
        r["rank_eligible"] = True
        r["hist_success"] = None
        p["capital_policy"] = {"blocked": False, "reason": "stale"}
        p["backtest"] = p["portfolio"] = p["livecheck"] = None
        html = render.render_html(p)
        self.assertIn('data-eligible="0"', html)
        self.assertIn('disabled title="目前僅供研究，不允許配置"', html)
        self.assertIn("尚未通過樣本外驗證", html)
        self.assertIn("資金保留現金，不產生配置試算", html)

    def test_existing_loss_is_closed_on_day_ten_and_preserved(self):
        pf = tracking._blank_portfolio()
        pf["strategy"] = strategy.contract()
        pf["positions"] = [{"code": "A", "name": "A", "shares": 10, "entry": 100.,
                            "entry_date": "2026-01-05", "held": 0, "last_processed": None,
                            "target1": 103.5, "stop": None}]
        state, summary = paper_target.update(pf, [], "2026-01-16", None, {"A": bars([90.] * 10)})
        self.assertEqual(state["positions"], [])
        self.assertEqual(state["trades"][0]["net_pct"], -10.5)
        self.assertEqual(summary["loss_trades"], 1)
        self.assertEqual(state["trades"][0]["held"], 10)

    @patch.object(C, "STOP_LOSS_NET_PCT", 2.)
    def test_stop_trigger_never_clamps_gap_loss_to_threshold(self):
        result = strategy.outcome(bars([90.] * 10), 0)
        self.assertEqual(result["reason"], "風險停損")
        self.assertAlmostEqual(result["net"], -10.5)
        self.assertEqual(result["days"], 1)

    def test_day_two_exact_target_exits_and_day_one_does_not(self):
        result = strategy.outcome(bars([103.5] * 10), 0)
        self.assertEqual(result["days"], 2)
        result = strategy.outcome(bars([110.] + [100.] * 9), 0)
        self.assertFalse(result["success"])
        self.assertEqual(result["days"], 10)


if __name__ == "__main__":
    unittest.main()
