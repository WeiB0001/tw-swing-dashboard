import unittest
from unittest.mock import patch

from test_expected_return import row, valid_research
import config as C
import strategy
import risk_stats
import ranking
import backtest
import build
import paper_target


class RiskRewardTests(unittest.TestCase):
    def test_ui_summary_retains_strategy_validation_evidence(self):
        self.assertTrue(ranking.research_validated(build._backtest_summary(valid_research())))

    def test_small_profit_is_neither_success_reward_nor_loss(self):
        d = risk_stats.risk_reward([3., 1., 0., -2.])
        self.assertAlmostEqual(d["target_gain_component"], .75)
        self.assertAlmostEqual(d["loss_component"], .5)
        self.assertAlmostEqual(d["risk_reward_score"], 0.)
        self.assertEqual(d["avg_target_gain"], 3.)
        self.assertEqual(d["avg_loss_magnitude"], 2.)
        self.assertEqual(strategy.utility(2.999), 0.)

    def test_combined_risk_can_outweigh_higher_raw_return(self):
        high_return = [14.] * 50 + [-8.] * 50  # EV 3%, score 1.
        balanced = [3.5] * 80 + [-4.] * 20    # EV 2%, score 1.6.
        rows = []
        for code, nets in [("high_ev", high_return), ("balanced", balanced)]:
            d = backtest.describe(nets, [0.] * len(nets))
            r = row(code, d["expectancy"])
            r["hist_risk_reward"] = d["risk_reward_score"]
            rows.append(r)
        result = ranking.sort(rows)
        self.assertEqual(result[0]["code"], "balanced")
        self.assertLess(result[0]["hist_expectancy"], result[1]["hist_expectancy"])

    def test_high_success_with_rare_large_loss_gets_negative_score(self):
        d = risk_stats.risk_reward([3.1] * 90 + [-30.] * 10)
        self.assertAlmostEqual(d["risk_reward_score"], -1.71)

    def test_all_losses_preserve_full_magnitude(self):
        d = risk_stats.risk_reward([-10., -20.])
        self.assertEqual(d["risk_reward_score"], -22.5)
        self.assertIsNone(d["avg_target_gain"])
        self.assertEqual(risk_stats.risk_reward([])["risk_reward_score"], None)

    def test_loss_weight_change_invalidates_old_statistics(self):
        old = {"strategy": strategy.contract()}
        with patch.object(C, "LOSS_AVERSION", 2.):
            self.assertFalse(strategy.compatible(old))
            self.assertEqual(strategy.utility(-4.), -8.)

    def test_block_bootstrap_preserves_both_score_and_net_return(self):
        signals = []
        for day in range(1, 31):
            for net in [4., -2., 1.]:
                signals.append({"date": f"2026-01-{day:02}", "exit": {"net": net}})
        d = risk_stats.expected_return_lower(signals)
        self.assertAlmostEqual(d["ev_lower"], 1.)
        self.assertAlmostEqual(d["risk_reward_lower"], 1 / 3, places=4)

    def test_positive_ev_does_not_override_uncertain_composite_score(self):
        r = row("A", 5.)
        r["hist_risk_reward_lower"] = -.1
        self.assertFalse(ranking.sort([r])[0]["research_eligible"])
        bt = valid_research()
        bt["walk_forward"]["risk_reward_lower"] = -.1
        self.assertFalse(ranking.research_validated(bt))

    def test_validated_new_policy_can_create_paper_order(self):
        rows = ranking.sort([row("A", 4.)])
        policy = ranking.apply_policy(rows, valid_research())
        state, _ = paper_target.update({}, rows, "2026-01-05", None, {}, capital_policy=policy)
        self.assertEqual(len(state["pending"]), 1)
        self.assertTrue(state["pending"][0]["entry_approved"])

    def test_stale_policy_cannot_create_paper_order(self):
        rows = ranking.sort([row("A", 4.)])
        policy = ranking.apply_policy(rows, valid_research())
        policy["strategy"] = {"version": "expected-net-v2"}
        state, _ = paper_target.update({}, rows, "2026-01-05", None, {}, capital_policy=policy)
        self.assertFalse(state["pending"])

    def test_composite_rank_validation_is_distinct_from_raw_ev(self):
        groups = [{"samples": 100, "risk_reward_score": v, "expectancy": ev}
                  for v, ev in zip([3., 2., 1., 0.], [1., 3., 2., 4.])]
        self.assertTrue(risk_stats.rank_order_check(groups, "risk_reward_score")["monotonic"])
        self.assertFalse(risk_stats.rank_order_check(groups)["monotonic"])


if __name__ == "__main__":
    unittest.main()
