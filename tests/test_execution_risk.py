import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_target_strategy import bars, mature
from test_expected_return import row, valid_research
import config as C
import execution
import strategy
import risk_budget
import paper_target
import research
import ranking
import backtest


class ExecutionRiskTests(unittest.TestCase):
    def test_minimum_fee_and_sell_tax_are_real_cash_items(self):
        self.assertAlmostEqual(execution.buy_cash(100,1),120.1)
        self.assertAlmostEqual(execution.sell_cash(100,1,"2330"),79.6003)
        self.assertGreater(execution.sell_cash(100,1,"0050"),execution.sell_cash(100,1,"2330"))

    def test_stop_uses_next_open_and_preserves_overnight_gap(self):
        h=mature([96.],opens=[100.,80.]+[96.]*13)
        r=execution.simulate(h,0)
        self.assertEqual(r["days"],2)
        self.assertEqual(r["exit_price"],80.)
        self.assertLess(r["net"],-20.)

    def test_exit_day_later_low_cannot_change_prior_open_fill(self):
        h=mature([105.]);a=execution.simulate(h,0)
        h.iloc[1,h.columns.get_loc("low")]=1.
        b=execution.simulate(h,0)
        self.assertEqual(a,b)

    def test_locked_exit_delays_instead_of_fabricating_fill(self):
        h=mature([96.]);h.iloc[1]=[90.,90.,90.,90.]
        r=execution.simulate(h,0)
        self.assertEqual(r["days"],3)
        self.assertEqual(r["delayed_days"],1)

    def test_unfilled_entry_is_reported_and_not_a_win(self):
        h=mature([110.]);h.iloc[0]=[110.,110.,110.,110.]
        r=execution.simulate(h,0)
        # With no prior bar the model cannot infer a directional lock; adding
        # yesterday's valid close makes the adverse one-price day observable.
        h=bars([100.]+[110.]*15);h.iloc[1]=[110.,110.,110.,110.]
        r=execution.simulate(h,1)
        self.assertFalse(r["entered"])
        self.assertFalse(r["closed"])

    def test_tail_unfilled_exit_stays_unresolved(self):
        h=mature([96.]);h.iloc[1:]=[90.,90.,90.,90.]
        r=execution.simulate(h,0)
        self.assertFalse(r["closed"])
        self.assertTrue(r["entered"])
        self.assertLess(r["mark_net"],-10)

    def test_risk_size_includes_minimum_fees_and_leaves_cash(self):
        s=risk_budget.position_size(100000,100000,100)
        self.assertEqual(s["shares"],124)
        self.assertLessEqual(s["planned_risk"],500)
        self.assertGreater(execution.buy_cash(100,125)*.04,500)

    def test_total_risk_and_sector_caps_are_not_relaxed_for_one_stock(self):
        self.assertEqual(risk_budget.position_size(100000,100000,100,committed_risk=1500)["shares"],0)
        self.assertEqual(risk_budget.position_size(100000,100000,100,sector_value=40000)["shares"],0)
        self.assertLessEqual(risk_budget.position_size(100000,500,100)["cash_required"],500)

    def test_cost_changes_invalidate_old_statistics(self):
        old={"strategy":strategy.contract()}
        with patch.object(C,"MIN_BROKER_FEE_TWD",1.):
            self.assertFalse(strategy.compatible(old))

    def test_flagged_data_blocks_allocation_without_removing_losses(self):
        bt=valid_research();bt["data_quality"]={"issue_count":1}
        self.assertFalse(ranking.research_validated(bt))
        h=mature([60.]);audit=execution.audit({"A":h})
        self.assertGreater(audit["issue_count"],0)
        self.assertLess(execution.simulate(h,0)["net"],-40.)

    def test_first_publication_is_immutable(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(research,"ROOT",Path(folder)):
            a=research.freeze([{"code":"A","final_rank":1}],"2026-01-05","2026-01-05T16:00:00+08:00")
            b=research.freeze([{"code":"B","final_rank":1}],"2026-01-05","2026-01-05T18:00:00+08:00")
            self.assertEqual(a,b)

    def test_shadow_tracking_does_not_unblock_allocation(self):
        candidate=dict(row("A",4.),research_eligible=True)
        state,_=paper_target.update({},[candidate],"2026-01-05",None,{},"2026-01-05T16:00:00+08:00",research=True)
        self.assertEqual(len(state["pending"]),1)
        self.assertTrue(state["capital_policy"]["blocked"])
        real,_=paper_target.update({},[candidate],"2026-01-05",None,{})
        self.assertFalse(real["pending"])

    def test_cash_ledger_never_reuses_funds_or_buys_duplicate_stock(self):
        rows=[dict(row(c,4.),research_eligible=True,group="電子") for c in ["A","A","B","C","D"]]
        state,_=paper_target.update({},rows,"2026-01-02",None,{},"2026-01-02T16:00:00+08:00",research=True)
        self.assertEqual(len(state["pending"]),3)
        h=mature([100.])
        state,_=paper_target.update(state,[],"2026-01-05",None,{c:h for c in ["A","B","C"]},research=True)
        self.assertGreaterEqual(state["cash"],0)
        self.assertEqual(len(state["positions"]),3)
        self.assertLessEqual(sum(p["planned_risk"] for p in state["positions"]),1500)
        self.assertAlmostEqual(state["cash"]+sum(p["entry_cash"] for p in state["positions"]),100000.)

    def test_unresolved_outcome_is_never_counted_as_zero_return(self):
        stats=backtest._pack([{"date":"2026-01-01","exit":{"closed":False,"entered":True,"mark_net":-50}}])
        self.assertEqual(stats["samples"],0)
        self.assertIsNone(stats["expectancy"])


if __name__ == "__main__":
    unittest.main()
