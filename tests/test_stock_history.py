import copy
import json
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

import backtest
import build
import ranking
import render
import stock_history
import strategy
import config as C
from test_target_strategy import bucket


def signal(code, net, date="2026-01-05", **changes):
    result = {"code": code, "date": date,
              "exit": {"closed": True, "entered": True, "net": net,
                       "label_end": "2026-01-20", "mdd": min(net, -1.),
                       "success": strategy.target_met(net), "days": 10}}
    result["exit"].update(changes)
    return result


def artifact(signals):
    return {"strategy": strategy.contract(), "mode": "live",
            "stock_history": stock_history.build(signals)}


class StockHistoryTests(unittest.TestCase):
    def test_per_stock_denominators_and_conditional_averages(self):
        a = [signal("A", n) for n in [3., 5., -2., 1.]]
        b = [signal("B", n) for n in [8., -6., -2.]]
        report = stock_history.build(a + b)["by_code"]
        self.assertEqual(report["A"]["samples"], 4)
        self.assertEqual(report["A"]["success_rate"], 50.)
        self.assertEqual(report["A"]["avg_target_gain"], 4.)
        self.assertEqual(report["A"]["loss_rate"], 25.)
        self.assertEqual(report["A"]["avg_loss_magnitude"], 2.)
        self.assertEqual(report["B"]["samples"], 3)
        self.assertEqual(report["B"]["avg_target_gain"], 8.)
        self.assertEqual(report["B"]["avg_loss_magnitude"], 4.)
        # The report has the exact same completed-sample semantics as calibration.
        pooled = backtest._pack(a + b)
        self.assertEqual(sum(r["samples"] for r in report.values()), pooled["samples"])
        self.assertEqual(sum(r["successes"] for r in report.values()), pooled["successes"])
        self.assertEqual(sum(r["losses"] for r in report.values()), pooled["losses"])

    def test_unfilled_unresolved_invalid_and_empty_conditional_results(self):
        report = stock_history.build([
            signal("A", 1.), signal("A", 99., closed=False),
            signal("A", 99., entered=False), signal("A", float("nan")),
            signal("A", 99., label_end=None)])
        self.assertEqual(report["by_code"]["A"]["samples"], 1)
        self.assertEqual(report["by_code"]["A"]["success_rate"], 0.)
        self.assertIsNone(report["by_code"]["A"]["avg_target_gain"])
        self.assertIsNone(report["by_code"]["A"]["avg_loss_magnitude"])
        self.assertEqual(stock_history.build([])["by_code"], {})

    def test_missing_stock_never_borrows_peer_and_future_stats_are_rejected(self):
        bt = artifact([signal("A", 5.)])
        rows = [{"code": code, "quote_date": date, "stock_history": {"old": True}}
                for code, date in [("A", "2026-01-21"), ("B", "2026-01-21"), ("A", "2026-01-19")]]
        stock_history.attach(rows, bt)
        self.assertEqual(rows[0]["stock_history"]["code"], "A")
        self.assertIsNone(rows[1]["stock_history"])
        self.assertIsNone(rows[2]["stock_history"])
        rows[0]["stock_history"]["samples"] = 999
        self.assertEqual(bt["stock_history"]["by_code"]["A"]["samples"], 1)

    def test_old_demo_and_incomplete_reports_do_not_supply_statistics(self):
        valid = artifact([signal("A", 5.)])
        variants = [dict(valid, mode="demo"), dict(valid, strategy={}),
                    dict(valid, stock_history={}), None]
        for bt in variants:
            rows = [{"code": "A", "quote_date": "2026-01-21", "stock_history": {"old": True}}]
            stock_history.attach(rows, bt)
            self.assertIsNone(rows[0]["stock_history"])
        self.assertTrue(stock_history.available(valid))
        self.assertFalse(stock_history.available({}))

    def test_live_builder_attaches_both_sources_and_resets_stale_report(self):
        bt = artifact([signal("A", 5.)])
        bt["calibration"] = {"score_buckets": [bucket()]}
        rows = [{"code": "A", "score": 50., "kind": "x", "quote_date": "2026-01-21"}]
        with tempfile.TemporaryDirectory() as folder, patch.object(build, "ROOT", Path(folder)):
            path = Path(folder) / C.BACKTEST_JSON
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(bt))
            build.attach_backtest(rows)
            self.assertEqual(rows[0]["stock_history"]["samples"], 1)
            self.assertEqual(rows[0]["hist_samples"], 100)
            path.unlink()
            build.attach_backtest(rows)
            self.assertIsNone(rows[0]["stock_history"])
            self.assertEqual(rows[0]["hist_samples"], 0)

    def test_same_cohort_has_distinct_stock_results_without_changing_rank(self):
        rows = [{"code": code, "score": score, "kind": "x", "quote_date": "2026-01-21"}
                for code, score in [("A", 50.), ("B", 55.)]]
        ranking.attach(rows, {"score_buckets": [bucket()]})
        build.add_momentum(rows)
        build.add_final_score(rows)
        before = copy.deepcopy(ranking.sort(rows))
        stock_history.attach(rows, artifact([signal("A", 8.), signal("B", -5.)]))
        after = ranking.sort(rows)
        self.assertEqual([(r["code"], r["final_rank"], r["rank_score"]) for r in before],
                         [(r["code"], r["final_rank"], r["rank_score"]) for r in after])
        view = {r["code"]: r for r in render.reference_rows(after)}
        self.assertEqual(view["A"]["hist_group_id"], view["B"]["hist_group_id"])
        self.assertEqual(view["A"]["history_group_peers"], 2)
        self.assertEqual(view["A"]["hist_success_raw"], view["B"]["hist_success_raw"])
        self.assertNotEqual(view["A"]["stock_history"]["success_rate"],
                            view["B"]["stock_history"]["success_rate"])

    def test_card_shows_security_outcomes_and_never_falls_back_to_group(self):
        payload = json.loads((Path(__file__).resolve().parents[1] / "data/latest.json").read_text())
        payload["rows"] = [copy.deepcopy(payload["rows"][0])]
        row = payload["rows"][0]
        row["quote_date"] = "2026-01-21"
        row["hist_success_raw"] = 77.7
        stock_history.attach([row], artifact([signal(row["code"], 8.), signal(row["code"], -2.)]))
        html = render.render_html(payload)
        card = html.split('class="card"', 1)[1].split('</summary>', 1)[0]
        for value in ["個股回測 · 2 筆", "小樣本", "50.0%", "+8.00%", "−2.00%", "排名依據"]:
            self.assertIn(value, card)
        self.assertNotIn("77.7%", card)
        row["stock_history"] = None
        card = render.render_html(payload).split('class="card"', 1)[1].split('</summary>', 1)[0]
        self.assertIn("不以其他股票統計代填", card)
        self.assertNotIn("77.7%", card)


if __name__ == "__main__":
    unittest.main()
