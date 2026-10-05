import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_target_strategy import bars, bucket
from test_expected_return import row
import pandas as pd
import config as C
import execution
import paper_target
import overseas_research as overseas
import ranking
import research
import strategy
import render


def entry_history(open_price=100.):
    h = bars([100.] * 36)
    h.iloc[20, h.columns.get_loc("open")] = open_price
    h.iloc[20, h.columns.get_loc("high")] = max(101., open_price)
    return h


class EntryGuardTests(unittest.TestCase):
    def test_gap_is_skipped_even_if_later_low_is_below_limit(self):
        h = entry_history(103.)
        h.iloc[20, h.columns.get_loc("low")] = 95.
        r = execution.simulate(h, 20)
        self.assertFalse(r["entered"])
        self.assertTrue(r["entry_skipped"])
        self.assertNotIn("net", r)
        self.assertTrue(execution.simulate(h, 20, guard=False)["closed"])

    def test_boundary_includes_slippage_and_uses_lower_ceiling(self):
        plan = execution.make_entry_plan(100., 90.)
        self.assertAlmostEqual(plan["max_fill_price"], 97.2)
        self.assertIsNone(execution.check_entry_price(plan["max_open_price"], plan))
        self.assertIsNotNone(execution.check_entry_price(plan["max_open_price"] + .001, plan))
        self.assertIsNotNone(execution.check_entry_price(102., execution.make_entry_plan(100.,100.)))

    def test_future_prices_never_raise_frozen_ceiling(self):
        h = entry_history(103.)
        first = execution.simulate(h,20)
        h.iloc[20:,h.columns.get_loc("close")] = 500.
        second = execution.simulate(h,20)
        self.assertEqual(first["entry_plan"],second["entry_plan"])
        self.assertFalse(second["entered"])

    def test_paper_and_simulator_both_cancel_gap_once(self):
        h = entry_history(103.)
        signal_date = str(h.index[19])[:10]
        candidate = dict(row("A",4.), research_eligible=True)
        book,_ = paper_target.update({},[candidate],signal_date,None,{"A":h},signal_date+"T16:00:00+08:00",research=True)
        entry_date = str(h.index[20])[:10]
        book,_ = paper_target.update(book,[],entry_date,None,{"A":h},research=True)
        self.assertFalse(book["positions"])
        self.assertFalse(book["pending"])
        self.assertEqual(book["cash"],100000.)
        self.assertEqual(book["skipped"][0]["reason"],execution.simulate(h,20)["reason"])
        book,_ = paper_target.update(book,[],str(h.index[21])[:10],None,{"A":h},research=True)
        self.assertFalse(book["positions"])

    def test_missing_signal_reference_fails_closed(self):
        self.assertFalse(execution.make_entry_plan(None)["available"])
        self.assertIsNotNone(execution.check_entry_price(100.,{}))
        self.assertFalse(execution.make_entry_plan(100.,float("nan"))["available"])

    def test_guard_change_invalidates_calibration_and_pending_books(self):
        old = {"strategy":strategy.contract()}
        with patch.object(C,"ENTRY_MAX_PREMIUM_PCT",1.):
            self.assertFalse(strategy.compatible(old))

    def test_entry_study_reports_excluded_trades_instead_of_fake_wins(self):
        h=entry_history(103.)
        signals=[{"code":"A","date":str(h.index[19])[:10]}]
        result=research.entry_study(signals,{"A":h})
        self.assertEqual(result["variants"][0]["samples"],1)
        self.assertEqual(result["variants"][1]["samples"],0)
        self.assertEqual(result["variants"][1]["skipped"],1)
        self.assertIsNone(result["variants"][1]["expectancy"])


def us_history(end="2026-03-10"):
    index = pd.bdate_range(end=end,periods=20)
    return {k:pd.DataFrame({"close":[20.]*20,"ret1":[1.]*20,"ret5":[2.]*20},index=index)
            for k in overseas.SYMBOLS}


class OverseasContextTests(unittest.TestCase):
    def test_same_day_us_close_is_future_at_taiwan_signal_time(self):
        h=us_history()
        c=overseas.context_for_day(h,"2026-03-09")
        self.assertTrue(c["available"])
        self.assertEqual(set(c["dates"].values()),{"2026-03-06"})
        for df in h.values(): df.loc[pd.Timestamp("2026-03-09"),"ret5"]=-99.
        self.assertEqual(c,overseas.context_for_day(h,"2026-03-09"))

    def test_summer_and_winter_closes_both_align_before_tw_signal(self):
        for day in ("2026-01-06","2026-07-07"):
            c=overseas.context_for_day(us_history(day),day)
            self.assertTrue(c["available"])
            self.assertTrue(all(date < day for date in c["dates"].values()))

    def test_missing_stale_and_nan_are_unknown_not_neutral(self):
        for mode in ("missing","stale","nan"):
            h=us_history()
            if mode=="missing": del h["sox"]
            if mode=="stale": h["sox"]=h["sox"].iloc[:-8]
            if mode=="nan": h["sox"].loc[:,"ret5"]=float("nan")
            c=overseas.context_for_day(h,"2026-03-10")
            r={"asset_type":"tech"};overseas.attach_context([r],c)
            self.assertFalse(c["available"])
            self.assertEqual(r["overseas_state"],"unknown")

    def test_industry_context_and_vix_are_explicit(self):
        h=us_history();h["sox"]["ret5"]=-2.;h["nasdaq"]["ret5"]=-3.
        c=overseas.context_for_day(h,"2026-03-10")
        rows=[{"asset_type":"tech"},{"asset_type":"other"}]
        overseas.attach_context(rows,c)
        self.assertEqual([r["overseas_state"] for r in rows],["headwind","tailwind"])
        h["vix"]["close"]=26.
        overseas.attach_context(rows,overseas.context_for_day(h,"2026-03-10"))
        self.assertEqual([r["overseas_state"] for r in rows],["headwind","headwind"])

    def test_shadow_rank_cannot_mutate_main_score_or_trade_permission(self):
        rows=[dict(row("A",4.),score=55.,kind="x",asset_type="tech")]
        overseas.attach_context(rows,overseas.context_for_day(us_history(),"2026-03-10"))
        base=copy.deepcopy(rows)
        tables={"score_buckets":[bucket(ev=1.)],"overseas_buckets":[dict(bucket(ev=5.),overseas_group="tech",overseas_state="tailwind")]}
        alt=overseas.shadow_rank(rows,tables)
        self.assertEqual(rows,base)
        self.assertEqual(alt[0]["hist_source"],"overseas")
        self.assertFalse(alt[0]["trade_eligible"])
        tables["overseas_buckets"][0]["samples"]=10
        self.assertEqual(overseas.shadow_rank(rows,tables)[0]["hist_source"],"bucket")

    def test_live_freeze_includes_price_limit_and_asof_dates(self):
        row={"code":"A","entry_plan":execution.make_entry_plan(100.,100.),"overseas_context":{"dates":{"sox":"2026-03-06"}},"overseas_rank":1}
        with tempfile.TemporaryDirectory() as folder,patch.object(research,"ROOT",Path(folder)):
            first=research.freeze([row],"2026-03-09","2026-03-09T16:00:00+08:00")
            row["entry_plan"]=execution.make_entry_plan(200.,200.)
            again=research.freeze([row],"2026-03-09","2026-03-09T17:00:00+08:00")
            self.assertEqual(first,again)
            self.assertAlmostEqual(first["rows"][0]["entry_plan"]["max_fill_price"],102.)

    def test_missing_context_produces_no_fake_overseas_evidence(self):
        study={};overseas.compare_day(study,[row("A",4.)],{}, {},1)
        result=overseas.summarize(study)
        self.assertEqual(result["coverage_pct"],0.)
        self.assertFalse(result["available"])
        self.assertIsNone(result["paired_ev_lift"])

    def test_main_research_list_is_collapsed_and_all_cards_preserved(self):
        payload=json.loads((Path(__file__).resolve().parents[1]/"data/latest.json").read_text())
        payload["rows"]=payload["rows"][:2]
        html=render.render_html(payload)
        from html.parser import HTMLParser
        class Tags(HTMLParser):
            def __init__(self): super().__init__();self.cards=[];self.fold=None
            def handle_starttag(self,tag,attrs):
                a=dict(attrs)
                if tag=="details" and a.get("class")=="card": self.cards.append(a["data-code"])
                if a.get("id")=="research-rank": self.fold=a
        tags=Tags();tags.feed(html)
        self.assertEqual(len(tags.cards),2)
        self.assertNotIn("open",tags.fold)
        self.assertIn('id="candidate-empty"',html)


if __name__=="__main__": unittest.main()
