import copy
import json
from html.parser import HTMLParser
from pathlib import Path
import unittest

import ranking
import render
from test_expected_return import row


class ReferenceRankingTests(unittest.TestCase):
    def strong_row(self):
        return dict(row("STRONG", 2.), score=60., day_open=99., day_high=101.,
                    day_low=98., close=100., volume=1000., quote_date="2026-10-05",
                    chg_pct=2., bias20=3., rsi=55., research_eligible=True,
                    hist_loss_rate_upper=35.,
                    entry_plan={"available": True, "max_open_price": 102.},
                    overseas_context={"available": True, "dates": {}}, overseas_state="tailwind")

    def test_strength_is_not_a_low_loss_claim_without_validation(self):
        marker = render.strength_marker(self.strong_row(), "2026-10-05")
        self.assertTrue(marker["technical"])
        self.assertFalse(marker["low_loss"])
        self.assertEqual(marker["label"], "技術強勢")
        self.assertIn("待驗證", marker["note"])
        self.assertTrue(render.strength_marker(self.strong_row(), "2026-10-05", True)["low_loss"])

    def test_weak_stale_invalid_and_missing_quotes_do_not_get_strong_badges(self):
        for changes in [dict(score=59.9), dict(momentum_tier=2), dict(quote_date="2026-10-02"),
                        dict(day_high=99.), dict(volume=0.), dict(day_low=None), dict(score=float("nan"))]:
            with self.subTest(changes=changes):
                self.assertFalse(render.strength_marker(dict(self.strong_row(), **changes), "2026-10-05", True)["technical"])
        self.assertFalse(render.strength_marker(self.strong_row(), None, True)["technical"])

    def test_chasing_and_overseas_risks_stay_visible_on_strong_names(self):
        for changes, message in [
            (dict(chg_pct=6.), "今日漲多，留意追價"),
            (dict(bias20=9.), "均線乖離過大"),
            (dict(rsi=90.), "RSI 偏熱"),
            (dict(entry_plan={"available": True, "max_open_price": 99.}), "現價高於進場上限"),
            (dict(entry_plan={}), "進場上限未知"),
            (dict(overseas_state="headwind"), "海外逆風"),
            (dict(overseas_state="mixed"), "海外走勢分歧"),
            (dict(overseas_context={"available": False}), "海外資料不足"),
        ]:
            with self.subTest(message=message):
                marker = render.strength_marker(dict(self.strong_row(), **changes), "2026-10-05", True)
                self.assertTrue(marker["technical"])
                self.assertIn(message, marker["cautions"])
                self.assertFalse(marker["low_loss"])

    def test_point_estimate_alone_cannot_earn_low_loss_badge(self):
        for changes in [dict(hist_loss_rate=25., hist_loss_rate_upper=45.),
                        dict(hist_loss_rate_upper=None), dict(hist_ev_lower=-.1),
                        dict(hist_samples=99), dict(hist_signal_dates=19), dict(research_eligible=False)]:
            with self.subTest(changes=changes):
                self.assertFalse(render.strength_marker(dict(self.strong_row(), **changes), "2026-10-05", True)["low_loss"])

    def test_positive_scores_remain_despite_failed_gates(self):
        high = dict(row("HIGH", 5.), score=80., momentum_tier=2,
                    hist_ev_lower=-.1, main_risk="追高風險", quote_date="2026-10-02")
        low = dict(row("LOW", 2.), score=40., quote_date="2026-10-05")
        ranked = ranking.sort([low, high])
        ranking.apply_policy(ranked, None, "2026-10-05")
        original = copy.deepcopy(ranked)
        view = render.reference_rows(ranked, "2026-10-05")
        self.assertEqual([r["code"] for r in view], ["HIGH", "LOW"])
        self.assertEqual(view[0]["technical_rank"], 1)
        self.assertEqual(view[0]["hist_risk_reward"], 5.)
        self.assertFalse(view[0]["trade_eligible"])
        self.assertIn("行情日期不一致", view[0]["reference_risk"])
        self.assertIn("追高風險", view[0]["reference_risk"])
        self.assertIn("統計保守估計未轉正", view[0]["reference_warnings"])
        self.assertEqual(ranked, original)

    def test_missing_and_negative_statistics_are_retained_without_relabeling(self):
        rows = [{"code": "NEG", "score": 10., "hist_risk_reward": -2.},
                {"code": "MISSING", "score": 70., "hist_risk_reward": None},
                {"code": "NO_TECH", "score": None}]
        view = render.reference_rows(rows)
        self.assertEqual([r["code"] for r in view], ["NEG", "MISSING", "NO_TECH"])
        self.assertEqual([r["technical_rank"] for r in view], [2, 1, 3])
        self.assertEqual(view[0]["hist_risk_reward"], -2.)
        self.assertIsNone(view[1]["hist_risk_reward"])
        self.assertIn("歷史統計不足", view[1]["reference_risk"])

    def test_cards_expose_both_scores_and_risks_without_opening_details(self):
        payload = json.loads((Path(__file__).resolve().parents[1]/"data/latest.json").read_text())
        payload["rows"] = payload["rows"][:1]
        payload["rows"][0].update(trade_eligible=False, score=73., hist_risk_reward=1.2,
                                   hist_ev_lower=-.1, main_risk="測試追高風險")
        html = render.render_html(payload)

        class CardSummary(HTMLParser):
            def __init__(self):
                super().__init__(); self.in_card=False; self.in_summary=False; self.text=[]
            def handle_starttag(self, tag, attrs):
                a = dict(attrs)
                if tag == "details" and a.get("class") == "card": self.in_card=True
                if tag == "summary" and self.in_card: self.in_summary=True
            def handle_endtag(self, tag):
                if tag == "summary" and self.in_summary: self.in_card=self.in_summary=False
            def handle_data(self, value):
                if self.in_summary: self.text.append(value)

        tags = CardSummary(); tags.feed(html)
        summary = "".join(tags.text)
        for value in ["73.0", "+1.20", "技術機會分", "風險報酬分", "測試追高風險", "已列入參考"]:
            self.assertIn(value, summary)
        self.assertNotIn("不允許配置", summary)
        self.assertIn('data-sort="technical"', html)

    def test_strong_marker_is_visible_without_opening_card(self):
        payload = json.loads((Path(__file__).resolve().parents[1]/"data/latest.json").read_text())
        payload["rows"] = payload["rows"][:1]
        payload["rows"][0].update(self.strong_row())
        payload["rows"][0].update(overseas_state="headwind", trade_eligible=False)
        payload["meta"]["data_date"] = "2026-10-05"
        html = render.render_html(payload)
        card = html.split('class="card"', 1)[1].split('</summary>', 1)[0]
        self.assertIn('data-strong="1"', card)
        self.assertIn("★ 技術強勢", card)
        self.assertIn("海外逆風", card)
        self.assertIn("低虧損獲利能力待驗證", card)
        self.assertIn("只看強勢（1）", html)


if __name__ == "__main__":
    unittest.main()
