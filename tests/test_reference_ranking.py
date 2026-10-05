import copy
import json
from html.parser import HTMLParser
from pathlib import Path
import unittest

import ranking
import render
from test_expected_return import row


class ReferenceRankingTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
