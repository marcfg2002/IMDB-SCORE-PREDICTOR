import unittest

from expert_system import evaluate_movie


class ExpertSystemTests(unittest.TestCase):
    def test_high_rating_and_awards_activate_converging_rules(self) -> None:
        result = evaluate_movie(7.0, award_wins=3, award_nominations=0)

        self.assertEqual(result["recommendation"], "Recomendación favorable")
        self.assertIn("señales convergentes", result["fired_rules"][-1])

    def test_middle_rating_without_awards_is_intermediate(self) -> None:
        result = evaluate_movie(5.5, award_wins=0, award_nominations=0)

        self.assertEqual(result["recommendation"], "Opinión intermedia")
        self.assertIn("R5:", result["fired_rules"][1])

    def test_low_rating_with_awards_reports_conflicting_signals(self) -> None:
        result = evaluate_movie(5.49, award_wins=0, award_nominations=5)

        self.assertEqual(result["recommendation"], "Recomendación prudente")
        self.assertIn("discrepan", result["conclusion"])


if __name__ == "__main__":
    unittest.main()
