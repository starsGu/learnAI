from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from pretrain.prompts import get_prompts
from pretrain.score import summarize


class EvaluationTests(unittest.TestCase):
    def test_prompt_sets_have_required_size_and_do_not_overlap(self) -> None:
        dev = get_prompts("dev")
        final = get_prompts("final")
        self.assertEqual(len(dev), 120)
        self.assertEqual(len(final), 60)
        self.assertFalse({row["prompt"] for row in dev} & {row["prompt"] for row in final})

    def test_human_score_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["id", "grammar", "relevance", "consistency", "completeness", "logic"])
                writer.writerow(["a", 2, 2, 1, 2, 1])
                writer.writerow(["b", 2, 1, 2, 1, 2])
            result = summarize(path)
            self.assertEqual(result["rows"], 2)
            self.assertEqual(result["overall_mean"], 1.6)
            self.assertEqual(result["grammar_full_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
