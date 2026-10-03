from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DIMENSIONS = ("grammar", "relevance", "consistency", "completeness", "logic")


def summarize(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no rows in {path}")
    scores: dict[str, list[float]] = {name: [] for name in DIMENSIONS}
    for row in rows:
        for name in DIMENSIONS:
            try:
                value = float(row[name])
            except (TypeError, ValueError) as error:
                raise ValueError(f"{row.get('id', '<unknown>')}: {name} is not scored") from error
            if value not in (0.0, 1.0, 2.0):
                raise ValueError(f"{row.get('id', '<unknown>')}: {name} must be 0, 1 or 2")
            scores[name].append(value)
    means = {name: sum(values) / len(values) for name, values in scores.items()}
    overall = sum(sum(values) for values in scores.values()) / (len(rows) * len(DIMENSIONS))
    return {
        "file": str(Path(path).resolve()),
        "rows": len(rows),
        "dimension_means": means,
        "overall_mean": overall,
        "grammar_full_rate": scores["grammar"].count(2.0) / len(rows),
        "relevance_full_rate": scores["relevance"].count(2.0) / len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize manual 0/1/2 continuation scores")
    parser.add_argument("--dev-csv", required=True)
    parser.add_argument("--final-csv")
    parser.add_argument("--output", default="human_score_summary.json")
    args = parser.parse_args()
    result = {"dev": summarize(args.dev_csv)}
    if args.final_csv:
        result["final"] = summarize(args.final_csv)
        result["dev_final_mean_gap"] = abs(
            result["dev"]["overall_mean"] - result["final"]["overall_mean"]
        )
    result["acceptance"] = {
        "grammar_full_rate_at_least_0.8": result["dev"]["grammar_full_rate"] >= 0.8,
        "relevance_full_rate_at_least_0.7": result["dev"]["relevance_full_rate"] >= 0.7,
        "overall_mean_at_least_1.4": result["dev"]["overall_mean"] >= 1.4,
        "dev_final_gap_at_most_0.2": result.get("dev_final_mean_gap", 0.0) <= 0.2,
    }
    with Path(args.output).open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(args.output)


if __name__ == "__main__":
    main()
