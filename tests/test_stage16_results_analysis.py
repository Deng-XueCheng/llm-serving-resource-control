from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from experiments.analyze_stage16_results import (
    bootstrap_ci,
    build_comparisons,
    exact_sign_test,
    exact_wilcoxon,
    holm_bonferroni,
    load_rows,
    markdown_report,
    write_analysis_figures,
)


def policy_metrics(*, gap_p99: float) -> dict[str, object]:
    return {
        "actual_recompute_tokens": 100,
        "preemption_count": 10,
        "ttft_p99_ms": 20.0,
        "itl_p99_ms": 30.0,
        "fairness": {
            "post_token_progress_gap_steps": {"p99": gap_p99},
            "oldest_waiting_age": {"max": 40.0},
        },
    }


class Stage16ResultsAnalysisTests(unittest.TestCase):
    def test_load_rows_exports_canonical_post_token_gap_column(self) -> None:
        aggregate = {
            "accepted": True,
            "matrix": {"matched_triples": 9},
            "triples": [
                {
                    "capacity": 4,
                    "seed": 1,
                    "pressure": policy_metrics(gap_p99=11.0),
                    "recompute": policy_metrics(gap_p99=22.0),
                    "bounded": policy_metrics(gap_p99=33.0),
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aggregate.json"
            path.write_text(json.dumps(aggregate), encoding="utf-8")

            row = load_rows(path)[0]

        self.assertEqual(
            row["bounded_post_token_progress_gap_p99_steps"],
            33.0,
        )

    def test_exact_sign_and_wilcoxon_are_deterministic_for_all_positive_pairs(
        self,
    ) -> None:
        differences = [1.0, 2.0, 3.0]

        self.assertEqual(exact_sign_test(differences), 0.25)
        statistic, p_value = exact_wilcoxon(differences)
        self.assertEqual(statistic, 6.0)
        self.assertEqual(p_value, 0.25)

    def test_exact_tests_handle_ties_zeros_mixed_signs_and_all_zero(self) -> None:
        statistic, p_value = exact_wilcoxon([1.0, -1.0, 2.0])
        self.assertEqual(statistic, 4.5)
        self.assertEqual(p_value, 0.75)
        statistic, p_value = exact_wilcoxon([0.0, 1.0, -2.0])
        self.assertEqual((statistic, p_value), (1.0, 1.0))
        self.assertEqual(exact_wilcoxon([0.0, 0.0]), (0.0, 1.0))
        self.assertEqual(exact_sign_test([0.0, 0.0]), 1.0)

    def test_bootstrap_ci_is_deterministic_for_fixed_seed(self) -> None:
        first = bootstrap_ci([1.0, 2.0, 8.0], seed=7, draws=200)
        second = bootstrap_ci([1.0, 2.0, 8.0], seed=7, draws=200)
        self.assertEqual(first, second)

    def test_ratio_of_totals_differs_from_mean_paired_reduction(self) -> None:
        rows = []
        for seed, pressure, bounded in ((1, 100.0, 50.0), (2, 10.0, 0.0)):
            row = {"capacity": 4, "seed": seed}
            for policy in ("pressure", "recompute", "bounded"):
                for metric in (
                    "actual_recompute_tokens",
                    "preemption_count",
                    "ttft_p99_ms",
                    "itl_p99_ms",
                    "post_token_progress_gap_p99_steps",
                    "max_waiting_age_steps",
                ):
                    row[f"{policy}_{metric}"] = 10.0
            row["pressure_actual_recompute_tokens"] = pressure
            row["bounded_actual_recompute_tokens"] = bounded
            rows.append(row)

        comparison = next(
            item
            for item in build_comparisons(rows)
            if item["metric"] == "actual_recompute_tokens"
        )
        ratio_of_totals = 1.0 - (50.0 / 110.0)

        self.assertAlmostEqual(comparison["mean_reduction"], 0.75)
        self.assertAlmostEqual(ratio_of_totals, 0.5454545454545454)
        self.assertNotAlmostEqual(comparison["mean_reduction"], ratio_of_totals)

    def test_markdown_report_tracks_aggregate_values_and_decision(self) -> None:
        gates = {}
        for metric in (
            "post_token_progress_gap_p99",
            "max_waiting_age",
            "itl_p99_ms",
            "actual_recompute_tokens",
            "preemption_count",
            "ttft_p99_ms",
        ):
            gates[metric] = {
                "baseline_total": 100.0,
                "candidate_total": 25.0,
                "reduction": 0.75,
                "improved_pairs": 1,
                "passed": True,
            }
        aggregate = {
            "matrix": {"runs": 3, "matched_triples": 1},
            "completion_gates_validated": True,
            "raw_metrics_recomputed": True,
            "aggregate_gates": gates,
            "accepted": True,
            "decision": "custom_success",
        }
        rows = [{"bounded_max_waiting_age_steps": 17.0}]

        report = markdown_report(
            aggregate,
            Path("custom.aggregate.json"),
            rows,
            [],
        )

        self.assertIn("custom.aggregate.json", report)
        self.assertIn("降低 75.00%", report)
        self.assertIn("1/1 matched pairs", report)
        self.assertIn("decision=custom_success", report)
        self.assertNotIn("stage16_diagnostic_r2", report)

    def test_holm_bonferroni_preserves_order_and_monotonicity(self) -> None:
        self.assertEqual(
            holm_bonferroni([0.01, 0.04, 0.20]),
            [0.03, 0.08, 0.20],
        )

    def test_write_analysis_figures_outputs_svg_and_png(self) -> None:
        rows = [
            {
                "capacity": 4,
                "seed": 1,
                "pressure_actual_recompute_tokens": 100.0,
                "pressure_itl_p99_ms": 10.0,
                "recompute_actual_recompute_tokens": 20.0,
                "recompute_itl_p99_ms": 100.0,
                "bounded_actual_recompute_tokens": 50.0,
                "bounded_itl_p99_ms": 30.0,
            }
        ]
        comparisons = [
            {
                "metric": "itl_p99_ms",
                "family": "latency",
                "mean_reduction": 0.7,
                "improved_pairs": 1,
                "n": 1,
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)

            write_analysis_figures(output, rows, comparisons)

            for stem in ("stage16_pareto", "stage16_reductions"):
                self.assertTrue((output / f"{stem}.svg").is_file())
                self.assertTrue((output / f"{stem}.png").is_file())
                with Image.open(output / f"{stem}.png") as image:
                    self.assertGreaterEqual(image.info["dpi"][0], 449.0)
                    self.assertGreater(image.width, 0)
                    self.assertGreater(image.height, 0)
            self.assertIn(
                "1/1",
                (output / "stage16_reductions.svg").read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
