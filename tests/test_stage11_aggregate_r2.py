from __future__ import annotations

import unittest
from unittest.mock import patch

from experiments import aggregate_stage11 as r1
from experiments import aggregate_stage11_r2 as r2
from experiments.aggregate_stage11_r2 import (
    attach_metric_coverage,
    nullable_mean_std,
)


class Stage11AggregateR2Tests(unittest.TestCase):
    def test_nullable_metric_reports_missing_repeats(self) -> None:
        self.assertEqual(
            nullable_mean_std([10.0, None, 14.0]),
            {
                "mean": 12.0,
                "sample_std": 2.8284271247461903,
                "valid_runs": 2,
                "missing_runs": 1,
            },
        )

    def test_all_missing_metric_stays_undefined(self) -> None:
        self.assertEqual(
            nullable_mean_std([None, None, None]),
            {
                "mean": None,
                "sample_std": None,
                "valid_runs": 0,
                "missing_runs": 3,
            },
        )

    def test_non_finite_metric_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            nullable_mean_std([1.0, float("nan")])

    def test_validator_helper_is_restored_when_base_aggregate_fails(self) -> None:
        original = r1.nullable_mean_std
        def fail_after_observing_patch() -> None:
            self.assertIs(r1.nullable_mean_std, r2.nullable_mean_std)
            raise RuntimeError("base failed")

        with patch.object(r1, "aggregate", side_effect=fail_after_observing_patch):
            with self.assertRaisesRegex(RuntimeError, "base failed"):
                r2.aggregate()
        self.assertIs(r1.nullable_mean_std, original)

    def test_coverage_keeps_policy_specific_seed_masks(self) -> None:
        value = {
            "runs": [
                {
                    "reuse_ratio": 0,
                    "capacity": 6,
                    "policy": "kv_aware_fifo",
                    "seed": 1,
                    "metrics": {"interactive_ttft_p99_ms": None},
                },
                {
                    "reuse_ratio": 0,
                    "capacity": 6,
                    "policy": "kv_aware_fifo",
                    "seed": 2,
                    "metrics": {"interactive_ttft_p99_ms": 10.0},
                },
                {
                    "reuse_ratio": 0,
                    "capacity": 6,
                    "policy": "prefix_aware_fifo",
                    "seed": 1,
                    "metrics": {"interactive_ttft_p99_ms": 5.0},
                },
            ],
            "by_cell": {
                "reuse0_kv6_kv_aware_fifo": {
                    "interactive_ttft_p99_ms": {
                        "valid_runs": 1,
                        "missing_runs": 1,
                    }
                },
                "reuse0_kv6_prefix_aware_fifo": {
                    "interactive_ttft_p99_ms": {
                        "valid_runs": 1,
                        "missing_runs": 0,
                    }
                },
            },
        }

        attach_metric_coverage(value)

        self.assertEqual(
            value["by_cell"]["reuse0_kv6_kv_aware_fifo"]
            ["interactive_ttft_p99_ms"]["valid_seeds"],
            [2],
        )
        self.assertEqual(
            value["by_cell"]["reuse0_kv6_kv_aware_fifo"]
            ["interactive_ttft_p99_ms"]["missing_seeds"],
            [1],
        )


if __name__ == "__main__":
    unittest.main()
