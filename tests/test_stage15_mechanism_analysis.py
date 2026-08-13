from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.analyze_stage15_mechanism import analyze_pair, bucket_steps


class Stage15MechanismAnalysisTests(unittest.TestCase):
    def test_analyze_pair_requires_frozen_matrix_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "matrix.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "cells": [
                            {
                                "capacity": 8,
                                "seed": 2,
                                "policy": "pressure",
                                "run_id": "frozen-pressure",
                            },
                            {
                                "capacity": 8,
                                "seed": 2,
                                "policy": "recompute",
                                "run_id": "frozen-recompute",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            aggregate = {
                "revision": "stage15_diagnostic_r3",
                "matrix": {"runs": 18, "matched_pairs": 9},
                "execution_code_fingerprint": {"scheduler_sha256": "abc"},
                "pairs": [{"capacity": 8, "seed": 2}],
            }
            record = {
                "paths": {"steps": root / "steps.jsonl"},
                "elapsed_seconds": 1.0,
                "recompute_metrics": {},
                "steps": [],
            }

            with (
                patch(
                    "experiments.analyze_stage15_mechanism.aggregate_stage15",
                    return_value=aggregate,
                ) as aggregate_mock,
                patch(
                    "experiments.analyze_stage15_mechanism.load_validated_stage15_run",
                    return_value=record,
                ) as load_mock,
            ):
                analysis = analyze_pair(
                    capacity=8,
                    seed=2,
                    revision="stage15_diagnostic_r3",
                    results_directory=root,
                    manifest_path=manifest_path,
                    bucket_seconds=1.0,
                )

            aggregate_mock.assert_called_once_with(
                root.resolve(), manifest_path.resolve()
            )
            self.assertEqual(load_mock.call_count, 2)
            self.assertEqual(
                load_mock.call_args_list[0].args[0],
                root / "frozen-pressure.summary.json",
            )
            self.assertEqual(
                analysis["policies"]["baseline"]["summary_path"],
                str((root / "frozen-pressure.summary.json").resolve()),
            )
            self.assertEqual(
                analysis["policies"]["baseline"]["steps_path"],
                str(root / "steps.jsonl"),
            )

    def test_analyze_pair_rejects_non_positive_bucket(self) -> None:
        for value in (0, -1, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "finite and positive"
            ):
                analyze_pair(
                    capacity=8,
                    seed=2,
                    revision="stage15_diagnostic_r3",
                    results_directory=Path("."),
                    manifest_path=Path("matrix.json"),
                    bucket_seconds=value,
                )

    def test_bucket_steps_aggregates_raw_causal_signals(self) -> None:
        steps = [
            {
                "started_at": 0.2,
                "phase": "prefill",
                "events": [
                    {
                        "actual_recompute_tokens": 8,
                        "finished": False,
                    }
                ],
                "scheduler": {
                    "mode": "starvation_prefill",
                    "kv_used_blocks_before": 4,
                    "kv_total_blocks": 4,
                    "waiting_ids_before": [1, 2],
                    "oldest_waiting_age": 5,
                    "preemptions": [{"seq_id": 3}],
                },
            },
            {
                "started_at": 0.8,
                "phase": "decode",
                "events": [
                    {"actual_recompute_tokens": 0, "finished": True}
                ],
                "scheduler": {
                    "mode": "drain_decode",
                    "kv_used_blocks_before": 2,
                    "kv_total_blocks": 4,
                    "waiting_ids_before": [2],
                    "oldest_waiting_age": 6,
                    "preemptions": [],
                },
            },
        ]

        rows = bucket_steps(steps, bucket_seconds=1.0)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["preemptions"], 1)
        self.assertEqual(rows[0]["recompute_tokens"], 8)
        self.assertEqual(rows[0]["finished_requests"], 1)
        self.assertEqual(rows[0]["kv_utilization_mean"], 0.75)
        self.assertEqual(rows[0]["oldest_waiting_age_max"], 6)


if __name__ == "__main__":
    unittest.main()
