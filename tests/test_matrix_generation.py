from __future__ import annotations

import unittest
import copy
import json
import tempfile
from pathlib import Path

from experiments.generate_matrix import (
    build_benchmark_config,
    generate_trace,
    load_matrix_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class MatrixGenerationTests(unittest.TestCase):
    def load_source_matrix(self) -> dict:
        return json.loads(
            (
                REPO_ROOT
                / "experiments/configs/baseline_calibration_matrix.json"
            ).read_text(encoding="utf-8")
        )

    def write_matrix(self, value: dict) -> Path:
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            delete=False,
        )
        with handle:
            json.dump(value, handle)
        self.addCleanup(Path(handle.name).unlink, missing_ok=True)
        return Path(handle.name)

    def test_trace_is_deterministic_and_preserves_class_ratio(self) -> None:
        arguments = {
            "duration_seconds": 4.0,
            "request_count": 10,
            "interactive_per_cycle": 4,
            "long_per_cycle": 1,
            "interactive": {
                "prompt_length": 32,
                "max_output_tokens": 32,
            },
            "long_request": {
                "prompt_length": 768,
                "max_output_tokens": 64,
            },
            "seed": 1,
            "prompt_seed_base": 30000,
            "description": "test",
        }

        first = generate_trace(**arguments)
        second = generate_trace(**arguments)

        self.assertEqual(first, second)
        classes = [
            request["request_class"] for request in first["requests"]
        ]
        self.assertEqual(classes.count("interactive"), 8)
        self.assertEqual(classes.count("long"), 2)
        arrivals = [
            request["arrival_time"] for request in first["requests"]
        ]
        self.assertEqual(arrivals, sorted(arrivals))
        self.assertTrue(all(0 <= arrival < 4.0 for arrival in arrivals))

    def test_benchmark_config_changes_only_run_specific_fields(self) -> None:
        template = {
            "sampling": {"seed": 1},
            "workload": {"trace_path": "old.json"},
            "measurement": {
                "start_seconds": 1.0,
                "end_seconds": 2.0,
            },
            "output": {"run_id": "old"},
            "engine": {"max_num_seqs": 8},
        }

        generated = build_benchmark_config(
            template,
            trace_path="experiments/data/new.json",
            run_id="new-run",
            inference_seed=42,
            duration_seconds=4.0,
        )

        self.assertEqual(generated["sampling"]["seed"], 42)
        self.assertEqual(
            generated["workload"]["trace_path"],
            "experiments/data/new.json",
        )
        self.assertEqual(generated["measurement"]["start_seconds"], 0.0)
        self.assertEqual(generated["measurement"]["end_seconds"], 4.0)
        self.assertEqual(generated["output"]["run_id"], "new-run")
        self.assertEqual(generated["engine"], template["engine"])
        self.assertEqual(template["output"]["run_id"], "old")

    def test_matrix_rejects_unsafe_run_id_prefix(self) -> None:
        matrix = self.load_source_matrix()
        matrix["run_id_prefix"] = "../escape"

        with self.assertRaisesRegex(ValueError, "unsafe"):
            load_matrix_config(self.write_matrix(matrix))

    def test_matrix_rejects_mislabeled_offered_load(self) -> None:
        matrix = self.load_source_matrix()
        matrix["loads"]["critical"]["offered_output_tps"] = 95.0

        with self.assertRaisesRegex(ValueError, "does not match"):
            load_matrix_config(self.write_matrix(matrix))

    def test_matrix_requires_exactly_three_unique_seeds(self) -> None:
        for seeds in ([1, 2], [1, 1, 2]):
            with self.subTest(seeds=seeds):
                matrix = self.load_source_matrix()
                matrix["seeds"] = seeds

                with self.assertRaisesRegex(ValueError, "exactly 3 unique"):
                    load_matrix_config(self.write_matrix(matrix))


if __name__ == "__main__":
    unittest.main()
