from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from experiments.aggregate_stage15 import (
        _resolve_config_trace_path,
        _resolve_recorded_artifact_path,
        load_validated_stage15_run,
        raw_recompute_metrics,
        validate_stage15_manifest,
    )
    from experiments.benchmark.calibration_evidence import validate_step_schema
except ModuleNotFoundError:
    raw_recompute_metrics = None
    _resolve_config_trace_path = None
    _resolve_recorded_artifact_path = None
    load_validated_stage15_run = None
    validate_stage15_manifest = None
    validate_step_schema = None


class Stage15AggregationTests(unittest.TestCase):
    def valid_steps(self) -> list[dict]:
        return [
            {
                "step_index": 0,
                "phase": "prefill",
                "num_scheduled_tokens": 4,
                "events": [
                    {
                        "request_id": "r1",
                        "seq_id": 1,
                        "phase": "prefill",
                        "num_scheduled_tokens": 4,
                        "emitted_token": True,
                        "finished": False,
                        "prefill_kind": "initial_prefill",
                        "actual_recompute_tokens": 0,
                        "resumed": False,
                    }
                ],
                "scheduler": {
                    "selected_prefill_ids": [1],
                    "selected_decode_ids": [],
                    "preemptions": [],
                },
            },
            {
                "step_index": 1,
                "phase": "prefill",
                "num_scheduled_tokens": 2,
                "events": [
                    {
                        "request_id": "r2",
                        "seq_id": 2,
                        "phase": "prefill",
                        "num_scheduled_tokens": 2,
                        "emitted_token": True,
                        "finished": False,
                        "prefill_kind": "initial_prefill",
                        "actual_recompute_tokens": 0,
                        "resumed": False,
                    }
                ],
                "scheduler": {
                    "selected_prefill_ids": [2],
                    "selected_decode_ids": [],
                    "preemptions": [
                        {
                            "seq_id": 1,
                            "reason": "waiting_prefill_allocation",
                            "estimated_recompute_tokens": 3,
                        }
                    ],
                },
            },
            {
                "step_index": 2,
                "phase": "prefill",
                "num_scheduled_tokens": 3,
                "events": [
                    {
                        "request_id": "r1",
                        "seq_id": 1,
                        "phase": "prefill",
                        "num_scheduled_tokens": 3,
                        "emitted_token": True,
                        "finished": False,
                        "prefill_kind": "recompute_prefill",
                        "actual_recompute_tokens": 3,
                        "resumed": True,
                    }
                ],
                "scheduler": {
                    "selected_prefill_ids": [1],
                    "selected_decode_ids": [],
                    "preemptions": [],
                },
            },
        ]

    def write_steps(self, steps: list[dict]) -> Path:
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".jsonl",
            delete=False,
        )
        with handle:
            for step in steps:
                handle.write(json.dumps(step) + "\n")
        path = Path(handle.name)
        self.addCleanup(path.unlink, missing_ok=True)
        return path

    def write_json(self, value: dict) -> Path:
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            delete=False,
        )
        with handle:
            json.dump(value, handle)
        path = Path(handle.name)
        self.addCleanup(path.unlink, missing_ok=True)
        return path

    def test_raw_events_recompute_strict_metrics(self) -> None:
        self.assertIsNotNone(raw_recompute_metrics)

        metrics = raw_recompute_metrics(self.write_steps(self.valid_steps()))

        self.assertEqual(metrics["initial_prefill_tokens"], 6)
        self.assertEqual(metrics["actual_recompute_tokens"], 3)
        self.assertEqual(metrics["preemption_count"], 1)
        self.assertEqual(metrics["resume_count"], 1)
        self.assertEqual(metrics["recompute_amplification"], 1.5)
        self.assertEqual(metrics["extra_prefill_ratio"], 0.5)
        self.assertEqual(metrics["recompute_tokens_per_preemption"], 3.0)

    def test_rejects_recompute_token_mismatch(self) -> None:
        self.assertIsNotNone(raw_recompute_metrics)
        steps = self.valid_steps()
        steps[2]["events"][0]["actual_recompute_tokens"] = 2

        with self.assertRaisesRegex(ValueError, "recompute"):
            raw_recompute_metrics(self.write_steps(steps))

    def test_rejects_resume_without_prior_preemption(self) -> None:
        self.assertIsNotNone(raw_recompute_metrics)
        steps = self.valid_steps()
        steps[1]["scheduler"]["preemptions"] = []

        with self.assertRaisesRegex(ValueError, "resume|preemption"):
            raw_recompute_metrics(self.write_steps(steps))

    def test_rejects_scheduler_selection_mismatch(self) -> None:
        self.assertIsNotNone(raw_recompute_metrics)
        steps = self.valid_steps()
        steps[0]["scheduler"]["selected_prefill_ids"] = [99]

        with self.assertRaisesRegex(ValueError, "selected"):
            raw_recompute_metrics(self.write_steps(steps))

    def test_rejects_unresolved_preemption_in_complete_mode(self) -> None:
        self.assertIsNotNone(raw_recompute_metrics)
        steps = self.valid_steps()[:2]

        with self.assertRaisesRegex(ValueError, "pending|resume"):
            raw_recompute_metrics(self.write_steps(steps))

    def test_incomplete_mode_preserves_pending_sequence_ids(self) -> None:
        self.assertIsNotNone(raw_recompute_metrics)
        steps = self.valid_steps()[:2]

        metrics = raw_recompute_metrics(
            self.write_steps(steps),
            require_complete=False,
        )

        self.assertFalse(metrics["complete_recompute_lifecycle"])
        self.assertEqual(metrics["pending_recompute_seq_ids"], [1])

    def test_stage15_v2_step_schema_accepts_real_scheduler_shape(self) -> None:
        step = self.valid_steps()[0]
        step.update(
            {
                "schema_version": 2,
                "started_at": 0.0,
                "finished_at": 0.1,
                "duration_ms": 100.0,
            }
        )
        step["scheduler"] = {
            "policy": "recompute_aware",
            "state": "normal",
            "mode": "prefill",
            "kv_total_blocks": 8,
            "kv_used_blocks_before": 0,
            "kv_free_blocks_before": 8,
            "kv_used_blocks_after": 1,
            "kv_free_blocks_after": 7,
            "running_ids_before": [],
            "waiting_ids_before": [1],
            "running_ids_after": [1],
            "waiting_ids_after": [],
            "selected_decode_ids": [],
            "selected_prefill_ids": [1],
            "oldest_waiting_age": 0,
            "resident_costs": [],
            "preemptions": [],
            "selected_decode_request_ids": [],
            "selected_prefill_request_ids": ["r1"],
        }

        validate_step_schema(step)

    def test_stage15_v2_step_schema_rejects_missing_scheduler_field(self) -> None:
        step = self.valid_steps()[0]
        step.update(
            {
                "schema_version": 2,
                "started_at": 0.0,
                "finished_at": 0.1,
                "duration_ms": 100.0,
            }
        )

        with self.assertRaisesRegex(ValueError, "scheduler step schema"):
            validate_step_schema(step)

    def test_full_run_validator_rejects_failed_status(self) -> None:
        summary = self.write_json({"status": "failed"})

        with self.assertRaisesRegex(ValueError, "status"):
            load_validated_stage15_run(summary)

    def test_full_run_validator_rejects_tampered_artifact_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "config": root / "experiments/configs/config.json",
                "trace": root / "experiments/data/trace.json",
                "requests": root / "experiments/results/requests.jsonl",
                "steps": root / "experiments/results/steps.jsonl",
            }
            for name, path in paths.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"artifact": name}), encoding="utf-8")
            summary = root / "summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "config_path": str(paths["config"]),
                        "provenance": {
                            "trace_path": str(paths["trace"]),
                            "config_sha256": "0" * 64,
                            "trace_sha256": "0" * 64,
                        },
                        "artifacts": {
                            "requests_path": str(paths["requests"]),
                            "requests_sha256": "0" * 64,
                            "steps_path": str(paths["steps"]),
                            "steps_sha256": "0" * 64,
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch("experiments.aggregate_stage15.REPO_ROOT", root):
                with self.assertRaisesRegex(
                    ValueError,
                    "config artifact hash",
                ):
                    load_validated_stage15_run(summary)

    def test_real_stage15_pressure_smoke_passes_full_validation(self) -> None:
        summary = (
            Path(__file__).resolve().parents[1]
            / "experiments/results/final/scheduler/stage15/"
            "stage15_diagnostic_r3_kv8_pressure_seed1.summary.json"
        )
        self.assertTrue(summary.is_file())

        record = load_validated_stage15_run(
            summary,
            expected_policy="pressure_aware_decode",
            expected_capacity=8,
        )

        self.assertEqual(
            record["recompute_metrics"]["actual_recompute_tokens"],
            record["summary"]["recompute_observability"][
                "actual_recompute_tokens"
            ],
        )
        self.assertTrue(
            record["recompute_metrics"]["complete_recompute_lifecycle"]
        )

    def test_stage15_relocates_artifacts_from_another_checkout(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        source = (
            repository
            / "experiments/results/final/scheduler/stage15/"
            "stage15_diagnostic_r3_kv8_pressure_seed1.summary.json"
        )
        self.assertTrue(source.is_file())
        document = json.loads(source.read_text(encoding="utf-8"))
        for container, field in (
            (document, "config_path"),
            (document["provenance"], "trace_path"),
            (document["artifacts"], "requests_path"),
            (document["artifacts"], "steps_path"),
        ):
            marker = "/experiments/"
            suffix = container[field].replace("\\", "/").split(marker, 1)[1]
            container[field] = f"/different/checkout/experiments/{suffix}"
        document["working_directory"] = "/different/checkout"
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / source.name
            summary.write_text(json.dumps(document), encoding="utf-8")

            record = load_validated_stage15_run(
                summary,
                expected_policy="pressure_aware_decode",
                expected_capacity=8,
            )

        self.assertTrue(
            record["recompute_metrics"]["complete_recompute_lifecycle"]
        )

    def test_stage15_resolver_ignores_existing_old_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = root / "current"
            old = root / "old"
            current_artifact = current / "experiments/data/trace.json"
            old_artifact = old / "experiments/data/trace.json"
            current_artifact.parent.mkdir(parents=True)
            old_artifact.parent.mkdir(parents=True)
            current_artifact.write_text("current\n", encoding="utf-8")
            old_artifact.write_text("old\n", encoding="utf-8")

            with patch("experiments.aggregate_stage15.REPO_ROOT", current):
                resolved = _resolve_recorded_artifact_path(
                    str(old_artifact),
                    label="trace",
                    allowed_directory=current / "experiments/data",
                )

            self.assertEqual(resolved, current_artifact.resolve())

    def test_stage15_relative_trace_uses_current_repository_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "experiments/data/trace.json"
            trace.parent.mkdir(parents=True)
            trace.write_text("{}\n", encoding="utf-8")

            with patch("experiments.aggregate_stage15.REPO_ROOT", root):
                resolved = _resolve_config_trace_path(
                    "experiments/data/trace.json"
                )

            self.assertEqual(resolved, trace.resolve())

    def test_frozen_stage15_r3_manifest_has_exact_18_cells(self) -> None:
        manifest = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "experiments/configs/stage15_diagnostic_r3_matrix.json"
            ).read_text(encoding="utf-8")
        )

        validate_stage15_manifest(manifest)

        manifest["required_cells"] = 17
        with self.assertRaisesRegex(ValueError, "manifest"):
            validate_stage15_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
