from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments.benchmark.open_loop import (
    OpenLoopResult,
    load_trace,
    prepare_requests,
)
from experiments.run_open_loop import (
    RUN_LIMITATIONS,
    build_pressure_aware_summary,
    build_pressure_summary,
    build_recompute_observability,
    error_has_cuda_oom,
    fail_result,
    file_sha256,
    load_config,
    main,
    python_package_sha256,
    resolve_output_paths,
    runtime_metadata,
    startup_failure_result,
    validate_kv_request_feasibility,
)
from nanovllm.engine.model_runner import resolve_num_kvcache_blocks
from nanovllm.config import Config


REPO_ROOT = Path(__file__).resolve().parents[1]


class TraceValidationTests(unittest.TestCase):
    def write_json(self, value) -> Path:
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

    def valid_trace(self) -> dict:
        return {
            "schema_version": 1,
            "description": "test trace",
            "time_unit": "seconds",
            "requests": [
                {
                    "request_id": "interactive-000",
                    "request_class": "interactive",
                    "arrival_time": 0.0,
                    "prompt_length": 2,
                    "max_output_tokens": 2,
                    "seed": 1,
                }
            ],
        }

    def test_trace_rejects_nonfinite_arrival(self) -> None:
        trace = self.valid_trace()
        trace["requests"][0]["arrival_time"] = float("nan")

        with self.assertRaisesRegex(ValueError, "finite"):
            load_trace(self.write_json(trace))

    def test_trace_rejects_wrong_time_unit(self) -> None:
        trace = self.valid_trace()
        trace["time_unit"] = "milliseconds"

        with self.assertRaisesRegex(ValueError, "time_unit"):
            load_trace(self.write_json(trace))

    def test_config_rejects_existing_run_id(self) -> None:
        config_path = REPO_ROOT / "experiments/configs/open_loop_smoke.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["output"]["run_id"] = f"unit_test_existing_{os.getpid()}"
        paths = resolve_output_paths(config["output"])
        paths["summary"].parent.mkdir(parents=True, exist_ok=True)
        paths["summary"].touch()
        self.addCleanup(paths["summary"].unlink, missing_ok=True)

        with self.assertRaisesRegex(FileExistsError, "overwrite"):
            load_config(self.write_json(config))

    def test_legacy_engine_config_gets_prefill_defaults(self) -> None:
        config = json.loads(
            (
                REPO_ROOT / "experiments/configs/open_loop_smoke.json"
            ).read_text(encoding="utf-8")
        )
        config["output"]["run_id"] = f"unit_test_legacy_{os.getpid()}"

        loaded = load_config(self.write_json(config))

        self.assertEqual(loaded["engine"]["scheduler_policy"], "prefill_first")
        self.assertEqual(loaded["engine"]["decode_token_budget"], 0)
        self.assertEqual(loaded["engine"]["decode_step_guard"], 0)
        self.assertFalse(loaded["profiling"]["enabled"])

    def test_profiling_config_accepts_optional_profiler_contract(self) -> None:
        config = json.loads(
            (
                REPO_ROOT / "experiments/configs/open_loop_smoke.json"
            ).read_text(encoding="utf-8")
        )
        config["output"]["run_id"] = f"unit_test_profiling_{os.getpid()}"
        config["profiling"] = {
            "enabled": True,
            "output_directory": "experiments/profiling",
            "record_shapes": True,
            "profile_memory": True,
            "with_stack": False,
            "wait_steps": 0,
            "warmup_steps": 1,
            "active_steps": 2,
            "repeat": 1,
            "cuda_events": True,
        }

        loaded = load_config(self.write_json(config))

        self.assertTrue(loaded["profiling"]["enabled"])
        self.assertTrue(loaded["profiling"]["cuda_events"])

    def test_profiling_config_rejects_non_positive_schedule(self) -> None:
        config = json.loads(
            (
                REPO_ROOT / "experiments/configs/open_loop_smoke.json"
            ).read_text(encoding="utf-8")
        )
        config["profiling"] = {
            "enabled": True,
            "output_directory": "experiments/profiling",
            "record_shapes": True,
            "profile_memory": True,
            "with_stack": False,
            "wait_steps": 0,
            "warmup_steps": 0,
            "active_steps": 0,
            "repeat": 1,
            "cuda_events": True,
        }

        with self.assertRaisesRegex(ValueError, "active_steps"):
            load_config(self.write_json(config))

    def test_config_rejects_mismatched_cuda_event_timing(self) -> None:
        config = json.loads(
            (
                REPO_ROOT / "experiments/configs/open_loop_smoke.json"
            ).read_text(encoding="utf-8")
        )
        config["output"]["run_id"] = f"unit_test_cuda_timing_{os.getpid()}"
        config["engine"]["cuda_event_timing"] = True
        config["profiling"] = {
            "enabled": False,
            "output_directory": "experiments/profiling",
            "record_shapes": True,
            "profile_memory": True,
            "with_stack": False,
            "wait_steps": 0,
            "warmup_steps": 1,
            "active_steps": 1,
            "repeat": 1,
            "cuda_events": False,
        }

        with self.assertRaisesRegex(ValueError, "cuda_event_timing"):
            load_config(self.write_json(config))

    def test_budgeted_scheduler_config_requires_positive_budget(self) -> None:
        source = json.loads(
            (
                REPO_ROOT / "experiments/configs/open_loop_smoke.json"
            ).read_text(encoding="utf-8")
        )
        source["output"]["run_id"] = f"unit_test_policy_{os.getpid()}"
        source["engine"]["scheduler_policy"] = "decode_first_budgeted"
        source["engine"]["decode_token_budget"] = 0

        with self.assertRaisesRegex(ValueError, "positive"):
            load_config(self.write_json(source))

        source["engine"]["decode_token_budget"] = 16
        loaded = load_config(self.write_json(source))
        self.assertEqual(
            loaded["engine"]["scheduler_policy"],
            "decode_first_budgeted",
        )
        self.assertEqual(loaded["engine"]["decode_token_budget"], 16)

    def test_runner_accepts_valid_decode_step_guard(self) -> None:
        source = json.loads(
            (
                REPO_ROOT / "experiments/configs/open_loop_smoke.json"
            ).read_text(encoding="utf-8")
        )
        source["output"]["run_id"] = f"unit_test_step_guard_{os.getpid()}"
        source["engine"]["scheduler_policy"] = "decode_first_budgeted"
        source["engine"]["decode_token_budget"] = 8
        source["engine"]["decode_step_guard"] = 2

        loaded = load_config(self.write_json(source))

        self.assertEqual(loaded["engine"]["decode_step_guard"], 2)

    def test_pressure_aware_engine_config_accepts_valid_contract(self) -> None:
        source = json.loads(
            (
                REPO_ROOT
                / "experiments/configs/open_loop_smoke.json"
            ).read_text(encoding="utf-8")
        )
        source["output"]["run_id"] = f"unit_test_pressure_{os.getpid()}"
        source["engine"].update(
            {
                "scheduler_policy": "pressure_aware_decode",
                "decode_token_budget": 8,
                "decode_step_guard": 0,
                "pressure_decode_token_budget": 4,
                "pressure_decode_step_guard": 0,
                "pressure_high_utilization": 0.75,
                "pressure_critical_utilization": 1.0,
                "pressure_preemption_window": 4,
                "pressure_preemption_threshold": 2,
                "pressure_hysteresis_steps": 2,
                "pressure_waiting_age_threshold": 4,
            }
        )

        loaded = load_config(self.write_json(source))

        self.assertEqual(
            loaded["engine"]["scheduler_policy"],
            "pressure_aware_decode",
        )
        self.assertEqual(
            loaded["engine"]["pressure_decode_token_budget"],
            4,
        )

    def test_chunked_prefill_engine_config_accepts_valid_contract(self) -> None:
        source = json.loads(
            (
                REPO_ROOT
                / "experiments/configs/open_loop_smoke.json"
            ).read_text(encoding="utf-8")
        )
        source["output"]["run_id"] = (
            f"unit_test_chunked_prefill_{os.getpid()}"
        )
        source["engine"].update(
            {
                "scheduler_policy": "chunked_prefill_budgeted",
                "decode_token_budget": 8,
                "decode_step_guard": 0,
                "prefill_chunk_token_budget": 128,
                "pressure_decode_token_budget": 4,
                "pressure_decode_step_guard": 0,
                "pressure_high_utilization": 0.75,
                "pressure_critical_utilization": 1.0,
                "pressure_preemption_window": 4,
                "pressure_preemption_threshold": 2,
                "pressure_hysteresis_steps": 2,
                "pressure_waiting_age_threshold": 4,
            }
        )

        loaded = load_config(self.write_json(source))

        self.assertEqual(
            loaded["engine"]["scheduler_policy"],
            "chunked_prefill_budgeted",
        )
        self.assertEqual(
            loaded["engine"]["prefill_chunk_token_budget"],
            128,
        )

    def test_recompute_aware_engine_config_accepts_valid_contract(self) -> None:
        source = json.loads(
            (
                REPO_ROOT
                / "experiments/configs/open_loop_smoke.json"
            ).read_text(encoding="utf-8")
        )
        source["output"]["run_id"] = (
            f"unit_test_recompute_aware_{os.getpid()}"
        )
        source["engine"].update(
            {
                "scheduler_policy": "recompute_aware",
                "decode_token_budget": 8,
                "decode_step_guard": 0,
                "pressure_decode_token_budget": 4,
                "pressure_decode_step_guard": 0,
                "pressure_high_utilization": 0.75,
                "pressure_critical_utilization": 1.0,
                "pressure_preemption_window": 4,
                "pressure_preemption_threshold": 2,
                "pressure_hysteresis_steps": 2,
                "pressure_waiting_age_threshold": 4,
            }
        )

        loaded = load_config(self.write_json(source))

        self.assertEqual(
            loaded["engine"]["scheduler_policy"],
            "recompute_aware",
        )
        self.assertEqual(
            loaded["engine"]["pressure_high_utilization"],
            0.75,
        )

    def test_bounded_recompute_engine_config_reuses_slo_contract(self) -> None:
        source = json.loads(
            (
                REPO_ROOT
                / "experiments/configs/open_loop_smoke.json"
            ).read_text(encoding="utf-8")
        )
        source["output"]["run_id"] = (
            f"unit_test_bounded_recompute_{os.getpid()}"
        )
        source["engine"].update(
            {
                "scheduler_policy": "recompute_aware_bounded",
                "decode_token_budget": 8,
                "decode_step_guard": 0,
                "pressure_decode_token_budget": 4,
                "pressure_decode_step_guard": 0,
                "pressure_high_utilization": 0.75,
                "pressure_critical_utilization": 1.0,
                "pressure_preemption_window": 4,
                "pressure_preemption_threshold": 2,
                "pressure_hysteresis_steps": 2,
                "pressure_waiting_age_threshold": 4,
                "max_drain_steps": 16,
                "waiting_age_limit": 32,
                "ttft_slo_ms": source["slo"]["ttft_slo_ms"],
                "itl_slo_ms": source["slo"]["itl_slo_ms"],
            }
        )

        loaded = load_config(self.write_json(source))

        self.assertEqual(
            loaded["engine"]["scheduler_policy"],
            "recompute_aware_bounded",
        )
        self.assertEqual(loaded["engine"]["max_drain_steps"], 16)
        self.assertEqual(loaded["engine"]["waiting_age_limit"], 32)

        source["engine"]["itl_slo_ms"] += 1
        with self.assertRaisesRegex(ValueError, "must match"):
            load_config(self.write_json(source))

    def test_chunked_prefill_engine_rejects_budget_above_batch_limit(
        self,
    ) -> None:
        source = json.loads(
            (
                REPO_ROOT
                / "experiments/configs/open_loop_smoke.json"
            ).read_text(encoding="utf-8")
        )
        source["output"]["run_id"] = (
            f"unit_test_chunked_prefill_invalid_{os.getpid()}"
        )
        source["engine"].update(
            {
                "scheduler_policy": "chunked_prefill_budgeted",
                "decode_token_budget": 8,
                "decode_step_guard": 0,
                "prefill_chunk_token_budget": (
                    source["engine"]["max_num_batched_tokens"] + 1
                ),
                "pressure_decode_token_budget": 4,
                "pressure_decode_step_guard": 0,
                "pressure_high_utilization": 0.75,
                "pressure_critical_utilization": 1.0,
                "pressure_preemption_window": 4,
                "pressure_preemption_threshold": 2,
                "pressure_hysteresis_steps": 2,
                "pressure_waiting_age_threshold": 4,
            }
        )

        with self.assertRaisesRegex(
            ValueError,
            "prefill_chunk_token_budget",
        ):
            load_config(self.write_json(source))

    def test_pressure_aware_engine_rejects_budget_above_normal(self) -> None:
        source = json.loads(
            (
                REPO_ROOT
                / "experiments/configs/open_loop_smoke.json"
            ).read_text(encoding="utf-8")
        )
        source["output"]["run_id"] = f"unit_test_pressure_invalid_{os.getpid()}"
        source["engine"].update(
            {
                "scheduler_policy": "pressure_aware_decode",
                "decode_token_budget": 4,
                "decode_step_guard": 0,
                "pressure_decode_token_budget": 8,
                "pressure_decode_step_guard": 0,
                "pressure_high_utilization": 0.75,
                "pressure_critical_utilization": 1.0,
                "pressure_preemption_window": 4,
                "pressure_preemption_threshold": 2,
                "pressure_hysteresis_steps": 2,
                "pressure_waiting_age_threshold": 4,
            }
        )

        with self.assertRaisesRegex(ValueError, "pressure_decode_token_budget"):
            load_config(self.write_json(source))

    def test_static_scheduler_rejects_pressure_fields(self) -> None:
        source = json.loads(
            (
                REPO_ROOT / "experiments/configs/open_loop_smoke.json"
            ).read_text(encoding="utf-8")
        )
        source["output"]["run_id"] = f"unit_test_static_pressure_{os.getpid()}"
        source["engine"]["pressure_decode_token_budget"] = 4

        with self.assertRaisesRegex(ValueError, "pressure"):
            load_config(self.write_json(source))

    def test_runner_accepts_only_positive_explicit_kv_block_count(
        self,
    ) -> None:
        source = json.loads(
            (
                REPO_ROOT / "experiments/configs/open_loop_smoke.json"
            ).read_text(encoding="utf-8")
        )
        source["output"]["run_id"] = f"unit_test_kv_blocks_{os.getpid()}"
        source["engine"]["num_kvcache_blocks"] = 4

        loaded = load_config(self.write_json(source))
        self.assertEqual(loaded["engine"]["num_kvcache_blocks"], 4)

        for invalid in (0, -1, True, 1.5):
            with self.subTest(invalid=invalid):
                source["engine"]["num_kvcache_blocks"] = invalid
                with self.assertRaisesRegex(ValueError, "num_kvcache_blocks"):
                    load_config(self.write_json(source))

    def test_explicit_kv_capacity_must_fit_each_complete_request(
        self,
    ) -> None:
        specs = load_trace(
            REPO_ROOT
            / "experiments/data/baseline_matrix/"
            "calibration_prefill_first_overload_seed1.json"
        )
        engine = {
            "kvcache_block_size": 256,
            "num_kvcache_blocks": 3,
        }

        with self.assertRaisesRegex(
            ValueError,
            "long-000.*requires 4.*provides 3",
        ):
            validate_kv_request_feasibility(specs, engine)

        engine["num_kvcache_blocks"] = 4
        validate_kv_request_feasibility(specs, engine)
        validate_kv_request_feasibility(
            specs,
            {"kvcache_block_size": 256},
        )

    def test_kv_footprint_excludes_the_final_emitted_token(self) -> None:
        engine = {
            "kvcache_block_size": 256,
            "num_kvcache_blocks": 1,
        }
        cases = (
            (256, 1, True),
            (256, 2, False),
            (255, 2, True),
        )

        for prompt_length, max_output_tokens, accepted in cases:
            with self.subTest(
                prompt_length=prompt_length,
                max_output_tokens=max_output_tokens,
            ):
                spec = SimpleNamespace(
                    request_id="boundary",
                    prompt_length=prompt_length,
                    max_output_tokens=max_output_tokens,
                )
                if accepted:
                    validate_kv_request_feasibility([spec], engine)
                else:
                    with self.assertRaisesRegex(
                        ValueError,
                        "requires 2.*provides 1",
                    ):
                        validate_kv_request_feasibility([spec], engine)

    def test_warmup_uses_the_same_kv_footprint_preflight(self) -> None:
        engine = {
            "kvcache_block_size": 256,
            "num_kvcache_blocks": 1,
        }
        warmup = {
            "enabled": True,
            "prompt_length": 256,
            "max_output_tokens": 2,
        }

        with self.assertRaisesRegex(ValueError, "warmup.*requires 2"):
            validate_kv_request_feasibility(
                [],
                engine,
                warmup=warmup,
            )

    def test_explicit_kv_block_count_overrides_auto_computation(self) -> None:
        self.assertEqual(resolve_num_kvcache_blocks(-1, 99), 99)
        self.assertEqual(resolve_num_kvcache_blocks(4, 99), 4)
        with self.assertRaisesRegex(ValueError, "safe capacity"):
            resolve_num_kvcache_blocks(100, 99)

    def test_core_config_rejects_invalid_explicit_kv_block_count(
        self,
    ) -> None:
        with (
            tempfile.TemporaryDirectory() as model_directory,
            patch(
                "nanovllm.config.AutoConfig.from_pretrained",
                return_value=SimpleNamespace(max_position_embeddings=1024),
            ),
        ):
            for invalid in (0, -2, True, 1.5):
                with self.subTest(invalid=invalid):
                    with self.assertRaisesRegex(
                        ValueError,
                        "num_kvcache_blocks",
                    ):
                        Config(
                            model_directory,
                            num_kvcache_blocks=invalid,
                        )

    def test_pressure_summary_preserves_rejection_and_exact_oom_semantics(
        self,
    ) -> None:
        class FakeEngine:
            def observability_snapshot(self):
                return {
                    "preemption_count": 3,
                    "kv_cache": {
                        "total_blocks": 4,
                        "final_used_blocks": 0,
                        "final_free_blocks": 4,
                        "peak_used_blocks": 4,
                        "peak_utilization": 1.0,
                    },
                    "pressure_aware": {
                        "state": "pressure",
                        "decision_count": 7,
                    },
                }

        terminal_counts = {"Rejected": 2}
        gpu_memory = {
            "allocated_bytes_current": 10,
            "allocated_bytes_peak": 20,
            "reserved_bytes_current": 30,
            "reserved_bytes_peak": 40,
            "device_free_bytes": 50,
            "device_total_bytes": 60,
        }
        pressure = build_pressure_summary(
            FakeEngine(),
            terminal_counts=terminal_counts,
            error={"type": "OutOfMemoryError", "is_cuda_oom": True},
            gpu_memory=gpu_memory,
            observation_active=True,
            admission_rejection_supported=False,
        )

        self.assertEqual(pressure["preemption_count"], 3)
        self.assertEqual(pressure["rejected_requests"], 2)
        self.assertFalse(pressure["admission_rejection_supported"])
        self.assertTrue(pressure["oom_detected"])
        self.assertEqual(pressure["gpu_memory"], gpu_memory)
        self.assertEqual(
            build_pressure_aware_summary(FakeEngine()),
            {"state": "pressure", "decision_count": 7},
        )

    def test_recompute_observability_exposes_scheduler_cross_check(self) -> None:
        class FakeEngine:
            def observability_snapshot(self):
                return {
                    "preemption_count": 3,
                    "actual_recompute_tokens": 17,
                    "resume_count": 2,
                }

        self.assertEqual(
            build_recompute_observability(FakeEngine()),
            {
                "schema_version": 1,
                "preemption_count": 3,
                "actual_recompute_tokens": 17,
                "resume_count": 2,
            },
        )

    def test_nested_cleanup_or_collection_oom_is_not_lost(self) -> None:
        self.assertTrue(
            error_has_cuda_oom(
                {
                    "type": "RuntimeError",
                    "is_cuda_oom": False,
                    "cleanup_error": {
                        "type": "OutOfMemoryError",
                        "is_cuda_oom": True,
                    },
                }
            )
        )
        self.assertFalse(
            error_has_cuda_oom(
                {"type": "RuntimeError", "is_cuda_oom": False}
            )
        )

    def test_pressure_collection_failure_forces_failed_result(self) -> None:
        original = OpenLoopResult(
            status="passed",
            records=[],
            step_events=[],
            summary={},
            error=None,
        )
        for is_cuda_oom in (False, True):
            with self.subTest(is_cuda_oom=is_cuda_oom):
                collection_error = {
                    "type": (
                        "OutOfMemoryError"
                        if is_cuda_oom
                        else "RuntimeError"
                    ),
                    "message": "pressure snapshot failed",
                    "is_cuda_oom": is_cuda_oom,
                }

                failed = fail_result(
                    original,
                    collection_error,
                    label="pressure_collection_error",
                )

                self.assertEqual(failed.status, "failed")
                self.assertEqual(
                    failed.error["is_cuda_oom"],
                    is_cuda_oom,
                )
                self.assertEqual(
                    failed.error["pressure_collection_error"],
                    collection_error,
                )

    def test_scheduler_config_rejects_invalid_policy_budget_pairs(self) -> None:
        source = json.loads(
            (
                REPO_ROOT / "experiments/configs/open_loop_smoke.json"
            ).read_text(encoding="utf-8")
        )
        source["output"]["run_id"] = f"unit_test_policy_pairs_{os.getpid()}"
        invalid_values = (
            ("unknown", 0, 0),
            ([], 0, 0),
            ("prefill_first", 1, 0),
            ("prefill_first", False, 0),
            ("prefill_first", 0, 1),
            ("decode_first_budgeted", True, 0),
            ("decode_first_budgeted", 8, True),
            ("decode_first_budgeted", 8, -1),
        )

        for policy, budget, guard in invalid_values:
            with self.subTest(policy=policy, budget=budget, guard=guard):
                source["engine"]["scheduler_policy"] = policy
                source["engine"]["decode_token_budget"] = budget
                source["engine"]["decode_step_guard"] = guard
                with self.assertRaisesRegex(
                    ValueError,
                    "scheduler_policy|decode_token_budget|decode_step_guard",
                ):
                    load_config(self.write_json(source))

    def test_startup_failure_cancels_unadmitted_requests(self) -> None:
        config_path = REPO_ROOT / "experiments/configs/open_loop_smoke.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        specs = load_trace(
            REPO_ROOT / "experiments/data/synthetic_smoke_trace.json"
        )

        result = startup_failure_result(
            specs,
            config,
            RuntimeError("synthetic startup failure"),
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(
            {record.terminal_state.value for record in result.records},
            {"Cancelled"},
        )
        self.assertTrue(result.summary["terminal_counts"]["reconciled"])

    def test_prompt_generation_is_repeatable_for_fixed_trace_seed(self) -> None:
        specs = load_trace(
            REPO_ROOT / "experiments/data/synthetic_smoke_trace.json"
        )

        first = prepare_requests(specs, token_id_upper_bound=10000)
        second = prepare_requests(specs, token_id_upper_bound=10000)

        self.assertEqual(
            [request.prompt_token_ids for request in first],
            [request.prompt_token_ids for request in second],
        )

    def test_runtime_metadata_survives_unavailable_cuda(self) -> None:
        with (
            patch(
                "experiments.run_open_loop.torch.cuda.is_available",
                return_value=False,
            ),
            patch(
                "experiments.run_open_loop.torch.cuda.get_device_name",
                side_effect=AssertionError("must not be called"),
            ),
        ):
            metadata = runtime_metadata()

        self.assertFalse(metadata["cuda_available"])
        self.assertIsNone(metadata["gpu"])

    def test_python_package_hash_binds_paths_and_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.py").write_text("VALUE = 1\n", encoding="utf-8")
            first = python_package_sha256(root)
            (root / "a.py").write_text("VALUE = 2\n", encoding="utf-8")
            second = python_package_sha256(root)
            (root / "nested").mkdir()
            (root / "nested/b.py").write_text(
                "VALUE = 2\n",
                encoding="utf-8",
            )
            third = python_package_sha256(root)

        self.assertNotEqual(first, second)
        self.assertNotEqual(second, third)

    def test_cli_startup_failure_writes_artifacts_and_exits_one(self) -> None:
        source_config = json.loads(
            (
                REPO_ROOT / "experiments/configs/open_loop_smoke.json"
            ).read_text(encoding="utf-8")
        )
        model_directory = tempfile.TemporaryDirectory()
        self.addCleanup(model_directory.cleanup)
        source_config["model"]["path"] = model_directory.name
        source_config["output"]["run_id"] = (
            f"unit_test_startup_failure_{os.getpid()}"
        )
        config_path = self.write_json(source_config)
        output_paths = resolve_output_paths(source_config["output"])
        for path in output_paths.values():
            self.addCleanup(path.unlink, missing_ok=True)

        with (
            patch(
                "experiments.run_open_loop.LLM",
                side_effect=RuntimeError("synthetic startup failure"),
            ),
            patch(
                "experiments.run_open_loop.verify_model_files",
                return_value=source_config["model"]["sha256"],
            ),
            patch(
                "experiments.run_open_loop.runtime_metadata",
                return_value={"test_runtime": True},
            ),
            patch(
                "experiments.run_open_loop.git_output",
                return_value="clean-snapshot-test-head",
            ),
            patch(
                "sys.argv",
                ["run_open_loop", "--config", str(config_path)],
            ),
            self.assertRaises(SystemExit) as exit_context,
        ):
            main()

        self.assertEqual(exit_context.exception.code, 1)
        self.assertTrue(all(path.is_file() for path in output_paths.values()))
        summary = json.loads(
            output_paths["summary"].read_text(encoding="utf-8")
        )
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["error"]["type"], "RuntimeError")
        self.assertEqual(
            summary["summary"]["terminal_counts"]["Cancelled"],
            3,
        )
        self.assertTrue(
            summary["summary"]["terminal_counts"]["reconciled"]
        )
        self.assertEqual(
            summary["provenance"]["scheduler_sha256"],
            file_sha256(REPO_ROOT / "nanovllm/engine/scheduler.py"),
        )
        self.assertEqual(
            summary["provenance"]["config_module_sha256"],
            file_sha256(REPO_ROOT / "nanovllm/config.py"),
        )
        self.assertEqual(summary["limitations"], list(RUN_LIMITATIONS))
        self.assertFalse(
            any(
                "SLO thresholds are not calibrated" in item
                for item in summary["limitations"]
            )
        )


if __name__ == "__main__":
    unittest.main()
