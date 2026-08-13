from __future__ import annotations

import unittest
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from experiments.benchmark.open_loop import (
    AdmissionConfig,
    ReservationRuntime,
    SlackEstimator,
    PreparedRequest,
    RequestSpec,
    prepare_requests,
    run_open_loop,
)
from experiments.run_open_loop import load_config, main as runner_main
from experiments.aggregate_stage12 import (
    execution_code_fingerprint,
    load_frozen_manifest,
    validate_admission_events,
    validate_run_artifacts,
    validate_slack_events,
)
from experiments.derive_stage12_eta import derive
from experiments.benchmark.lifecycle import RequestRecord, TerminalState, summarize_requests
from nanovllm.engine.block_manager import BlockManager
from nanovllm import SamplingParams
from nanovllm.engine.block_manager import PrefixCachePreview
from nanovllm.engine.llm_engine import EngineStepResult, SequenceStepEvent


class VirtualClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def sleep(self, duration: float) -> None:
        self.value += duration


class SlackFakeEngine:
    def __init__(self, clock: VirtualClock) -> None:
        self.clock = clock
        self.preview_count = 0
        self.preview_history: list[tuple[tuple[int, ...], int]] = []
        self.next_seq_id = 1
        self.active: set[int] = set()
        self.step_count = 0
        self.active_prefix: list[int] | None = None

    def preview_prefix_cache(
        self, prompt: list[int], max_tokens: int
    ) -> PrefixCachePreview:
        self.preview_count += 1
        self.preview_history.append((tuple(prompt), self.step_count))
        if self.active_prefix is not None and prompt[:4] == self.active_prefix:
            return PrefixCachePreview(
                full_required_blocks=3,
                matched_prefix_blocks=1,
                active_shared_blocks=1,
                inactive_cached_blocks=0,
                matched_prefix_tokens=4,
                incremental_reservation_blocks=2,
                matched_block_ids=(1,),
                active_block_ids=(1,),
                inactive_block_ids=(),
            )
        return PrefixCachePreview(
            full_required_blocks=3,
            matched_prefix_blocks=0,
            active_shared_blocks=0,
            inactive_cached_blocks=0,
            matched_prefix_tokens=0,
            incremental_reservation_blocks=3,
        )

    def cache_state_snapshot(self) -> list[dict]:
        if self.step_count == 0:
            return []
        assert self.active_prefix is not None
        prefix = self.active_prefix
        return [
            {
                "block_id": 1,
                "hash": BlockManager.compute_hash(prefix, -1),
                "token_ids": prefix,
                "used": True,
                "ref_count": 1,
            }
        ]

    def add_request(self, prompt: list[int], sampling_params: SamplingParams) -> int:
        sequence_id = self.next_seq_id
        self.next_seq_id += 1
        self.active.add(sequence_id)
        if self.active_prefix is None:
            self.active_prefix = prompt[:4]
        return sequence_id

    def is_finished(self) -> bool:
        return not self.active

    def step_with_events(self) -> EngineStepResult:
        self.clock.value += 0.01
        active = sorted(self.active)
        self.step_count += 1
        if self.step_count == 1:
            return EngineStepResult(
                outputs=[],
                phase="decode",
                num_scheduled_tokens=0,
                signed_token_count=0,
                events=[],
            )
        finished = self.step_count >= 3
        if finished:
            self.active.clear()
        return EngineStepResult(
            outputs=[(sequence_id, [100 + sequence_id]) for sequence_id in active],
            phase="decode",
            num_scheduled_tokens=len(active),
            signed_token_count=-len(active),
            events=[
                SequenceStepEvent(
                    seq_id=sequence_id,
                    phase="decode",
                    num_scheduled_tokens=1,
                    emitted_token_id=100 + sequence_id,
                    finished=finished,
                )
                for sequence_id in active
            ],
        )


class SlackEstimatorTests(unittest.TestCase):
    def write_json(self, value: dict) -> Path:
        handle = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".json", delete=False
        )
        with handle:
            json.dump(value, handle)
        self.addCleanup(Path(handle.name).unlink, missing_ok=True)
        return Path(handle.name)

    def write_stage12_artifacts(
        self,
        *,
        result: object,
        trace: dict,
    ) -> tuple[Path, Path, Path]:
        repository = Path(__file__).resolve().parents[1]
        config_directory = tempfile.TemporaryDirectory(
            dir=repository / "experiments/configs"
        )
        trace_directory = tempfile.TemporaryDirectory(
            dir=repository / "experiments/data"
        )
        result_directory = tempfile.TemporaryDirectory(
            dir=repository / "experiments/results"
        )
        derivation_directory = tempfile.TemporaryDirectory(
            dir=repository / "experiments"
        )
        for directory in (
            config_directory,
            trace_directory,
            result_directory,
            derivation_directory,
        ):
            self.addCleanup(directory.cleanup)
        root = Path(result_directory.name)
        run_id = "unit_stage12_verified"
        trace_path = Path(trace_directory.name) / "trace.json"
        trace_path.write_text(json.dumps(trace), encoding="utf-8")
        documents = {
            "requests": [record.to_dict() for record in result.records],
            "steps": result.step_events,
            "admission_events": result.admission_events,
            "cache_events": result.cache_events,
            "cache_states": result.cache_states,
            "slack_events": result.slack_events,
        }
        suffixes = {
            "requests": "requests",
            "steps": "steps",
            "admission_events": "admission",
            "cache_events": "cache",
            "cache_states": "cache_states",
            "slack_events": "slack",
        }
        artifacts = {}
        for name, document in documents.items():
            path = root / f"{run_id}.{suffixes[name]}.jsonl"
            path.write_text(
                "".join(json.dumps(item) + "\n" for item in document),
                encoding="utf-8",
            )
            artifacts[f"{name}_path"] = str(path)
            artifacts[f"{name}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        config = {
            "upstream_commit": "0" * 40,
            "model": {
                "repo_id": "unit/model",
                "revision": "unit-revision",
                "sha256": {"model.safetensors": "unit-hash"},
            },
            "workload": {"trace_path": str(trace_path)},
            "output": {"directory": str(root), "run_id": run_id},
            "engine": {"kvcache_block_size": 4, "enforce_eager": True},
            "sampling": {"temperature": 1.0, "ignore_eos": True, "seed": 1},
            "slo": {
                "ttft_slo_ms": 100.0,
                "itl_slo_ms": 100.0,
                "require_itl": True,
            },
            "measurement": {
                "start_seconds": 0.0,
                "end_seconds": 1.0,
                "max_run_seconds": 1.0,
            },
            "admission": {
                "policy": "slack_aware_prefix_fifo",
                "max_queue_wait_seconds": 0.10,
                "observe_prefix_cache": True,
                "eta_prefill_seconds": 0.02,
                "eta_decode_seconds_per_token": 0.04,
                "eta_safety_margin_seconds": 0.01,
            },
        }
        config_path = Path(config_directory.name) / f"{run_id}.config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        calibration_run_id = "unit_stage12_calibration"
        calibration_files = {
            "config": Path(config_directory.name) / "calibration.json",
            "trace": Path(trace_directory.name) / "calibration.json",
            "requests": root / "calibration.requests.jsonl",
            "steps": root / "calibration.steps.jsonl",
            "summary": root / "calibration.summary.json",
            "phase_timings": root / "calibration.phase_timings.jsonl",
            "derivation_script": Path(__file__).resolve().parents[1]
            / "experiments/derive_stage12_eta.py",
            "derivation_output": root / "calibration.eta.json",
        }
        calibration_config = {
            "model": config["model"],
            "engine": config["engine"],
            "sampling": config["sampling"],
            "slo": config["slo"],
            "measurement": config["measurement"],
            "admission": {"policy": "disabled"},
            "profiling": {"cuda_events": True},
            "output": {"run_id": calibration_run_id},
            "workload": {"trace_path": str(calibration_files["trace"])},
        }
        calibration_trace = {
            "schema_version": 2,
            "description": "unit calibration",
            "time_unit": "seconds",
            "requests": [{
                "request_id": "calibration-000",
                "request_class": "long",
                "arrival_time": 0.0,
                "prompt_length": 4,
                "max_output_tokens": 2,
                "seed": 1,
                "prefix_group": None,
                "shared_prefix_length": 0,
            }],
        }
        calibration_record = RequestRecord("calibration-000", "long", 0.0)
        calibration_record.mark_admitted(0.0)
        calibration_record.record_token(0.06)
        calibration_record.record_token(0.10)
        calibration_record.mark_terminal(TerminalState.FINISHED, 0.10)
        calibration_summary_metrics = summarize_requests(
            [calibration_record], measurement_start=0.0, measurement_end=1.0,
            ttft_slo_ms=100.0, itl_slo_ms=100.0, require_itl=True,
        )
        calibration_requests = [calibration_record.to_dict()]
        calibration_steps = [
            {"step_index": 0, "started_at": 0.0, "finished_at": 0.02,
             "duration_ms": 20.0, "phase": "prefill", "num_scheduled_tokens": 4,
             "events": [{"request_id": "calibration-000", "seq_id": 1,
                         "phase": "prefill", "num_scheduled_tokens": 4,
                         "emitted_token": False, "finished": False}]},
            {"step_index": 1, "started_at": 0.02, "finished_at": 0.06,
             "duration_ms": 40.0, "phase": "decode", "num_scheduled_tokens": 1,
             "events": [{"request_id": "calibration-000", "seq_id": 1,
                         "phase": "decode", "num_scheduled_tokens": 1,
                         "emitted_token": True, "finished": False}]},
            {"step_index": 2, "started_at": 0.06, "finished_at": 0.10,
             "duration_ms": 40.0, "phase": "decode", "num_scheduled_tokens": 1,
             "events": [{"request_id": "calibration-000", "seq_id": 1,
                         "phase": "decode", "num_scheduled_tokens": 1,
                         "emitted_token": True, "finished": True}]},
        ]
        calibration_timings = [
            {"step_index": 0, "model_runner_cuda_ms": 20.0, "step_wall_ms": 30.0},
            {"step_index": 1, "model_runner_cuda_ms": 40.0, "step_wall_ms": 50.0},
            {"step_index": 2, "model_runner_cuda_ms": 40.0, "step_wall_ms": 50.0},
        ]
        calibration_eta = derive(steps=calibration_steps, timings=calibration_timings)
        calibration_files["config"].write_text(json.dumps(calibration_config), encoding="utf-8")
        calibration_files["trace"].write_text(json.dumps(calibration_trace), encoding="utf-8")
        for name, document in (
            ("requests", calibration_requests),
            ("steps", calibration_steps),
            ("phase_timings", calibration_timings),
        ):
            calibration_files[name].write_text(
                "".join(json.dumps(item) + "\n" for item in document), encoding="utf-8"
            )
        calibration_summary = {
            "status": "passed", "error": None,
            "provenance": {
                "config_sha256": hashlib.sha256(calibration_files["config"].read_bytes()).hexdigest(),
                "trace_sha256": hashlib.sha256(calibration_files["trace"].read_bytes()).hexdigest(),
                **execution_code_fingerprint(),
            },
            "runtime": {"cuda_available": True, "gpu": "unit-gpu"},
            "measurement": config["measurement"], "slo": config["slo"],
            "model": {**config["model"], "verified_sha256": config["model"]["sha256"]},
            "engine": config["engine"],
            "sampling": config["sampling"],
            "summary": {**calibration_summary_metrics, "runtime": {"timed_out": False}},
            "artifacts": {},
        }
        for name in ("requests", "steps", "phase_timings"):
            calibration_summary["artifacts"][f"{name}_path"] = str(calibration_files[name])
            calibration_summary["artifacts"][f"{name}_sha256"] = hashlib.sha256(
                calibration_files[name].read_bytes()
            ).hexdigest()
        calibration_files["summary"].write_text(json.dumps(calibration_summary), encoding="utf-8")
        calibration_files["derivation_output"].write_text(
            json.dumps(calibration_eta), encoding="utf-8"
        )
        calibration = {
            **{
                f"{name}_path": str(path)
                for name, path in calibration_files.items()
            },
            **{
                f"{name}_sha256": hashlib.sha256(path.read_bytes()).hexdigest()
                for name, path in calibration_files.items()
            },
            "eta_prefill_seconds": calibration_eta["eta_prefill_seconds"],
            "eta_decode_seconds_per_token": calibration_eta["eta_decode_seconds_per_token"],
            "eta_safety_margin_seconds": calibration_eta["eta_safety_margin_seconds"],
        }
        manifest = {
            "schema_version": 1,
            "stage": "stage12",
            "kind": "smoke",
            "run_id": run_id,
            "config_path": str(config_path),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "trace_path": str(trace_path),
            "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
            "execution_code": execution_code_fingerprint(),
            "aggregate_sha256": hashlib.sha256(
                (Path(__file__).resolve().parents[1] / "experiments/aggregate_stage12.py").read_bytes()
            ).hexdigest(),
            "upstream_commit": config["upstream_commit"],
            "calibration": calibration,
        }
        manifest_path = root / f"{run_id}.manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        summary = {
            "status": "passed",
            "error": None,
            "provenance": {
                "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                **execution_code_fingerprint(),
            },
            "upstream_commit": config["upstream_commit"],
            "model": {**config["model"], "verified_sha256": config["model"]["sha256"]},
            "runtime": {"cuda_available": True, "gpu": "unit-gpu"},
            "admission": result.admission,
            "engine": config["engine"],
            "sampling": config["sampling"],
            "measurement": config["measurement"],
            "slo": config["slo"],
            "summary": result.summary,
            "pressure": {
                "oom_detected": False,
                "kv_cache": {"final_used_blocks": 0},
            },
            "artifacts": artifacts,
            "stage12_manifest": {
                "path": str(manifest_path),
                "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            },
        }
        summary_path = root / f"{run_id}.summary.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        return config_path, summary_path, manifest_path

    def setUp(self) -> None:
        self.config = AdmissionConfig(
            policy="slack_aware_prefix_fifo",
            total_kv_blocks=3,
            kvcache_block_size=4,
            max_queue_wait_seconds=0.10,
            observe_prefix_cache=True,
            eta_prefill_seconds=0.02,
            eta_decode_seconds_per_token=0.01,
            eta_safety_margin_seconds=0.01,
        )
        self.estimator = SlackEstimator(self.config)

    def test_unstarted_request_uses_conservative_prefill_plus_full_decode(self) -> None:
        runtime = ReservationRuntime(
            request_id="long-000",
            reservation_blocks=2,
            admitted_at=1.0,
            max_output_tokens=4,
        )

        self.assertEqual(self.estimator.predict_release_at(runtime), 1.07)

    def test_progress_uses_remaining_tokens_and_last_token_time(self) -> None:
        runtime = ReservationRuntime(
            request_id="interactive-000",
            reservation_blocks=1,
            admitted_at=1.0,
            max_output_tokens=5,
            generated_tokens=3,
            last_token_at=1.08,
        )

        self.assertEqual(self.estimator.predict_release_at(runtime), 1.11)

    def test_unknown_runtime_forces_conservative_fifo_fallback(self) -> None:
        runtime = ReservationRuntime(
            request_id="unknown-000",
            reservation_blocks=2,
            admitted_at=None,
            max_output_tokens=4,
        )
        self.assertIsNone(self.estimator.predict_release_at(runtime))
        self.assertIsNone(
            self.estimator.earliest_capacity_time([runtime], shortfall_blocks=1)
        )

    def test_eta_constants_are_rejected_for_non_slack_policy(self) -> None:
        with self.assertRaisesRegex(ValueError, "ETA fields"):
            AdmissionConfig(
                policy="disabled",
                eta_prefill_seconds=0.01,
            )

    def test_earliest_capacity_time_requires_enough_blocks_before_deadline(self) -> None:
        first = ReservationRuntime(
            request_id="first",
            reservation_blocks=1,
            admitted_at=0.0,
            max_output_tokens=4,
        )
        second = ReservationRuntime(
            request_id="second",
            reservation_blocks=2,
            admitted_at=0.02,
            max_output_tokens=2,
        )

        predicted, released = self.estimator.earliest_capacity_time(
            [first, second], shortfall_blocks=2
        )
        self.assertAlmostEqual(predicted, 0.07)
        self.assertEqual(released, 3)

    def test_runner_requires_all_positive_eta_constants(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        document = json.loads(
            (
                repo_root
                / "experiments/configs/stage11_prefix_admission_smoke_kv6_seed1.json"
            ).read_text(encoding="utf-8")
        )
        document["output"]["run_id"] = "unit_test_stage12_slack_loader"
        document["admission"] = {
            "policy": "slack_aware_prefix_fifo",
            "max_queue_wait_seconds": 0.25,
            "observe_prefix_cache": True,
            "eta_prefill_seconds": 0.02,
            "eta_decode_seconds_per_token": 0.01,
            "eta_safety_margin_seconds": 0.01,
        }

        loaded = load_config(self.write_json(document))
        self.assertEqual(loaded["admission"]["policy"], "slack_aware_prefix_fifo")

        document["output"]["run_id"] = "unit_test_stage12_missing_eta"
        document["admission"]["eta_safety_margin_seconds"] = 0.0
        with self.assertRaisesRegex(ValueError, "eta_safety_margin_seconds"):
            load_config(self.write_json(document))

    def test_runner_rejects_slack_config_without_pre_run_manifest(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        document = json.loads(
            (
                repo_root
                / "experiments/configs/stage11_prefix_admission_smoke_kv6_seed1.json"
            ).read_text(encoding="utf-8")
        )
        document["output"]["run_id"] = "unit_test_stage12_manifest_required"
        document["admission"] = {
            "policy": "slack_aware_prefix_fifo",
            "max_queue_wait_seconds": 0.25,
            "observe_prefix_cache": True,
            "eta_prefill_seconds": 0.02,
            "eta_decode_seconds_per_token": 0.01,
            "eta_safety_margin_seconds": 0.01,
        }
        path = self.write_json(document)
        with patch.object(sys, "argv", ["run_open_loop", "--config", str(path)]):
            with self.assertRaisesRegex(ValueError, "require --stage12-manifest"):
                runner_main()

    def test_predicted_deadline_miss_rejects_head_then_checks_next_fifo_request(self) -> None:
        clock = VirtualClock()
        specs = [
            RequestSpec("long-000", "long", 0.0, 8, 2, 1, "shared", 4),
            RequestSpec("interactive-000", "interactive", 0.005, 8, 2, 2),
            RequestSpec("interactive-001", "interactive", 0.005, 8, 2, 3, "shared", 4),
        ]
        requests = prepare_requests(specs, token_id_upper_bound=10000)

        result = run_open_loop(
            SlackFakeEngine(clock),
            requests,
            temperature=1.0,
            ignore_eos=True,
            measurement_start=0.0,
            measurement_end=1.0,
            ttft_slo_ms=100.0,
            itl_slo_ms=100.0,
            max_run_seconds=1.0,
            clock=clock,
            sleep=clock.sleep,
            synchronize=lambda: None,
            admission=AdmissionConfig(
                policy="slack_aware_prefix_fifo",
                total_kv_blocks=5,
                kvcache_block_size=4,
                max_queue_wait_seconds=0.10,
                observe_prefix_cache=True,
                eta_prefill_seconds=0.02,
                eta_decode_seconds_per_token=0.04,
                eta_safety_margin_seconds=0.01,
            ),
        )

        self.assertEqual(
            [record.terminal_state.value for record in result.records],
            ["Finished", "Rejected", "Finished"],
        )
        self.assertEqual(
            result.records[1].terminal_reason, "predicted_deadline_miss"
        )
        self.assertEqual(len(result.slack_events), 1)
        event = result.slack_events[0]
        self.assertEqual(event["request_id"], "interactive-000")
        self.assertEqual(event["capacity_shortfall_blocks"], 1)
        self.assertGreater(event["predicted_free_at"], event["deadline_at"])
        self.assertEqual(event["predicted_releasable_blocks"], 3)
        self.assertEqual(result.admission["final_reserved_blocks"], 0)
        trace = {
            "schema_version": 2,
            "description": "stage12 replay fixture",
            "time_unit": "seconds",
            "requests": [
                {
                    "request_id": request.spec.request_id,
                    "request_class": request.spec.request_class,
                    "arrival_time": request.spec.arrival_time,
                    "prompt_length": request.spec.prompt_length,
                    "max_output_tokens": request.spec.max_output_tokens,
                    "seed": request.spec.seed,
                    "prefix_group": request.spec.prefix_group,
                    "shared_prefix_length": request.spec.shared_prefix_length,
                }
                for request in requests
            ],
        }
        replay_inputs = {
            "admission_events": result.admission_events,
            "cache_events": result.cache_events,
            "cache_states": result.cache_states,
            "requests": [record.to_dict() for record in result.records],
            "steps": result.step_events,
            "trace": trace,
            "admission": result.admission,
            "block_size": 4,
        }
        validate_admission_events(**{
            key: value for key, value in replay_inputs.items()
        })
        validate_slack_events(
            slack_events=result.slack_events,
            **replay_inputs,
        )
        config_path, summary_path, manifest_path = self.write_stage12_artifacts(
            result=result,
            trace=trace,
        )
        validate_run_artifacts(
            config_path=config_path,
            summary_path=summary_path,
            manifest_path=manifest_path,
        )
        manifest_document = json.loads(manifest_path.read_text(encoding="utf-8"))
        missing_manifest_field = json.loads(json.dumps(manifest_document))
        del missing_manifest_field["aggregate_sha256"]
        manifest_path.write_text(json.dumps(missing_manifest_field), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "manifest schema"):
            validate_run_artifacts(
                config_path=config_path,
                summary_path=summary_path,
                manifest_path=manifest_path,
            )
        relative_trace_manifest = json.loads(json.dumps(manifest_document))
        relative_trace_manifest["trace_path"] = "experiments/data/relative.json"
        manifest_path.write_text(json.dumps(relative_trace_manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "trace_path must be an absolute"):
            validate_run_artifacts(
                config_path=config_path,
                summary_path=summary_path,
                manifest_path=manifest_path,
            )
        wrong_run_manifest = json.loads(json.dumps(manifest_document))
        wrong_run_manifest["run_id"] = "other-run"
        manifest_path.write_text(json.dumps(wrong_run_manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "run identifier"):
            validate_run_artifacts(
                config_path=config_path,
                summary_path=summary_path,
                manifest_path=manifest_path,
            )
        wrong_calibration_manifest = json.loads(json.dumps(manifest_document))
        wrong_calibration_manifest["calibration"]["steps_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(wrong_calibration_manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "calibration steps binding"):
            validate_run_artifacts(
                config_path=config_path,
                summary_path=summary_path,
                manifest_path=manifest_path,
            )
        target_config = json.loads(config_path.read_text(encoding="utf-8"))
        different_seed_config = json.loads(json.dumps(target_config))
        different_seed_config["sampling"]["seed"] = 2
        config_path.write_text(json.dumps(different_seed_config), encoding="utf-8")
        seed_manifest = json.loads(json.dumps(manifest_document))
        seed_manifest["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(seed_manifest), encoding="utf-8")
        load_frozen_manifest(manifest_path=manifest_path, config_path=config_path)
        different_profile_config = json.loads(json.dumps(different_seed_config))
        different_profile_config["sampling"]["temperature"] = 0.5
        config_path.write_text(json.dumps(different_profile_config), encoding="utf-8")
        profile_manifest = json.loads(json.dumps(seed_manifest))
        profile_manifest["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(profile_manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "sampling profile"):
            load_frozen_manifest(manifest_path=manifest_path, config_path=config_path)
        config_path.write_text(json.dumps(target_config), encoding="utf-8")
        manifest_path.write_text(json.dumps(manifest_document), encoding="utf-8")
        summary_document = json.loads(summary_path.read_text(encoding="utf-8"))
        tampered_code = json.loads(json.dumps(summary_document))
        tampered_code["provenance"]["open_loop_sha256"] = "0" * 64
        summary_path.write_text(json.dumps(tampered_code), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "execution code fingerprint"):
            validate_run_artifacts(
                config_path=config_path,
                summary_path=summary_path,
                manifest_path=manifest_path,
            )
        tampered_slo = json.loads(json.dumps(summary_document))
        tampered_slo["slo"]["ttft_slo_ms"] = 101.0
        summary_path.write_text(json.dumps(tampered_slo), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "summary slo"):
            validate_run_artifacts(
                config_path=config_path,
                summary_path=summary_path,
                manifest_path=manifest_path,
            )
        summary_path.write_text(json.dumps(summary_document), encoding="utf-8")
        raw_path = Path(summary_document["artifacts"]["requests_path"])
        raw_document = [
            json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()
        ]
        raw_document[0]["token_timestamps"][0] += 0.001
        raw_path.write_text(
            "".join(json.dumps(item) + "\n" for item in raw_document),
            encoding="utf-8",
        )
        summary_document["artifacts"]["requests_sha256"] = hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest()
        summary_path.write_text(json.dumps(summary_document), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "finished raw lifecycle"):
            validate_run_artifacts(
                config_path=config_path,
                summary_path=summary_path,
                manifest_path=manifest_path,
            )

        corrupted = dict(event)
        corrupted["predicted_free_at"] = corrupted["deadline_at"]
        with self.assertRaises(ValueError):
            validate_slack_events(
                slack_events=[corrupted],
                admission_events=result.admission_events,
                cache_events=result.cache_events,
                cache_states=result.cache_states,
                requests=[record.to_dict() for record in result.records],
                steps=result.step_events,
                trace=trace,
                admission=result.admission,
                block_size=4,
            )

    def test_feasible_head_waits_without_examining_later_request(self) -> None:
        clock = VirtualClock()
        specs = [
            RequestSpec("long-000", "long", 0.0, 8, 2, 1, "long-prefix", 4),
            RequestSpec("interactive-000", "interactive", 0.005, 8, 2, 2),
            RequestSpec("interactive-001", "interactive", 0.005, 8, 2, 3),
        ]
        requests = prepare_requests(specs, token_id_upper_bound=10000)
        engine = SlackFakeEngine(clock)
        result = run_open_loop(
            engine,
            requests,
            temperature=1.0,
            ignore_eos=True,
            measurement_start=0.0,
            measurement_end=1.0,
            ttft_slo_ms=100.0,
            itl_slo_ms=100.0,
            max_run_seconds=1.0,
            clock=clock,
            sleep=clock.sleep,
            synchronize=lambda: None,
            admission=AdmissionConfig(
                policy="slack_aware_prefix_fifo",
                total_kv_blocks=5,
                kvcache_block_size=4,
                max_queue_wait_seconds=0.10,
                observe_prefix_cache=True,
                eta_prefill_seconds=0.02,
                eta_decode_seconds_per_token=0.005,
                eta_safety_margin_seconds=0.01,
            ),
        )

        self.assertEqual(result.slack_events, [])
        self.assertTrue(
            all(record.terminal_state.value == "Finished" for record in result.records)
        )
        third_prompt = tuple(requests[2].prompt_token_ids)
        third_checks = [
            step_count
            for prompt, step_count in engine.preview_history
            if prompt == third_prompt
        ]
        self.assertTrue(third_checks)
        self.assertGreaterEqual(min(third_checks), 3)

    def test_unknown_eta_keeps_fifo_wait_in_the_run_loop(self) -> None:
        clock = VirtualClock()
        specs = [
            RequestSpec("long-000", "long", 0.0, 8, 2, 1, "long-prefix", 4),
            RequestSpec("interactive-000", "interactive", 0.005, 8, 2, 2),
            RequestSpec("interactive-001", "interactive", 0.005, 8, 2, 3),
        ]
        requests = prepare_requests(specs, token_id_upper_bound=10000)
        engine = SlackFakeEngine(clock)
        with patch.object(SlackEstimator, "earliest_capacity_time", return_value=None):
            result = run_open_loop(
                engine,
                requests,
                temperature=1.0,
                ignore_eos=True,
                measurement_start=0.0,
                measurement_end=1.0,
                ttft_slo_ms=100.0,
                itl_slo_ms=100.0,
                max_run_seconds=1.0,
                clock=clock,
                sleep=clock.sleep,
                synchronize=lambda: None,
                admission=AdmissionConfig(
                    policy="slack_aware_prefix_fifo",
                    total_kv_blocks=5,
                    kvcache_block_size=4,
                    max_queue_wait_seconds=0.10,
                    observe_prefix_cache=True,
                    eta_prefill_seconds=0.02,
                    eta_decode_seconds_per_token=0.04,
                    eta_safety_margin_seconds=0.01,
                ),
            )

        self.assertEqual(result.slack_events, [])
        self.assertEqual(result.admission["final_reserved_blocks"], 0)
        self.assertTrue(
            all(record.terminal_state.value == "Finished" for record in result.records)
        )
        third_prompt = tuple(requests[2].prompt_token_ids)
        third_checks = [
            step_count
            for prompt, step_count in engine.preview_history
            if prompt == third_prompt
        ]
        self.assertTrue(third_checks)
        self.assertGreaterEqual(min(third_checks), 3)

    def test_engine_add_error_releases_slack_reservation(self) -> None:
        class AddErrorEngine(SlackFakeEngine):
            def add_request(
                self, prompt: list[int], sampling_params: SamplingParams
            ) -> int:
                if self.next_seq_id == 2:
                    raise RuntimeError("injected add failure")
                return super().add_request(prompt, sampling_params)

        clock = VirtualClock()
        requests = prepare_requests(
            [
                RequestSpec("long-000", "long", 0.0, 8, 2, 1, "long-prefix", 4),
                RequestSpec("interactive-000", "interactive", 0.005, 8, 2, 2),
            ],
            token_id_upper_bound=10000,
        )
        result = run_open_loop(
            AddErrorEngine(clock),
            requests,
            temperature=1.0,
            ignore_eos=True,
            measurement_start=0.0,
            measurement_end=1.0,
            ttft_slo_ms=100.0,
            itl_slo_ms=100.0,
            max_run_seconds=1.0,
            clock=clock,
            sleep=clock.sleep,
            synchronize=lambda: None,
            admission=AdmissionConfig(
                policy="slack_aware_prefix_fifo",
                total_kv_blocks=5,
                kvcache_block_size=4,
                max_queue_wait_seconds=0.10,
                observe_prefix_cache=True,
                eta_prefill_seconds=0.02,
                eta_decode_seconds_per_token=0.005,
                eta_safety_margin_seconds=0.01,
            ),
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.admission["final_reserved_blocks"], 0)
        self.assertFalse(result.slack_events)
        self.assertEqual(
            [event["action"] for event in result.admission_events if event["request_id"] == "interactive-000"],
            ["admitted", "released"],
        )
        self.assertEqual(
            result.admission_events[-1]["reason"], "engine_add_error"
        )


if __name__ == "__main__":
    unittest.main()
