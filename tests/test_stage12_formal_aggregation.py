from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from experiments.aggregate_stage12_formal import (
    FORMAL_MANIFEST,
    RESULTS,
    _normalized_pair,
    aggregate,
    aggregation_fingerprint,
    load_formal_plan,
    validate_prefix_run,
    validate_matched_configs,
)
from experiments.aggregate_stage12 import (
    REPO_ROOT,
    aggregate_source_is_verified,
    execution_code_fingerprint,
    resolve_recorded_manifest_path,
)
from experiments.benchmark.lifecycle import RequestRecord, TerminalState, summarize_requests


class Stage12FormalPlanTests(unittest.TestCase):
    def write_prefix_fixture(self) -> dict:
        """Write a minimal but fully replayable Prefix control artifact set."""
        repository = Path(__file__).resolve().parents[1]
        run_id = f"unit_stage12_prefix_{uuid.uuid4().hex}"
        config_directory = tempfile.TemporaryDirectory(dir=repository / "experiments/configs")
        trace_directory = tempfile.TemporaryDirectory(dir=repository / "experiments/data")
        self.addCleanup(config_directory.cleanup)
        self.addCleanup(trace_directory.cleanup)
        config_path = Path(config_directory.name) / f"{run_id}.json"
        trace_path = Path(trace_directory.name) / f"{run_id}.json"
        trace = {
            "schema_version": 2,
            "description": "minimal prefix control fixture",
            "time_unit": "seconds",
            "requests": [{
                "request_id": "interactive-000", "request_class": "interactive",
                "arrival_time": 0.0, "prompt_length": 4, "max_output_tokens": 1,
                "seed": 1, "prefix_group": None, "shared_prefix_length": 0,
            }],
        }
        trace_path.write_text(json.dumps(trace), encoding="utf-8")
        config = {
            "schema_version": 1, "upstream_commit": "0" * 40,
            "model": {"path": "/opt/unit", "repo_id": "unit/model", "revision": "unit", "sha256": {"model.safetensors": "unit"}},
            "engine": {"enforce_eager": True, "tensor_parallel_size": 1, "max_model_len": 16, "max_num_batched_tokens": 16, "max_num_seqs": 1, "gpu_memory_utilization": 0.8, "kvcache_block_size": 4, "scheduler_policy": "prefill_first", "decode_token_budget": 0, "decode_step_guard": 0, "num_kvcache_blocks": 6, "cuda_event_timing": True},
            "sampling": {"temperature": 1.0, "ignore_eos": True, "seed": 1},
            "workload": {"trace_path": str(trace_path), "token_id_upper_bound": 10000},
            "slo": {"ttft_slo_ms": 100.0, "itl_slo_ms": 100.0, "require_itl": True},
            "measurement": {"start_seconds": 0.0, "end_seconds": 1.0, "max_run_seconds": 1.0},
            "admission": {"policy": "prefix_aware_fifo", "max_queue_wait_seconds": 0.2, "observe_prefix_cache": True, "eta_prefill_seconds": 0.0, "eta_decode_seconds_per_token": 0.0, "eta_safety_margin_seconds": 0.0},
            "output": {"directory": "experiments/results", "run_id": run_id},
        }
        config_path.write_text(json.dumps(config), encoding="utf-8")
        record = RequestRecord("interactive-000", "interactive", 0.0)
        record.mark_admitted(0.0)
        record.record_token(0.01)
        record.mark_terminal(TerminalState.FINISHED, 0.01)
        requests = [record.to_dict()]
        steps = [{"step_index": 0, "started_at": 0.0, "finished_at": 0.01, "duration_ms": 10.0, "phase": "decode", "num_scheduled_tokens": 1, "events": [{"request_id": "interactive-000", "seq_id": 1, "phase": "decode", "num_scheduled_tokens": 1, "emitted_token": True, "finished": True}]}]
        admission = [
            {"schema_version": 2, "event_index": 0, "action": "admitted", "request_id": "interactive-000", "arrival_at": 0.0, "observed_at": 0.0, "queue_wait_ms": 0.0, "required_blocks": 1, "reservation_blocks": 1, "reserved_blocks_after": 1, "reason": None},
            {"schema_version": 2, "event_index": 1, "action": "released", "request_id": "interactive-000", "arrival_at": 0.0, "observed_at": 0.01, "queue_wait_ms": None, "required_blocks": 1, "reservation_blocks": 1, "reserved_blocks_after": 0, "reason": None},
        ]
        cache_template = {"schema_version": 2, "request_id": "interactive-000", "full_required_blocks": 1, "matched_prefix_blocks": 0, "active_shared_blocks": 0, "inactive_cached_blocks": 0, "matched_prefix_tokens": 0, "incremental_reservation_blocks": 1, "matched_block_ids": [], "active_block_ids": [], "inactive_block_ids": [], "cache_state_index": 0, "reason": None}
        cache = [
            {**cache_template, "event_index": 0, "action": "admitted", "observed_at": 0.0, "reserved_blocks_after": 1},
            {**cache_template, "event_index": 1, "action": "released", "observed_at": 0.01, "reserved_blocks_after": 0},
        ]
        documents = {"requests": requests, "steps": steps, "admission_events": admission, "cache_events": cache, "cache_states": [{"schema_version": 1, "state_index": 0, "observed_at": 0.0, "blocks": []}]}
        suffixes = {"requests": "requests", "steps": "steps", "admission_events": "admission", "cache_events": "cache", "cache_states": "cache_states"}
        artifacts = {}
        paths = {}
        for name, document in documents.items():
            path = RESULTS / f"{run_id}.{suffixes[name]}.jsonl"
            path.write_text("".join(json.dumps(item) + "\n" for item in document), encoding="utf-8")
            self.addCleanup(path.unlink, missing_ok=True)
            paths[name] = path
            artifacts[f"{name}_path"] = str(path)
            artifacts[f"{name}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        metrics = summarize_requests([record], measurement_start=0.0, measurement_end=1.0, ttft_slo_ms=100.0, itl_slo_ms=100.0, require_itl=True)
        metrics["runtime"] = {"elapsed_seconds": 0.01, "steps": 1, "timed_out": False}
        summary = {
            "status": "passed", "error": None, "upstream_commit": config["upstream_commit"],
            "provenance": {"config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(), "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(), **execution_code_fingerprint()},
            "model": {**config["model"], "verified_sha256": config["model"]["sha256"]},
            "runtime": {"cuda_available": True}, "engine": config["engine"], "sampling": config["sampling"], "slo": config["slo"], "measurement": config["measurement"],
            "admission": {"schema_version": 2, "policy": config["admission"]["policy"], "max_queue_wait_seconds": config["admission"]["max_queue_wait_seconds"], "observe_prefix_cache": config["admission"]["observe_prefix_cache"], "total_kv_blocks": 6, "admitted_requests": 1, "rejected_requests": 0, "max_observed_queue_wait_ms": 0.0, "peak_reserved_blocks": 1, "final_reserved_blocks": 0},
            "summary": metrics,
            "pressure": {"schema_version": 1, "observation_active": True, "observation_complete": True, "oom_detected": False, "collection_error": None, "rejected_requests": 0, "admission_rejection_supported": True, "kv_cache": {"total_blocks": 6, "final_used_blocks": 0, "final_free_blocks": 6, "peak_used_blocks": 1}},
            "artifacts": artifacts,
        }
        summary_path = RESULTS / f"{run_id}.summary.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        self.addCleanup(summary_path.unlink, missing_ok=True)
        return {"config_path": config_path, "trace_path": trace_path, "config_sha256": summary["provenance"]["config_sha256"], "trace_sha256": summary["provenance"]["trace_sha256"], "cache_path": paths["cache_events"], "summary_path": summary_path}

    def test_frozen_r3_plan_has_exactly_six_hash_bound_cells(self) -> None:
        plan = load_formal_plan()
        self.assertEqual(len(plan), 6)
        validate_matched_configs(plan)

    def test_formal_plan_relocates_paths_from_another_checkout(self) -> None:
        document = json.loads(FORMAL_MANIFEST.read_text(encoding="utf-8"))
        for cell in document["cells"]:
            for field in ("config_path", "trace_path", "manifest_path"):
                if field not in cell:
                    continue
                marker = "/experiments/"
                suffix = cell[field].replace("\\", "/").split(marker, 1)[1]
                cell[field] = f"/different/checkout/experiments/{suffix}"
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            encoding="utf-8",
            delete=False,
        ) as handle:
            json.dump(document, handle)
            path = Path(handle.name)
        self.addCleanup(path.unlink, missing_ok=True)

        plan = load_formal_plan(path)

        self.assertEqual(len(plan), 6)
        validate_matched_configs(plan)

    def test_manifest_resolver_accepts_windows_checkout_path(self) -> None:
        config = next((REPO_ROOT / "experiments/configs").glob("*.json"))
        relative = config.relative_to(REPO_ROOT).as_posix().replace("/", "\\")

        resolved = resolve_recorded_manifest_path(
            rf"C:\old\checkout\{relative}",
            field="config_path",
            allowed_directory=REPO_ROOT / "experiments/configs",
        )

        self.assertEqual(resolved, config.resolve())

    @unittest.skip(
        "requires legacy repository Git objects; clean snapshot uses PROVENANCE.md"
    )
    def test_historical_aggregate_source_is_verified_from_git(self) -> None:
        manifest = json.loads(
            (
                REPO_ROOT
                / "experiments/configs/stage12_formal_r3/"
                "stage12_formal_r3_slack_seed1.manifest.json"
            ).read_text(encoding="utf-8")
        )

        self.assertTrue(
            aggregate_source_is_verified(manifest["aggregate_sha256"])
        )
        self.assertFalse(aggregate_source_is_verified("0" * 64))

    @unittest.skip(
        "requires legacy Stage 12 directory binding; frozen evidence is not rewritten"
    )
    def test_real_r3_slack_manifest_loads_after_path_relocation(self) -> None:
        cell = load_formal_plan()[("slack", 1)]
        from experiments.aggregate_stage12 import load_frozen_manifest

        frozen = load_frozen_manifest(
            manifest_path=cell["manifest_path"],
            config_path=cell["config_path"],
            replay=True,
        )

        self.assertEqual(frozen["config_sha256"], cell["config_sha256"])

    @unittest.skip(
        "requires legacy Stage 12 result layout; clean snapshot preserves final evidence"
    )
    def test_real_r3_formal_aggregate_replays_all_cells(self) -> None:
        value = aggregate()
        frozen = json.loads(
            (
                REPO_ROOT
                / "experiments/configs/stage12_formal_r3/"
                "stage12_formal_r3_slack_seed1.manifest.json"
            ).read_text(encoding="utf-8")
        )

        self.assertEqual(len(value["runs"]), 6)
        self.assertEqual(value["execution_code"], frozen["execution_code"])
        self.assertNotEqual(value["execution_code"], execution_code_fingerprint())

    def test_aggregation_fingerprint_binds_all_replay_modules(self) -> None:
        fingerprint = aggregation_fingerprint()
        self.assertEqual(
            set(fingerprint),
            {"aggregate_stage12_formal_sha256", "aggregate_stage12_sha256", "aggregate_stage11_sha256"},
        )
        self.assertTrue(all(len(value) == 64 for value in fingerprint.values()))

    def test_missing_required_cell_is_rejected(self) -> None:
        document = json.loads(FORMAL_MANIFEST.read_text(encoding="utf-8"))
        document["cells"].pop()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8", delete=False) as handle:
            json.dump(document, handle)
            path = Path(handle.name)
        self.addCleanup(path.unlink, missing_ok=True)
        with self.assertRaisesRegex(ValueError, "required cells"):
            load_formal_plan(path)

    def test_nonpolicy_difference_survives_pair_normalization(self) -> None:
        plan = load_formal_plan()
        prefix = json.loads(plan[("prefix", 1)]["config_path"].read_text(encoding="utf-8"))
        slack = json.loads(plan[("slack", 1)]["config_path"].read_text(encoding="utf-8"))
        self.assertEqual(_normalized_pair(prefix), _normalized_pair(slack))
        broken = copy.deepcopy(slack)
        broken["engine"]["max_num_batched_tokens"] += 1
        self.assertNotEqual(_normalized_pair(prefix), _normalized_pair(broken))

    def test_prefix_raw_cache_and_kv_tampering_fail_closed(self) -> None:
        fixture = self.write_prefix_fixture()
        validate_prefix_run(cell=fixture)
        cache = fixture["cache_path"]
        cache.write_text(cache.read_text(encoding="utf-8").replace('"full_required_blocks": 1', '"full_required_blocks": 2', 1), encoding="utf-8")
        summary = json.loads(fixture["summary_path"].read_text(encoding="utf-8"))
        summary["artifacts"]["cache_events_sha256"] = hashlib.sha256(cache.read_bytes()).hexdigest()
        fixture["summary_path"].write_text(json.dumps(summary), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "cache full footprint"):
            validate_prefix_run(cell=fixture)
        fixture = self.write_prefix_fixture()
        fixture["trace_path"].write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "provenance"):
            validate_prefix_run(cell=fixture)
        fixture = self.write_prefix_fixture()
        summary = json.loads(fixture["summary_path"].read_text(encoding="utf-8"))
        summary["pressure"]["kv_cache"]["final_used_blocks"] = 1
        fixture["summary_path"].write_text(json.dumps(summary), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "completion gate"):
            validate_prefix_run(cell=fixture)

    def test_aggregate_fails_closed_when_any_slack_cell_fails(self) -> None:
        plan = load_formal_plan()
        stub_summary = {
            "summary": {"terminal_counts": {"Rejected": 0}, "interactive": {"slo_goodput_rps": 0.0, "ttft_ms": {"p99": 1.0}, "itl_ms": {"p99": 1.0}}, "long": {"token_goodput_tps": 0.0}, "output_throughput_tps": 0.0, "runtime": {"elapsed_seconds": 1.0}},
            "admission": {},
        }
        with patch("experiments.aggregate_stage12_formal.load_formal_plan", return_value=plan), patch("experiments.aggregate_stage12_formal.validate_matched_configs"), patch("experiments.aggregate_stage12_formal.validate_prefix_run", return_value=stub_summary), patch("experiments.aggregate_stage12_formal.read_jsonl", return_value=[]), patch("experiments.aggregate_stage12_formal.validate_run_artifacts", side_effect=ValueError("slack replay differs")):
            with self.assertRaisesRegex(ValueError, "slack replay"):
                aggregate()

    def test_formal_aggregate_rejects_an_unapproved_manifest_path(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".json") as handle:
            with self.assertRaisesRegex(ValueError, "approved r3 manifest"):
                aggregate(Path(handle.name))


if __name__ == "__main__":
    unittest.main()
