from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

from experiments.benchmark.open_loop import (
    AdmissionConfig,
    PreparedRequest,
    RequestSpec,
    load_trace,
    prepare_requests,
    run_open_loop,
)
from nanovllm import SamplingParams
from nanovllm.engine.block_manager import BlockManager, PrefixCachePreview
from nanovllm.engine.llm_engine import EngineStepResult, SequenceStepEvent
from nanovllm.engine.sequence import Sequence
from experiments.run_open_loop import (
    load_config,
    validate_kv_request_feasibility,
)


class VirtualClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def sleep(self, duration: float) -> None:
        self.value += duration


class PrefixAwareFakeEngine:
    def __init__(self, clock: VirtualClock) -> None:
        self.clock = clock
        self.next_seq_id = 1
        self.active: set[int] = set()
        self.preview_calls = 0
        self.admissions: list[float] = []

    def preview_prefix_cache(self, prompt: list[int], max_tokens: int):
        self.preview_calls += 1
        return PrefixCachePreview(
            full_required_blocks=2,
            matched_prefix_blocks=1 if self.preview_calls > 1 else 0,
            active_shared_blocks=1 if self.preview_calls > 1 else 0,
            inactive_cached_blocks=0,
            matched_prefix_tokens=4 if self.preview_calls > 1 else 0,
            incremental_reservation_blocks=(
                1 if self.preview_calls > 1 else 2
            ),
            matched_block_ids=(1,) if self.preview_calls > 1 else (),
            active_block_ids=(1,) if self.preview_calls > 1 else (),
            inactive_block_ids=(),
        )

    def cache_state_snapshot(self) -> list[dict]:
        return []

    def add_request(self, prompt, sampling_params) -> int:
        seq_id = self.next_seq_id
        self.next_seq_id += 1
        self.active.add(seq_id)
        self.admissions.append(self.clock())
        return seq_id

    def is_finished(self) -> bool:
        return not self.active

    def step_with_events(self) -> EngineStepResult:
        self.clock.value += 0.01
        active = sorted(self.active)
        self.active.clear()
        return EngineStepResult(
            outputs=[(seq_id, [100 + seq_id]) for seq_id in active],
            phase="decode",
            num_scheduled_tokens=len(active),
            signed_token_count=-len(active),
            events=[
                SequenceStepEvent(
                    seq_id=seq_id,
                    phase="decode",
                    num_scheduled_tokens=1,
                    emitted_token_id=100 + seq_id,
                    finished=True,
                )
                for seq_id in active
            ],
        )


class NoHitPrefixEngine(PrefixAwareFakeEngine):
    def preview_prefix_cache(self, prompt: list[int], max_tokens: int):
        return PrefixCachePreview(
            full_required_blocks=2,
            matched_prefix_blocks=0,
            active_shared_blocks=0,
            inactive_cached_blocks=0,
            matched_prefix_tokens=0,
            incremental_reservation_blocks=2,
        )


class PrefixAwareAdmissionTests(unittest.TestCase):
    def write_json(self, value: dict) -> Path:
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

    def request(self, request_id: str) -> PreparedRequest:
        return PreparedRequest(
            spec=RequestSpec(
                request_id=request_id,
                request_class="interactive",
                arrival_time=0.0,
                prompt_length=4,
                max_output_tokens=4,
                seed=1,
            ),
            prompt_token_ids=[1, 2, 3, 4],
        )

    def test_block_manager_preview_distinguishes_active_and_inactive_hits(self) -> None:
        original_block_size = Sequence.block_size
        self.addCleanup(setattr, Sequence, "block_size", original_block_size)
        Sequence.block_size = 4
        manager = BlockManager(num_blocks=4, block_size=4)
        sequence = Sequence(
            [1, 2, 3, 4, 5, 6, 7, 8],
            SamplingParams(max_tokens=2),
        )
        manager.allocate(sequence, num_cached_blocks=0)
        sequence.num_scheduled_tokens = 8
        manager.hash_blocks(sequence)

        active = manager.preview_prefix_cache(
            [1, 2, 3, 4, 5, 6, 7, 8],
            max_tokens=2,
        )
        # The final prompt block is excluded exactly as in can_allocate(),
        # because Decode can append into it.
        self.assertEqual(active.matched_prefix_blocks, 1)
        self.assertEqual(active.active_shared_blocks, 1)
        self.assertEqual(active.inactive_cached_blocks, 0)
        self.assertEqual(active.incremental_reservation_blocks, 2)
        candidate = Sequence(
            [1, 2, 3, 4, 5, 6, 7, 8],
            SamplingParams(max_tokens=2),
        )
        self.assertEqual(
            manager.can_allocate(candidate), active.matched_prefix_blocks
        )
        active_state = {
            item["block_id"]: item for item in manager.cache_state_snapshot()
        }
        self.assertTrue(active_state[active.active_block_ids[0]]["used"])
        self.assertGreater(active_state[active.active_block_ids[0]]["ref_count"], 0)

        manager.deallocate(sequence)
        inactive = manager.preview_prefix_cache(
            [1, 2, 3, 4, 5, 6, 7, 8],
            max_tokens=2,
        )
        self.assertEqual(inactive.matched_prefix_blocks, 1)
        self.assertEqual(inactive.active_shared_blocks, 0)
        self.assertEqual(inactive.inactive_cached_blocks, 1)
        self.assertEqual(inactive.incremental_reservation_blocks, 3)
        inactive_state = {
            item["block_id"]: item for item in manager.cache_state_snapshot()
        }
        self.assertFalse(inactive_state[inactive.inactive_block_ids[0]]["used"])
        self.assertEqual(inactive_state[inactive.inactive_block_ids[0]]["ref_count"], 0)

    def test_prefix_aware_fifo_reserves_only_active_shared_blocks(self) -> None:
        clock = VirtualClock()
        result = run_open_loop(
            PrefixAwareFakeEngine(clock),
            [self.request("interactive-000"), self.request("interactive-001")],
            temperature=1.0,
            ignore_eos=True,
            measurement_start=0.0,
            measurement_end=1.0,
            ttft_slo_ms=20.0,
            itl_slo_ms=20.0,
            max_run_seconds=1.0,
            clock=clock,
            sleep=clock.sleep,
            synchronize=lambda: None,
            admission=AdmissionConfig(
                policy="prefix_aware_fifo",
                total_kv_blocks=3,
                kvcache_block_size=4,
                max_queue_wait_seconds=1.0,
                observe_prefix_cache=True,
            ),
        )

        self.assertEqual(
            [record.terminal_state.value for record in result.records],
            ["Finished", "Finished"],
        )
        self.assertEqual(result.admission["peak_reserved_blocks"], 3)
        self.assertEqual(
            [event["reservation_blocks"] for event in result.admission_events],
            [2, 1, 2, 1],
        )
        self.assertEqual(
            [event["action"] for event in result.cache_events],
            ["admitted", "admitted", "released", "released"],
        )

    def test_waiting_prefix_request_has_no_orphan_preview_event(self) -> None:
        clock = VirtualClock()
        result = run_open_loop(
            NoHitPrefixEngine(clock),
            [self.request("interactive-000"), self.request("interactive-001")],
            temperature=1.0,
            ignore_eos=True,
            measurement_start=0.0,
            measurement_end=1.0,
            ttft_slo_ms=20.0,
            itl_slo_ms=20.0,
            max_run_seconds=1.0,
            clock=clock,
            sleep=clock.sleep,
            synchronize=lambda: None,
            admission=AdmissionConfig(
                policy="prefix_aware_fifo",
                total_kv_blocks=2,
                kvcache_block_size=4,
                max_queue_wait_seconds=1.0,
                observe_prefix_cache=True,
            ),
        )

        self.assertEqual(
            [event["action"] for event in result.cache_events],
            ["admitted", "released", "admitted", "released"],
        )
        self.assertEqual(
            [event["request_id"] for event in result.cache_events],
            ["interactive-000", "interactive-000", "interactive-001", "interactive-001"],
        )

    def test_active_prefix_share_changes_admission_decision(self) -> None:
        fifo_clock = VirtualClock()
        fifo_engine = PrefixAwareFakeEngine(fifo_clock)
        fifo_result = run_open_loop(
            fifo_engine,
            [self.request("interactive-000"), self.request("interactive-001")],
            temperature=1.0,
            ignore_eos=True,
            measurement_start=0.0,
            measurement_end=1.0,
            ttft_slo_ms=20.0,
            itl_slo_ms=20.0,
            max_run_seconds=1.0,
            clock=fifo_clock,
            sleep=fifo_clock.sleep,
            synchronize=lambda: None,
            admission=AdmissionConfig(
                policy="kv_aware_fifo",
                total_kv_blocks=3,
                kvcache_block_size=4,
                max_queue_wait_seconds=1.0,
            ),
        )
        prefix_clock = VirtualClock()
        prefix_engine = PrefixAwareFakeEngine(prefix_clock)
        prefix_result = run_open_loop(
            prefix_engine,
            [self.request("interactive-000"), self.request("interactive-001")],
            temperature=1.0,
            ignore_eos=True,
            measurement_start=0.0,
            measurement_end=1.0,
            ttft_slo_ms=20.0,
            itl_slo_ms=20.0,
            max_run_seconds=1.0,
            clock=prefix_clock,
            sleep=prefix_clock.sleep,
            synchronize=lambda: None,
            admission=AdmissionConfig(
                policy="prefix_aware_fifo",
                total_kv_blocks=3,
                kvcache_block_size=4,
                max_queue_wait_seconds=1.0,
                observe_prefix_cache=True,
            ),
        )

        self.assertEqual(fifo_engine.admissions, [0.0, 0.01])
        self.assertEqual(prefix_engine.admissions, [0.0, 0.0])
        self.assertEqual(fifo_result.admission["schema_version"], 1)
        self.assertEqual(prefix_result.admission["schema_version"], 2)
        self.assertTrue(prefix_result.admission["observe_prefix_cache"])
        self.assertTrue(
            all("reservation_blocks" not in event for event in fifo_result.admission_events)
        )
        self.assertTrue(
            all("reservation_blocks" in event for event in prefix_result.admission_events)
        )

    def test_runner_loads_prefix_aware_admission_only_with_observation(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        config = json.loads(
            (
                repo_root
                / "experiments/configs/stage9_admission_smoke_kv4_seed1.json"
            ).read_text(encoding="utf-8")
        )
        config["output"]["run_id"] = "unit_test_prefix_aware_loader"
        config["admission"] = {
            "policy": "prefix_aware_fifo",
            "max_queue_wait_seconds": 0.25,
            "observe_prefix_cache": True,
        }

        loaded = load_config(self.write_json(config))

        self.assertTrue(loaded["admission"]["observe_prefix_cache"])
        self.assertEqual(loaded["admission"]["policy"], "prefix_aware_fifo")

    def test_stage11_smoke_config_is_runner_loadable(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        prefix_document = json.loads(
            (
                repo_root
                / "experiments/configs/stage11_prefix_admission_smoke_kv6_seed1.json"
            ).read_text(encoding="utf-8")
        )
        fifo_document = json.loads(
            (
                repo_root
                / "experiments/configs/stage11_kv_admission_smoke_kv6_seed1.json"
            ).read_text(encoding="utf-8")
        )
        prefix_document["output"]["run_id"] = "unit_test_stage11_prefix_loader"
        fifo_document["output"]["run_id"] = "unit_test_stage11_fifo_loader"
        prefix = load_config(self.write_json(prefix_document))
        fifo = load_config(self.write_json(fifo_document))

        self.assertEqual(prefix["engine"]["num_kvcache_blocks"], 6)
        self.assertEqual(prefix["admission"]["policy"], "prefix_aware_fifo")
        self.assertTrue(prefix["admission"]["observe_prefix_cache"])
        normalized_prefix = dict(prefix)
        normalized_fifo = dict(fifo)
        normalized_prefix["admission"] = "<policy>"
        normalized_fifo["admission"] = "<policy>"
        normalized_prefix["output"] = "<output>"
        normalized_fifo["output"] = "<output>"
        self.assertEqual(normalized_prefix, normalized_fifo)

    def test_trace_v2_generates_stable_shared_prefixes(self) -> None:
        trace = {
            "schema_version": 2,
            "description": "shared-prefix test trace",
            "time_unit": "seconds",
            "requests": [
                {
                    "request_id": "interactive-000",
                    "request_class": "interactive",
                    "arrival_time": 0.0,
                    "prompt_length": 8,
                    "max_output_tokens": 2,
                    "seed": 11,
                    "prefix_group": "system-prompt-a",
                    "shared_prefix_length": 4,
                },
                {
                    "request_id": "interactive-001",
                    "request_class": "interactive",
                    "arrival_time": 0.1,
                    "prompt_length": 8,
                    "max_output_tokens": 2,
                    "seed": 12,
                    "prefix_group": "system-prompt-a",
                    "shared_prefix_length": 4,
                },
            ],
        }

        prepared = prepare_requests(
            load_trace(self.write_json(trace)),
            token_id_upper_bound=10000,
        )

        self.assertEqual(
            prepared[0].prompt_token_ids[:4],
            prepared[1].prompt_token_ids[:4],
        )
        self.assertNotEqual(
            prepared[0].prompt_token_ids[4:],
            prepared[1].prompt_token_ids[4:],
        )

    def test_trace_v2_rejects_prefix_group_without_shared_length(self) -> None:
        trace = {
            "schema_version": 2,
            "description": "invalid shared-prefix trace",
            "time_unit": "seconds",
            "requests": [
                {
                    "request_id": "interactive-000",
                    "request_class": "interactive",
                    "arrival_time": 0.0,
                    "prompt_length": 8,
                    "max_output_tokens": 2,
                    "seed": 11,
                    "prefix_group": "system-prompt-a",
                    "shared_prefix_length": 0,
                }
            ],
        }

        with self.assertRaisesRegex(ValueError, "shared_prefix_length"):
            load_trace(self.write_json(trace))

    def test_prefix_aware_feasibility_requires_block_aligned_prefix(self) -> None:
        spec = RequestSpec(
            request_id="interactive-000",
            request_class="interactive",
            arrival_time=0.0,
            prompt_length=12,
            max_output_tokens=2,
            seed=11,
            prefix_group="system-prompt-a",
            shared_prefix_length=6,
        )
        engine = {"kvcache_block_size": 4, "num_kvcache_blocks": 4}

        with self.assertRaisesRegex(ValueError, "block-aligned"):
            validate_kv_request_feasibility(
                [spec],
                engine,
                admission={"policy": "prefix_aware_fifo"},
            )

        with self.assertRaisesRegex(ValueError, "block-aligned"):
            validate_kv_request_feasibility(
                [spec],
                engine,
                admission={"policy": "slack_aware_prefix_fifo"},
            )

        validate_kv_request_feasibility(
            [spec],
            engine,
            admission={"policy": "kv_aware_fifo"},
        )


if __name__ == "__main__":
    unittest.main()
