from __future__ import annotations

import unittest

from experiments.benchmark.open_loop import (
    AdmissionConfig,
    PreparedRequest,
    RequestSpec,
    run_open_loop,
)
from experiments.run_open_loop import fail_result
from nanovllm.engine.llm_engine import (
    EngineStepResult,
    SequenceStepEvent,
)


class VirtualClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def sleep(self, duration: float) -> None:
        self.value += duration


class FakeEngine:
    def __init__(self, clock: VirtualClock) -> None:
        self.clock = clock
        self.next_seq_id = 10
        self.active: dict[int, int] = {}
        self.admissions: list[tuple[float, list[int]]] = []
        self.slo_contexts: list[tuple[int, float, float]] = []

    def add_request(self, prompt, sampling_params) -> int:
        seq_id = self.next_seq_id
        self.next_seq_id += 1
        self.active[seq_id] = sampling_params.max_tokens
        self.admissions.append((self.clock(), prompt))
        return seq_id

    def configure_request_slo(
        self,
        seq_id: int,
        *,
        ttft_deadline_at: float,
        itl_slo_s: float,
    ) -> None:
        self.slo_contexts.append(
            (seq_id, ttft_deadline_at, itl_slo_s)
        )

    def is_finished(self) -> bool:
        return not self.active

    def step_with_events(self) -> EngineStepResult:
        self.clock.value += 0.01
        events = []
        outputs = []
        for seq_id in list(self.active):
            self.active[seq_id] -= 1
            finished = self.active[seq_id] == 0
            events.append(
                SequenceStepEvent(
                    seq_id=seq_id,
                    phase="decode",
                    num_scheduled_tokens=1,
                    emitted_token_id=100 + seq_id,
                    finished=finished,
                )
            )
            if finished:
                outputs.append((seq_id, [100 + seq_id]))
                del self.active[seq_id]
        return EngineStepResult(
            outputs=outputs,
            phase="decode",
            num_scheduled_tokens=len(events),
            signed_token_count=-len(events),
            events=events,
            phase_timings_ms={
                "schedule_cpu_ms": 0.1,
                "model_runner_cuda_ms": 0.2,
                "postprocess_cpu_ms": 0.3,
                "step_wall_ms": 0.6,
            },
        )


class DelayedAdmissionEngine(FakeEngine):
    def add_request(self, prompt, sampling_params) -> int:
        self.clock.value += 0.005
        return super().add_request(prompt, sampling_params)


class FailingAdmissionEngine:
    def add_request(self, prompt, sampling_params) -> int:
        raise RuntimeError("synthetic admission failure")

    def is_finished(self) -> bool:
        return True

    def step_with_events(self) -> EngineStepResult:
        raise AssertionError("step_with_events must not be called")


class DuplicateSequenceEngine:
    def add_request(self, prompt, sampling_params) -> int:
        return 1

    def is_finished(self) -> bool:
        return True

    def step_with_events(self) -> EngineStepResult:
        raise AssertionError("step_with_events must not be called")


class SlowEngine:
    def __init__(self, clock: VirtualClock) -> None:
        self.clock = clock
        self.active = False

    def add_request(self, prompt, sampling_params) -> int:
        self.active = True
        return 1

    def is_finished(self) -> bool:
        return not self.active

    def step_with_events(self) -> EngineStepResult:
        self.clock.value += 2.0
        return EngineStepResult(
            outputs=[],
            phase="prefill",
            num_scheduled_tokens=1,
            signed_token_count=1,
            events=[
                SequenceStepEvent(
                    seq_id=1,
                    phase="prefill",
                    num_scheduled_tokens=1,
                    emitted_token_id=None,
                    finished=False,
                )
            ],
        )


class OpenLoopRunnerTests(unittest.TestCase):
    def admission_request(
        self,
        request_id: str,
        *,
        max_tokens: int,
    ) -> PreparedRequest:
        return PreparedRequest(
            spec=RequestSpec(
                request_id=request_id,
                request_class="interactive",
                arrival_time=0.0,
                prompt_length=3,
                max_output_tokens=max_tokens,
                seed=1,
            ),
            prompt_token_ids=[1, 2, 3],
        )

    def test_kv_aware_fifo_releases_reservation_before_next_admission(self) -> None:
        clock = VirtualClock()
        engine = FakeEngine(clock)
        result = run_open_loop(
            engine,
            [
                self.admission_request("interactive-000", max_tokens=1),
                self.admission_request("interactive-001", max_tokens=1),
            ],
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
                policy="kv_aware_fifo",
                total_kv_blocks=1,
                kvcache_block_size=4,
                max_queue_wait_seconds=1.0,
            ),
        )

        self.assertEqual([time for time, _ in engine.admissions], [0.0, 0.01])
        self.assertEqual(
            [record.terminal_state.value for record in result.records],
            ["Finished", "Finished"],
        )
        self.assertEqual(result.admission["peak_reserved_blocks"], 1)
        self.assertEqual(result.admission["final_reserved_blocks"], 0)
        self.assertEqual(
            [event["action"] for event in result.admission_events],
            ["admitted", "released", "admitted", "released"],
        )
        self.assertEqual(
            [event["reserved_blocks_after"] for event in result.admission_events],
            [1, 0, 1, 0],
        )

    def test_ttft_slack_excludes_admission_configuration_delay(self) -> None:
        clock = VirtualClock()
        engine = DelayedAdmissionEngine(clock)

        run_open_loop(
            engine,
            [self.admission_request("interactive-000", max_tokens=1)],
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
        )

        self.assertAlmostEqual(engine.slo_contexts[0][1], 0.02)

    def test_kv_aware_fifo_rejects_unadmitted_request_after_timeout(self) -> None:
        clock = VirtualClock()
        engine = FakeEngine(clock)
        result = run_open_loop(
            engine,
            [
                self.admission_request("interactive-000", max_tokens=2),
                self.admission_request("interactive-001", max_tokens=1),
            ],
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
                policy="kv_aware_fifo",
                total_kv_blocks=1,
                kvcache_block_size=4,
                max_queue_wait_seconds=0.005,
            ),
        )

        rejected = result.records[1]
        self.assertEqual(rejected.terminal_state.value, "Rejected")
        self.assertIsNone(rejected.admitted_at)
        self.assertEqual(rejected.token_timestamps, [])
        self.assertEqual(rejected.terminal_reason, "kv_reservation_timeout")
        self.assertEqual(result.admission["rejected_requests"], 1)
        rejection = next(
            event
            for event in result.admission_events
            if event["action"] == "rejected"
        )
        self.assertEqual(rejection["reason"], "kv_reservation_timeout")

    def test_kv_aware_fifo_timeout_wins_over_same_step_reservation_release(self) -> None:
        clock = VirtualClock()
        engine = FakeEngine(clock)
        result = run_open_loop(
            engine,
            [
                self.admission_request("interactive-000", max_tokens=1),
                self.admission_request("interactive-001", max_tokens=1),
            ],
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
                policy="kv_aware_fifo",
                total_kv_blocks=1,
                kvcache_block_size=4,
                max_queue_wait_seconds=0.005,
            ),
        )

        self.assertEqual(result.records[1].terminal_state.value, "Rejected")
        self.assertEqual(len(engine.admissions), 1)

    def test_kv_admission_requires_explicit_valid_capacity(self) -> None:
        with self.assertRaisesRegex(ValueError, "total KV blocks"):
            AdmissionConfig(policy="kv_aware_fifo")

    def test_admission_failure_releases_kv_reservation(self) -> None:
        clock = VirtualClock()
        result = run_open_loop(
            FailingAdmissionEngine(),
            [self.admission_request("interactive-000", max_tokens=1)],
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
                policy="kv_aware_fifo",
                total_kv_blocks=1,
                kvcache_block_size=4,
                max_queue_wait_seconds=1.0,
            ),
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.admission["final_reserved_blocks"], 0)
        self.assertEqual(
            [event["action"] for event in result.admission_events],
            ["admitted", "released"],
        )
        self.assertEqual(result.admission_events[-1]["reason"], "engine_add_error")

    def test_duplicate_sequence_id_releases_new_reservation(self) -> None:
        clock = VirtualClock()
        result = run_open_loop(
            DuplicateSequenceEngine(),
            [
                self.admission_request("interactive-000", max_tokens=1),
                self.admission_request("interactive-001", max_tokens=1),
            ],
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
                policy="kv_aware_fifo",
                total_kv_blocks=2,
                kvcache_block_size=4,
                max_queue_wait_seconds=1.0,
            ),
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.admission["final_reserved_blocks"], 0)
        self.assertEqual(
            [event["event_index"] for event in result.admission_events],
            list(range(len(result.admission_events))),
        )
        self.assertEqual(result.admission_events[-1]["reason"], "error_cleanup")

    def test_fail_result_preserves_admission_evidence(self) -> None:
        result = run_open_loop(
            FakeEngine(VirtualClock()),
            [self.admission_request("interactive-000", max_tokens=1)],
            temperature=1.0,
            ignore_eos=True,
            measurement_start=0.0,
            measurement_end=1.0,
            ttft_slo_ms=20.0,
            itl_slo_ms=20.0,
            max_run_seconds=1.0,
            clock=VirtualClock(),
            sleep=lambda _duration: None,
            synchronize=lambda: None,
            admission=AdmissionConfig(
                policy="kv_aware_fifo",
                total_kv_blocks=1,
                kvcache_block_size=4,
                max_queue_wait_seconds=1.0,
            ),
        )
        failed = fail_result(
            result,
            {"type": "RuntimeError", "message": "secondary", "is_cuda_oom": False},
            label="cleanup_error",
        )

        self.assertEqual(failed.admission, result.admission)
        self.assertEqual(failed.admission_events, result.admission_events)

    def test_timeout_releases_kv_reservation(self) -> None:
        clock = VirtualClock()
        result = run_open_loop(
            SlowEngine(clock),
            [self.admission_request("interactive-000", max_tokens=1)],
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
                policy="kv_aware_fifo",
                total_kv_blocks=1,
                kvcache_block_size=4,
                max_queue_wait_seconds=1.0,
            ),
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.admission["final_reserved_blocks"], 0)
        self.assertEqual(result.admission_events[-1]["reason"], "timeout_cleanup")

    def test_requests_are_admitted_at_trace_arrival_not_all_at_start(self) -> None:
        clock = VirtualClock()
        engine = FakeEngine(clock)
        requests = [
            PreparedRequest(
                spec=RequestSpec(
                    request_id="interactive-000",
                    request_class="interactive",
                    arrival_time=0.0,
                    prompt_length=2,
                    max_output_tokens=1,
                    seed=1,
                ),
                prompt_token_ids=[1, 2],
            ),
            PreparedRequest(
                spec=RequestSpec(
                    request_id="interactive-001",
                    request_class="interactive",
                    arrival_time=0.05,
                    prompt_length=2,
                    max_output_tokens=1,
                    seed=2,
                ),
                prompt_token_ids=[3, 4],
            ),
        ]

        result = run_open_loop(
            engine,
            requests,
            temperature=1.0,
            ignore_eos=True,
            measurement_start=0.0,
            measurement_end=0.1,
            ttft_slo_ms=20.0,
            itl_slo_ms=20.0,
            max_run_seconds=1.0,
            clock=clock,
            sleep=clock.sleep,
            synchronize=lambda: None,
        )

        self.assertEqual([time for time, _ in engine.admissions], [0.0, 0.05])
        self.assertEqual(
            engine.slo_contexts,
            [(10, 0.02, 0.02), (11, 0.07, 0.02)],
        )
        self.assertEqual(
            [record.terminal_state.value for record in result.records],
            ["Finished", "Finished"],
        )
        self.assertTrue(result.summary["terminal_counts"]["reconciled"])
        self.assertEqual(len(result.step_events), 2)
        self.assertEqual(len(result.phase_timings), 2)
        self.assertEqual(
            result.phase_timings[0]["model_runner_cuda_ms"],
            0.2,
        )

    def test_timeout_marks_active_request_unfinished(self) -> None:
        clock = VirtualClock()
        request = PreparedRequest(
            spec=RequestSpec(
                request_id="long-000",
                request_class="long",
                arrival_time=0.0,
                prompt_length=2,
                max_output_tokens=2,
                seed=1,
            ),
            prompt_token_ids=[1, 2],
        )

        result = run_open_loop(
            SlowEngine(clock),
            [request],
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
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.records[0].terminal_state.value, "Unfinished")
        self.assertTrue(result.summary["terminal_counts"]["reconciled"])
        self.assertEqual(len(result.step_events), 1)

    def test_admission_failure_is_persistable_and_reconciled(self) -> None:
        clock = VirtualClock()
        requests = [
            PreparedRequest(
                spec=RequestSpec(
                    request_id="interactive-000",
                    request_class="interactive",
                    arrival_time=0.0,
                    prompt_length=2,
                    max_output_tokens=1,
                    seed=1,
                ),
                prompt_token_ids=[1, 2],
            ),
            PreparedRequest(
                spec=RequestSpec(
                    request_id="long-000",
                    request_class="long",
                    arrival_time=0.0,
                    prompt_length=2,
                    max_output_tokens=1,
                    seed=2,
                ),
                prompt_token_ids=[3, 4],
            ),
        ]

        result = run_open_loop(
            FailingAdmissionEngine(),
            requests,
            temperature=1.0,
            ignore_eos=True,
            measurement_start=0.0,
            measurement_end=0.1,
            ttft_slo_ms=20.0,
            itl_slo_ms=20.0,
            max_run_seconds=1.0,
            clock=clock,
            sleep=clock.sleep,
            synchronize=lambda: None,
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error["type"], "RuntimeError")
        self.assertEqual(
            [record.terminal_state.value for record in result.records],
            ["Failed", "Cancelled"],
        )
        self.assertTrue(result.summary["terminal_counts"]["reconciled"])


if __name__ == "__main__":
    unittest.main()
