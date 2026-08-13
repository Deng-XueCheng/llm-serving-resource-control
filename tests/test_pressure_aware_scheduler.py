from __future__ import annotations

import unittest
from types import SimpleNamespace

from nanovllm import SamplingParams
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus


def scheduler_config(
    *,
    max_num_batched_tokens: int = 4,
    max_num_seqs: int = 4,
    block_size: int = 4,
    num_blocks: int = 32,
    scheduler_policy: str = "pressure_aware_decode",
    decode_token_budget: int = 8,
    decode_step_guard: int = 0,
    pressure_decode_token_budget: int = 4,
    pressure_decode_step_guard: int = 0,
    pressure_high_utilization: float = 0.75,
    pressure_critical_utilization: float = 1.0,
    pressure_preemption_window: int = 4,
    pressure_preemption_threshold: int = 2,
    pressure_hysteresis_steps: int = 2,
    pressure_waiting_age_threshold: int = 4,
) -> SimpleNamespace:
    return SimpleNamespace(
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        eos=-1,
        kvcache_block_size=block_size,
        num_kvcache_blocks=num_blocks,
        scheduler_policy=scheduler_policy,
        decode_token_budget=decode_token_budget,
        decode_step_guard=decode_step_guard,
        pressure_decode_token_budget=pressure_decode_token_budget,
        pressure_decode_step_guard=pressure_decode_step_guard,
        pressure_high_utilization=pressure_high_utilization,
        pressure_critical_utilization=pressure_critical_utilization,
        pressure_preemption_window=pressure_preemption_window,
        pressure_preemption_threshold=pressure_preemption_threshold,
        pressure_hysteresis_steps=pressure_hysteresis_steps,
        pressure_waiting_age_threshold=pressure_waiting_age_threshold,
    )


def sequence(token_count: int, *, max_tokens: int = 8) -> Sequence:
    return Sequence(
        list(range(1, token_count + 1)),
        SamplingParams(
            temperature=1.0,
            max_tokens=max_tokens,
            ignore_eos=True,
        ),
    )


class PressureAwareSchedulerTests(unittest.TestCase):
    def setUp(self) -> None:
        Sequence.block_size = 4

    def make_running_sequence(self, scheduler: Scheduler) -> Sequence:
        seq = sequence(2)
        scheduler.add(seq)
        scheduled, is_prefill = scheduler.schedule()
        self.assertTrue(is_prefill)
        scheduler.postprocess(scheduled, [101], is_prefill=True)
        self.assertEqual(seq.status, SequenceStatus.RUNNING)
        return seq

    def run_to_completion(self, scheduler: Scheduler, *, limit: int) -> int:
        for step in range(1, limit + 1):
            if scheduler.is_finished():
                return step - 1
            scheduled, is_prefill = scheduler.schedule()
            scheduler.postprocess(
                scheduled,
                [101] * len(scheduled),
                is_prefill=is_prefill,
            )
        self.fail(f"scheduler did not finish within {limit} decisions")

    def test_normal_state_uses_normal_decode_budget(self) -> None:
        scheduler = Scheduler(scheduler_config(num_blocks=32))
        running = self.make_running_sequence(scheduler)
        waiting = sequence(8)
        scheduler.add(waiting)

        scheduled, is_prefill = scheduler.schedule()

        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, [running])
        decision = scheduler.observability_snapshot()["pressure_aware"]
        self.assertEqual(decision["state"], "normal")
        self.assertEqual(decision["last_decision"]["decode_token_budget"], 8)

    def test_recent_preemptions_enter_pressure_state(self) -> None:
        scheduler = Scheduler(scheduler_config(num_blocks=32))
        self.make_running_sequence(scheduler)
        scheduler._recent_preemption_deltas.extend([1, 1])

        scheduler.schedule()

        decision = scheduler.observability_snapshot()["pressure_aware"]
        self.assertEqual(decision["state"], "pressure")
        self.assertEqual(decision["last_decision"]["decode_token_budget"], 4)

    def test_waiting_age_enters_critical_and_forces_prefill(self) -> None:
        scheduler = Scheduler(
            scheduler_config(
                max_num_batched_tokens=8,
                num_blocks=32,
                pressure_waiting_age_threshold=2,
            )
        )
        running = self.make_running_sequence(scheduler)
        waiting = sequence(8)
        scheduler.add(waiting)
        scheduler._waiting_age_steps[waiting.seq_id] = 2

        scheduled, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [waiting])
        decision = scheduler.observability_snapshot()["pressure_aware"]
        self.assertEqual(decision["state"], "critical")
        self.assertTrue(decision["last_decision"]["forced_prefill"])
        self.assertIn(running, scheduler.running)

    def test_hysteresis_delays_return_to_normal(self) -> None:
        scheduler = Scheduler(
            scheduler_config(
                num_blocks=32,
                pressure_hysteresis_steps=2,
            )
        )
        self.make_running_sequence(scheduler)
        scheduler._recent_preemption_deltas.extend([1, 1])
        scheduler.schedule()
        scheduler._recent_preemption_deltas.clear()

        scheduler.schedule()
        self.assertEqual(
            scheduler.observability_snapshot()["pressure_aware"]["state"],
            "pressure",
        )
        scheduler.schedule()
        self.assertEqual(
            scheduler.observability_snapshot()["pressure_aware"]["state"],
            "normal",
        )

    def test_critical_hysteresis_delays_deescalation_to_pressure(self) -> None:
        scheduler = Scheduler(
            scheduler_config(
                pressure_high_utilization=0.5,
                pressure_critical_utilization=0.9,
                pressure_hysteresis_steps=2,
            )
        )
        scheduler.block_manager.current_utilization = lambda: 1.0

        state, _ = scheduler._classify_pressure_state()
        self.assertEqual(state, "critical")

        scheduler.block_manager.current_utilization = lambda: 0.75
        state, _ = scheduler._classify_pressure_state()
        self.assertEqual(state, "critical")

        state, _ = scheduler._classify_pressure_state()
        self.assertEqual(state, "pressure")

    def test_critical_prefill_does_not_age_newly_preempted_sequence(self) -> None:
        scheduler = Scheduler(
            scheduler_config(
                max_num_batched_tokens=8,
                num_blocks=2,
                pressure_waiting_age_threshold=1,
            )
        )
        running = self.make_running_sequence(scheduler)
        waiting = sequence(8)
        scheduler.add(waiting)
        scheduler._waiting_age_steps[waiting.seq_id] = 1

        scheduled, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [waiting])
        self.assertEqual(running.status, SequenceStatus.WAITING)
        snapshot = scheduler.observability_snapshot()
        decision = snapshot["pressure_aware"]
        self.assertEqual(decision["max_waiting_age_steps"], 0)
        self.assertEqual(decision["recent_preemptions"], 1)
        self.assertLessEqual(
            snapshot["kv_cache"]["final_used_blocks"],
            snapshot["kv_cache"]["total_blocks"],
        )

    def test_full_kv_with_fresh_waiter_uses_pressure_decode(self) -> None:
        scheduler = Scheduler(
            scheduler_config(
                max_num_batched_tokens=8,
                num_blocks=3,
                pressure_waiting_age_threshold=1,
            )
        )
        self.make_running_sequence(scheduler)
        waiting = sequence(8)
        scheduler.add(waiting)
        scheduler._waiting_age_steps[waiting.seq_id] = 1

        scheduled, is_prefill = scheduler.schedule()
        self.assertTrue(is_prefill)
        scheduler.postprocess(scheduled, [101], is_prefill=True)

        scheduler.add(sequence(8))
        self.assertEqual(scheduler.block_manager.current_utilization(), 1.0)
        scheduled, is_prefill = scheduler.schedule()

        self.assertFalse(is_prefill)
        self.assertTrue(scheduled)
        decision = scheduler.observability_snapshot()["pressure_aware"]
        self.assertEqual(decision["state"], "critical")
        self.assertFalse(decision["last_decision"]["forced_prefill"])
        self.assertEqual(decision["last_decision"]["decode_token_budget"], 4)

    def test_four_block_mixed_workload_completes_without_prefill_livelock(
        self,
    ) -> None:
        scheduler = Scheduler(
            scheduler_config(
                max_num_batched_tokens=256,
                max_num_seqs=8,
                block_size=256,
                num_blocks=4,
                pressure_waiting_age_threshold=4,
            )
        )
        Sequence.block_size = 256
        for _ in range(4):
            scheduler.add(sequence(768, max_tokens=8))
        for _ in range(16):
            scheduler.add(sequence(32, max_tokens=8))

        decisions = self.run_to_completion(scheduler, limit=2_000)

        self.assertLess(decisions, 2_000)
        self.assertEqual(scheduler.block_manager.current_utilization(), 0.0)
        self.assertLess(scheduler.preemption_count, 500)

    def test_reset_clears_adaptive_history(self) -> None:
        scheduler = Scheduler(scheduler_config(num_blocks=32))
        self.make_running_sequence(scheduler)
        scheduler._recent_preemption_deltas.extend([1, 1])
        scheduler.schedule()
        scheduler.reset_observability()

        decision = scheduler.observability_snapshot()["pressure_aware"]
        self.assertEqual(decision["state"], "normal")
        self.assertEqual(decision["recent_preemptions"], 0)
        self.assertEqual(decision["decision_count"], 0)

    def test_invalid_pressure_thresholds_are_rejected(self) -> None:
        invalid = scheduler_config(
            pressure_high_utilization=0.9,
            pressure_critical_utilization=0.8,
        )
        with self.assertRaisesRegex(ValueError, "critical_utilization"):
            Scheduler(invalid)

    def test_pressure_high_utilization_must_be_below_one(self) -> None:
        invalid = scheduler_config(
            pressure_high_utilization=1.0,
            pressure_critical_utilization=1.0,
        )
        with self.assertRaisesRegex(ValueError, "pressure_high_utilization"):
            Scheduler(invalid)


if __name__ == "__main__":
    unittest.main()
