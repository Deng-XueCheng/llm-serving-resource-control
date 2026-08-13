from __future__ import annotations

import unittest
from types import SimpleNamespace

from nanovllm import SamplingParams
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus


def scheduler_config(
    *,
    max_num_batched_tokens: int = 8,
    max_num_seqs: int = 4,
    block_size: int = 4,
    num_blocks: int = 32,
    decode_token_budget: int = 4,
    decode_step_guard: int = 0,
    prefill_chunk_token_budget: int = 3,
    pressure_decode_token_budget: int = 1,
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
        scheduler_policy="chunked_prefill_budgeted",
        decode_token_budget=decode_token_budget,
        decode_step_guard=decode_step_guard,
        prefill_chunk_token_budget=prefill_chunk_token_budget,
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


class ChunkedPrefillBudgetedSchedulerTests(unittest.TestCase):
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

    def test_prefill_without_decode_contention_uses_full_budget(self) -> None:
        scheduler = Scheduler(
            scheduler_config(prefill_chunk_token_budget=3)
        )
        waiting = sequence(10)
        scheduler.add(waiting)

        scheduled, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [waiting])
        self.assertEqual(waiting.num_scheduled_tokens, 8)
        self.assertEqual(waiting.status, SequenceStatus.WAITING)
        decision = scheduler.observability_snapshot()[
            "chunked_prefill_budgeted"
        ]["last_decision"]
        self.assertEqual(decision["prefill_chunk_token_budget"], 3)
        self.assertEqual(decision["effective_prefill_token_budget"], 8)
        self.assertEqual(decision["scheduled_prefill_tokens"], 8)

    def test_decode_budget_then_runs_bounded_prefill_chunk(self) -> None:
        scheduler = Scheduler(
            scheduler_config(
                decode_token_budget=1,
                pressure_decode_token_budget=1,
                prefill_chunk_token_budget=3,
            )
        )
        running = self.make_running_sequence(scheduler)
        waiting = sequence(10)
        scheduler.add(waiting)

        decoded, is_prefill = scheduler.schedule()
        self.assertFalse(is_prefill)
        self.assertEqual(decoded, [running])
        scheduler.postprocess(decoded, [102], is_prefill=False)

        prefilled, is_prefill = scheduler.schedule()
        self.assertTrue(is_prefill)
        self.assertEqual(prefilled, [waiting])
        self.assertEqual(waiting.num_scheduled_tokens, 3)
        decision = scheduler.observability_snapshot()[
            "chunked_prefill_budgeted"
        ]["last_decision"]
        self.assertEqual(decision["effective_prefill_token_budget"], 3)

    def test_pressure_state_switches_to_pressure_decode_budget(self) -> None:
        scheduler = Scheduler(
            scheduler_config(
                decode_token_budget=4,
                pressure_decode_token_budget=1,
            )
        )
        first = sequence(2)
        second = sequence(2)
        scheduler.add(first)
        scheduler.add(second)
        scheduled, is_prefill = scheduler.schedule()
        scheduler.postprocess(scheduled, [101, 102], is_prefill=True)
        scheduler.add(sequence(10))
        scheduler._recent_preemption_deltas.extend([1, 1])

        decoded, is_prefill = scheduler.schedule()

        self.assertFalse(is_prefill)
        self.assertEqual(len(decoded), 1)
        decision = scheduler.observability_snapshot()[
            "chunked_prefill_budgeted"
        ]["last_decision"]
        self.assertEqual(decision["state"], "pressure")
        self.assertEqual(decision["decode_token_budget"], 1)

    def test_waiting_age_forces_bounded_prefill_progress(self) -> None:
        scheduler = Scheduler(
            scheduler_config(
                prefill_chunk_token_budget=2,
                pressure_waiting_age_threshold=2,
            )
        )
        running = self.make_running_sequence(scheduler)
        waiting = sequence(10)
        scheduler.add(waiting)
        scheduler._waiting_age_steps[waiting.seq_id] = 2

        scheduled, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [waiting])
        self.assertEqual(waiting.num_scheduled_tokens, 2)
        self.assertIn(running, scheduler.running)
        decision = scheduler.observability_snapshot()[
            "chunked_prefill_budgeted"
        ]["last_decision"]
        self.assertTrue(decision["forced_prefill"])

    def test_invalid_prefill_chunk_budget_is_rejected(self) -> None:
        for value in (0, -1, True, 9):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "prefill_chunk_token_budget",
                ):
                    Scheduler(
                        scheduler_config(
                            max_num_batched_tokens=8,
                            prefill_chunk_token_budget=value,
                        )
                    )

    def test_mixed_workload_completes_without_prefill_livelock(self) -> None:
        scheduler = Scheduler(
            scheduler_config(
                max_num_batched_tokens=256,
                max_num_seqs=8,
                block_size=256,
                num_blocks=4,
                decode_token_budget=8,
                pressure_decode_token_budget=4,
                pressure_waiting_age_threshold=4,
                prefill_chunk_token_budget=128,
            )
        )
        Sequence.block_size = 256
        for _ in range(4):
            scheduler.add(sequence(768, max_tokens=8))
        for _ in range(16):
            scheduler.add(sequence(32, max_tokens=8))

        decisions = self.run_to_completion(scheduler, limit=4_000)

        self.assertLess(decisions, 4_000)
        self.assertEqual(scheduler.block_manager.current_utilization(), 0.0)

    def test_reset_clears_chunked_scheduler_history(self) -> None:
        scheduler = Scheduler(scheduler_config())
        self.make_running_sequence(scheduler)
        scheduler._recent_preemption_deltas.extend([1, 1])
        scheduler.schedule()

        scheduler.reset_observability()

        observation = scheduler.observability_snapshot()[
            "chunked_prefill_budgeted"
        ]
        self.assertEqual(observation["state"], "normal")
        self.assertEqual(observation["recent_preemptions"], 0)
        self.assertEqual(observation["decision_count"], 0)


if __name__ == "__main__":
    unittest.main()
