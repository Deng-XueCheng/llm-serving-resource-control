from __future__ import annotations

import unittest
from types import SimpleNamespace

from nanovllm import SamplingParams
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def scheduler_config(
    *,
    policy: str = "recompute_aware_bounded",
    max_num_batched_tokens: int = 8,
    max_num_seqs: int = 4,
    block_size: int = 4,
    num_blocks: int = 8,
    max_drain_steps: int = 16,
    waiting_age_limit: int = 32,
    ttft_slo_ms: float = 100.0,
    itl_slo_ms: float = 100.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        eos=-1,
        kvcache_block_size=block_size,
        num_kvcache_blocks=num_blocks,
        scheduler_policy=policy,
        decode_token_budget=8,
        decode_step_guard=0,
        pressure_decode_token_budget=4,
        pressure_decode_step_guard=0,
        pressure_high_utilization=0.75,
        pressure_critical_utilization=1.0,
        pressure_preemption_window=4,
        pressure_preemption_threshold=2,
        pressure_hysteresis_steps=2,
        pressure_waiting_age_threshold=4,
        prefill_chunk_token_budget=0,
        max_drain_steps=max_drain_steps,
        waiting_age_limit=waiting_age_limit,
        ttft_slo_ms=ttft_slo_ms,
        itl_slo_ms=itl_slo_ms,
    )


def sequence(
    prompt_tokens: int,
    *,
    generated_tokens: int = 0,
    max_tokens: int = 64,
    token_offset: int = 0,
) -> Sequence:
    seq = Sequence(
        list(range(token_offset + 1, token_offset + prompt_tokens + 1)),
        SamplingParams(
            temperature=1.0,
            max_tokens=max_tokens,
            ignore_eos=True,
        ),
    )
    for index in range(generated_tokens):
        seq.append_token(10_000 + token_offset + index)
    return seq


def install_running(scheduler: Scheduler, seq: Sequence) -> Sequence:
    cached = scheduler.block_manager.can_allocate(seq)
    if cached < 0:
        raise AssertionError("test fixture cannot allocate resident sequence")
    scheduler.block_manager.allocate(seq, cached)
    seq.num_cached_tokens = seq.num_tokens
    seq.status = SequenceStatus.RUNNING
    seq.is_prefill = False
    scheduler.running.append(seq)
    scheduler.initialize_request_progress(seq)
    return seq


class BoundedRecomputeSchedulerTests(unittest.TestCase):
    def setUp(self) -> None:
        Sequence.block_size = 4
        self.clock = FakeClock()

    def make_scheduler(self, **kwargs) -> Scheduler:
        return Scheduler(scheduler_config(**kwargs), clock=self.clock)

    def execute_step(self, scheduler: Scheduler, token_id: int = 50_000):
        scheduled, is_prefill = scheduler.schedule()
        scheduler.postprocess(
            scheduled,
            [token_id] * len(scheduled),
            is_prefill=is_prefill,
        )
        return scheduled, is_prefill, scheduler.last_step_event

    def test_drain_episode_exits_when_budget_is_exhausted(self) -> None:
        scheduler = self.make_scheduler(num_blocks=4, max_drain_steps=2)
        install_running(scheduler, sequence(7, max_tokens=16))
        install_running(
            scheduler,
            sequence(7, max_tokens=16, token_offset=100),
        )
        waiting = sequence(4, generated_tokens=1, token_offset=200)
        waiting.pending_recompute = True
        scheduler.add(waiting)

        _, first_prefill, first = self.execute_step(scheduler, 101)
        _, second_prefill, second = self.execute_step(scheduler, 102)
        scheduled, third_prefill, third = self.execute_step(scheduler, 103)

        self.assertFalse(first_prefill)
        self.assertFalse(second_prefill)
        self.assertTrue(third_prefill)
        self.assertIn(waiting, scheduled)
        self.assertEqual(first.drain_episode_step, 1)
        self.assertEqual(second.drain_episode_step, 2)
        self.assertEqual(third.drain_exit_reason, "max_drain_steps")

    def test_waiting_age_guard_forces_one_prefill_progress_window(self) -> None:
        scheduler = self.make_scheduler(
            num_blocks=4,
            max_drain_steps=100,
            waiting_age_limit=2,
        )
        install_running(
            scheduler,
            sequence(10, generated_tokens=1, max_tokens=16),
        )
        waiting = sequence(4, generated_tokens=1, token_offset=100)
        waiting.pending_recompute = True
        scheduler.add(waiting)
        scheduler._waiting_age_steps[waiting.seq_id] = 2

        scheduled, is_prefill, event = self.execute_step(scheduler)

        self.assertTrue(is_prefill)
        self.assertIn(waiting, scheduled)
        self.assertEqual(event.mode, "bounded_prefill")
        self.assertIsNone(event.drain_episode_id)
        self.assertIsNone(event.drain_exit_reason)
        self.assertEqual(event.fairness_trigger_reason, "waiting_age_limit")

    def test_positive_itl_slack_guard_ends_drain_before_budget(self) -> None:
        scheduler = self.make_scheduler(
            num_blocks=4,
            max_drain_steps=100,
            waiting_age_limit=100,
            itl_slo_ms=100.0,
        )
        install_running(
            scheduler,
            sequence(10, generated_tokens=1, max_tokens=16),
        )
        waiting = sequence(4, generated_tokens=1, token_offset=100)
        waiting.pending_recompute = True
        scheduler.add(waiting)

        _, is_prefill, first = self.execute_step(scheduler, 101)
        self.assertFalse(is_prefill)
        self.assertEqual(first.drain_episode_step, 1)

        self.clock.advance(0.101)
        scheduled, is_prefill, second = self.execute_step(scheduler, 102)

        self.assertTrue(is_prefill)
        self.assertIn(waiting, scheduled)
        self.assertEqual(second.drain_exit_reason, "itl_slo_deadline")
        self.assertEqual(second.slo_guard_seq_id, waiting.seq_id)

    def test_request_already_overdue_at_episode_entry_uses_hard_bound(self) -> None:
        scheduler = self.make_scheduler(
            num_blocks=4,
            max_drain_steps=2,
            waiting_age_limit=100,
        )
        waiting = sequence(4, generated_tokens=1, token_offset=100)
        waiting.pending_recompute = True
        scheduler.add(waiting)
        self.clock.advance(0.101)
        install_running(scheduler, sequence(7, max_tokens=16))
        install_running(
            scheduler,
            sequence(5, max_tokens=16, token_offset=300),
        )

        _, first_prefill, _ = self.execute_step(scheduler, 101)
        _, second_prefill, _ = self.execute_step(scheduler, 102)
        _, third_prefill, third = self.execute_step(scheduler, 103)

        self.assertFalse(first_prefill)
        self.assertFalse(second_prefill)
        self.assertTrue(third_prefill)
        self.assertEqual(third.drain_exit_reason, "max_drain_steps")

    def test_resource_release_ends_episode_below_high_watermark(self) -> None:
        scheduler = self.make_scheduler(
            num_blocks=4,
            max_drain_steps=100,
            waiting_age_limit=100,
        )
        finishing = install_running(
            scheduler,
            sequence(7, max_tokens=1),
        )
        install_running(
            scheduler,
            sequence(7, max_tokens=16, token_offset=100),
        )
        waiting = sequence(4, generated_tokens=1, token_offset=200)
        waiting.pending_recompute = True
        scheduler.add(waiting)

        selected, is_prefill, first = self.execute_step(scheduler, 101)
        self.assertFalse(is_prefill)
        self.assertEqual(selected, [finishing])
        self.assertTrue(finishing.is_finished)
        self.assertEqual(first.drain_episode_step, 1)

        scheduled, is_prefill, second = self.execute_step(scheduler, 102)
        self.assertTrue(is_prefill)
        self.assertIn(waiting, scheduled)
        self.assertEqual(second.drain_exit_reason, "resource_released")

    def test_low_pressure_does_not_enter_drain_episode(self) -> None:
        scheduler = self.make_scheduler(num_blocks=16)
        first = install_running(scheduler, sequence(2, max_tokens=4))
        second = install_running(
            scheduler,
            sequence(2, max_tokens=4, token_offset=100),
        )
        scheduler.add(sequence(4, token_offset=200))

        scheduled, is_prefill, event = self.execute_step(scheduler)

        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, [first, second])
        self.assertIsNone(event.drain_episode_id)
        self.assertEqual(event.mode, "baseline_decode")

    def test_failed_first_drain_decode_does_not_create_zero_step_episode(self) -> None:
        scheduler = self.make_scheduler(num_blocks=1)
        resident = install_running(
            scheduler,
            sequence(4, max_tokens=8),
        )
        resident.append_token(10_000)
        scheduler.add(sequence(4, token_offset=100))

        _, is_prefill, event = self.execute_step(scheduler)

        self.assertTrue(is_prefill)
        self.assertEqual(resident.status, SequenceStatus.WAITING)
        self.assertIsNone(event.drain_episode_id)
        self.assertEqual(event.drain_episode_step, 0)
        self.assertEqual(event.drain_tokens, 0)
        self.assertIsNone(event.drain_exit_reason)
        self.assertEqual(
            scheduler.observability_snapshot()["recompute_aware_bounded"][
                "drain_episode_count"
            ],
            0,
        )

    def test_last_resident_finish_closes_episode_before_waiting_prefill(self) -> None:
        scheduler = self.make_scheduler(num_blocks=2)
        resident = install_running(
            scheduler,
            sequence(7, max_tokens=1),
        )
        waiting = sequence(4, generated_tokens=1, token_offset=100)
        waiting.pending_recompute = True
        scheduler.add(waiting)

        selected, first_prefill, first = self.execute_step(scheduler, 101)
        scheduled, second_prefill, second = self.execute_step(scheduler, 102)

        self.assertEqual(selected, [resident])
        self.assertFalse(first_prefill)
        self.assertEqual(first.drain_episode_step, 1)
        self.assertTrue(second_prefill)
        self.assertIn(waiting, scheduled)
        self.assertEqual(second.drain_episode_id, first.drain_episode_id)
        self.assertEqual(second.drain_exit_reason, "resident_empty")
        self.assertFalse(
            scheduler.observability_snapshot()["recompute_aware_bounded"][
                "drain_active"
            ]
        )

    def test_drain_stops_at_policy_token_budget(self) -> None:
        scheduler = self.make_scheduler(
            num_blocks=4,
            max_drain_steps=16,
            waiting_age_limit=32,
        )
        install_running(scheduler, sequence(7, max_tokens=32))
        install_running(
            scheduler,
            sequence(7, max_tokens=32, token_offset=100),
        )
        scheduler.add(sequence(4, token_offset=200))

        events = [self.execute_step(scheduler, 101 + index)[2] for index in range(5)]

        self.assertTrue(
            all(event.mode == "drain_decode" for event in events[:4])
        )
        self.assertEqual(events[3].drain_episode_step, 4)
        self.assertEqual(events[4].mode, "prefill")

    def test_bounded_prefill_keeps_recompute_aware_victim_selection(self) -> None:
        scheduler = self.make_scheduler(
            num_blocks=4,
            waiting_age_limit=1,
        )
        cheap = install_running(scheduler, sequence(5, max_tokens=8))
        expensive = install_running(
            scheduler,
            sequence(4, generated_tokens=4, max_tokens=8, token_offset=100),
        )
        waiting = sequence(5, generated_tokens=1, token_offset=200)
        waiting.pending_recompute = True
        scheduler.add(waiting)
        scheduler._waiting_age_steps[waiting.seq_id] = 1

        _, is_prefill, event = self.execute_step(scheduler)

        self.assertTrue(is_prefill)
        self.assertEqual(cheap.status, SequenceStatus.WAITING)
        self.assertEqual(expensive.status, SequenceStatus.RUNNING)
        self.assertEqual([item.seq_id for item in event.preemptions], [cheap.seq_id])

    def test_decode_expansion_still_uses_recompute_aware_victim(self) -> None:
        scheduler = self.make_scheduler(num_blocks=3)
        candidate = install_running(scheduler, sequence(1, max_tokens=4))
        cheap = install_running(
            scheduler,
            sequence(2, max_tokens=8, token_offset=100),
        )
        expensive = install_running(
            scheduler,
            sequence(4, max_tokens=8, token_offset=200),
        )

        scheduled, is_prefill, event = self.execute_step(scheduler)

        self.assertFalse(is_prefill)
        self.assertIn(candidate, scheduled)
        self.assertEqual(cheap.status, SequenceStatus.WAITING)
        self.assertEqual(expensive.status, SequenceStatus.RUNNING)
        self.assertEqual(event.preemptions[0].reason, "decode_kv_expansion")

    def test_progress_gap_accounting_distinguishes_post_token_progress(self) -> None:
        scheduler = self.make_scheduler(num_blocks=8)
        seq = sequence(4, max_tokens=8)
        scheduler.add(seq)

        _, _, first = self.execute_step(scheduler, 101)
        scheduler.running.remove(seq)
        scheduler.preempt(seq, reason="unit_test", prepend=True)
        scheduler._scheduler_step_index += 2
        _, _, resumed = self.execute_step(scheduler, 102)

        first_state = next(
            item for item in first.active_progress_states if item.seq_id == seq.seq_id
        )
        resumed_state = next(
            item for item in resumed.active_progress_states if item.seq_id == seq.seq_id
        )
        self.assertFalse(first_state.had_emitted_token)
        self.assertTrue(resumed_state.had_emitted_token)
        self.assertEqual(seq.last_progress_gap_steps, 3)

    def test_unbounded_policy_remains_isolated(self) -> None:
        scheduler = self.make_scheduler(
            policy="recompute_aware",
            num_blocks=4,
            max_drain_steps=1,
            waiting_age_limit=1,
        )
        install_running(scheduler, sequence(7, max_tokens=16))
        waiting = sequence(4, generated_tokens=1, token_offset=100)
        waiting.pending_recompute = True
        scheduler.add(waiting)
        scheduler._waiting_age_steps[waiting.seq_id] = 10

        phases = [self.execute_step(scheduler, 101 + index)[1] for index in range(3)]

        self.assertEqual(phases, [False, False, False])

    def test_mixed_workload_finishes_without_kv_leak(self) -> None:
        scheduler = self.make_scheduler(
            max_num_batched_tokens=256,
            max_num_seqs=8,
            block_size=256,
            num_blocks=4,
            max_drain_steps=16,
            waiting_age_limit=32,
        )
        Sequence.block_size = 256
        for _ in range(4):
            scheduler.add(sequence(768, max_tokens=8))
        for _ in range(16):
            scheduler.add(sequence(32, max_tokens=8))

        for step in range(1, 4_001):
            if scheduler.is_finished():
                break
            self.execute_step(scheduler, 60_000 + step)
        else:
            self.fail("bounded scheduler did not finish within 4000 steps")

        self.assertTrue(scheduler.is_finished())
        self.assertEqual(scheduler.block_manager.current_utilization(), 0.0)
        self.assertTrue(
            all(
                block.ref_count == 0
                for block in scheduler.block_manager.blocks
            )
        )


if __name__ == "__main__":
    unittest.main()
