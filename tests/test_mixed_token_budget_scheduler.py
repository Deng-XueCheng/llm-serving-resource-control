from __future__ import annotations

import unittest
from types import SimpleNamespace

from nanovllm import SamplingParams
from nanovllm.engine.scheduler import Scheduler, ScheduledSequence
from nanovllm.engine.sequence import Sequence, SequenceStatus


def config(*, budget: int = 4, max_seqs: int = 4, blocks: int = 32,
           policy: str = "mixed_token_budget"):
    return SimpleNamespace(
        max_num_batched_tokens=budget,
        max_num_seqs=max_seqs,
        eos=-1,
        kvcache_block_size=4,
        num_kvcache_blocks=blocks,
        scheduler_policy=policy,
        decode_token_budget=budget,
        decode_step_guard=0,
        mixed_min_prefill_tokens=1,
        mixed_waiting_age_threshold=2,
        mixed_slack_threshold_ms=20.0,
        ttft_slo_ms=100.0,
        itl_slo_ms=100.0,
    )


def sequence(tokens: int, *, max_tokens: int = 4, offset: int = 0):
    return Sequence(
        list(range(offset + 1, offset + tokens + 1)),
        SamplingParams(temperature=1.0, max_tokens=max_tokens, ignore_eos=True),
    )


def install_running(scheduler: Scheduler, seq: Sequence) -> Sequence:
    cached = scheduler.block_manager.can_allocate(seq)
    scheduler.block_manager.allocate(seq, cached)
    seq.num_cached_tokens = seq.num_tokens
    seq.status = SequenceStatus.RUNNING
    seq.is_prefill = False
    scheduler.running.append(seq)
    return seq


class MixedTokenBudgetSchedulerTests(unittest.TestCase):
    def setUp(self):
        Sequence.block_size = 4

    def test_decode_and_prefill_share_one_global_budget(self):
        scheduler = Scheduler(config(budget=4))
        running = [install_running(scheduler, sequence(2, offset=i * 10)) for i in range(2)]
        waiting = sequence(8, offset=100)
        scheduler.add(waiting)

        batch = scheduler.schedule()

        self.assertEqual([item.phase for item in batch.items], ["decode", "decode", "prefill"])
        self.assertEqual([item.sequence for item in batch.items[:2]], running)
        self.assertEqual(batch.items[2].sequence, waiting)
        self.assertEqual(batch.items[2].num_scheduled_tokens, 2)
        self.assertEqual(batch.num_scheduled_tokens, 4)
        self.assertEqual(scheduler.last_step_event.mixed_step_ratio, 1.0)

    def test_partial_prefill_advances_across_steps_and_drains(self):
        scheduler = Scheduler(config(budget=3))
        running = install_running(scheduler, sequence(2, max_tokens=2))
        waiting = sequence(5, max_tokens=1, offset=100)
        scheduler.add(waiting)

        first = scheduler.schedule()
        self.assertEqual(first.num_scheduled_tokens, 3)
        scheduler.postprocess_mixed(first, {running.seq_id: 1001})
        self.assertEqual(waiting.num_cached_tokens, 2)

        while not scheduler.is_finished():
            batch = scheduler.schedule()
            sampled = {
                item.sequence.seq_id: 2000 + item.sequence.seq_id
                for item in batch.items if item.requires_sampling
            }
            scheduler.postprocess_mixed(batch, sampled)

        self.assertEqual(scheduler.block_manager.current_utilization(), 0.0)

    def test_prefix_cache_hit_only_schedules_uncached_tail(self):
        scheduler = Scheduler(config(budget=8))
        first = sequence(8, max_tokens=1)
        scheduler.add(first)
        initial = scheduler.schedule()
        scheduler.postprocess_mixed(initial, {first.seq_id: 1001})

        cached = sequence(10, max_tokens=1)
        scheduler.add(cached)
        batch = scheduler.schedule()
        item = batch.items[0]

        self.assertEqual(item.phase, "prefill")
        self.assertEqual(cached.num_cached_tokens, 8)
        self.assertEqual(item.num_scheduled_tokens, 2)
        self.assertEqual(len(cached.block_table), 3)

    def test_legacy_policy_keeps_phase_exclusive_protocol(self):
        legacy = SimpleNamespace(**vars(config()))
        legacy.scheduler_policy = "prefill_first"
        legacy.decode_token_budget = 0
        scheduler = Scheduler(legacy)
        scheduler.add(sequence(2))

        scheduled, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(len(scheduled), 1)

    def test_slo_allocator_increases_prefill_budget_with_waiting_age(self):
        scheduler = Scheduler(config(budget=8, policy="mixed_slo_budget"))
        install_running(scheduler, sequence(2))
        scheduler.add(sequence(20, offset=100))
        scheduler._waiting_age_steps[scheduler.waiting[0].seq_id] = 4

        batch = scheduler.schedule()

        event = scheduler.last_step_event
        self.assertGreaterEqual(event.planned_prefill_budget, 2)
        self.assertGreater(event.planned_prefill_budget, 1)
        self.assertLessEqual(batch.num_scheduled_tokens, 8)

    def test_ttft_slack_forces_prefill_progress(self):
        scheduler = Scheduler(config(budget=4, blocks=2, policy="mixed_slo_budget"))
        install_running(scheduler, sequence(4, offset=10))
        waiting = sequence(4, offset=100)
        scheduler.add(waiting)
        waiting.ttft_deadline_at = scheduler._clock() + 0.01

        batch = scheduler.schedule()

        self.assertTrue(any(item.sequence is waiting for item in batch.items))
        self.assertLessEqual(batch.num_scheduled_tokens, 4)
        self.assertEqual(len({item.sequence.seq_id for item in batch.items}), len(batch.items))

    def test_itl_slack_preserves_decode_progress(self):
        scheduler = Scheduler(config(budget=4, policy="mixed_slo_budget"))
        running = install_running(scheduler, sequence(2))
        running.next_token_deadline_at = scheduler._clock() + 0.01
        scheduler.add(sequence(12, offset=100))
        scheduler._waiting_age_steps[scheduler.waiting[0].seq_id] = 8

        batch = scheduler.schedule()

        self.assertTrue(any(item.sequence is running and item.phase == "decode" for item in batch.items))
        self.assertGreaterEqual(scheduler.last_step_event.planned_decode_budget, 1)

    def test_no_waiting_does_not_reserve_prefill_budget(self):
        scheduler = Scheduler(config(budget=4, policy="mixed_slo_budget"))
        install_running(scheduler, sequence(2))

        scheduler.schedule()

        self.assertEqual(scheduler.last_step_event.planned_prefill_budget, 0)


if __name__ == "__main__":
    unittest.main()
