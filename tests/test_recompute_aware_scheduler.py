from __future__ import annotations

import unittest
from types import SimpleNamespace

from nanovllm import SamplingParams
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus


def scheduler_config(
    *,
    policy: str = "recompute_aware",
    max_num_batched_tokens: int = 8,
    max_num_seqs: int = 4,
    block_size: int = 4,
    num_blocks: int = 16,
    decode_token_budget: int = 8,
    pressure_decode_token_budget: int = 4,
    pressure_high_utilization: float = 0.75,
    pressure_waiting_age_threshold: int = 4,
) -> SimpleNamespace:
    return SimpleNamespace(
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        eos=-1,
        kvcache_block_size=block_size,
        num_kvcache_blocks=num_blocks,
        scheduler_policy=policy,
        decode_token_budget=decode_token_budget,
        decode_step_guard=0,
        pressure_decode_token_budget=pressure_decode_token_budget,
        pressure_decode_step_guard=0,
        pressure_high_utilization=pressure_high_utilization,
        pressure_critical_utilization=1.0,
        pressure_preemption_window=4,
        pressure_preemption_threshold=2,
        pressure_hysteresis_steps=2,
        pressure_waiting_age_threshold=pressure_waiting_age_threshold,
        prefill_chunk_token_budget=0,
    )


def sequence(
    prompt_tokens: int,
    *,
    generated_tokens: int = 0,
    max_tokens: int = 8,
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
    return seq


class RecomputeAwareSchedulerTests(unittest.TestCase):
    def setUp(self) -> None:
        Sequence.block_size = 4

    def run_to_completion(self, scheduler: Scheduler, *, limit: int) -> int:
        for step in range(1, limit + 1):
            if scheduler.is_finished():
                return step - 1
            scheduled, is_prefill = scheduler.schedule()
            scheduler.postprocess(
                scheduled,
                [50_000 + step] * len(scheduled),
                is_prefill=is_prefill,
            )
        self.fail(f"scheduler did not finish within {limit} decisions")

    def test_low_pressure_matches_pressure_aware_phase_behavior(self) -> None:
        baseline = Scheduler(
            scheduler_config(policy="pressure_aware_decode", num_blocks=32)
        )
        candidate = Scheduler(scheduler_config(num_blocks=32))
        install_running(baseline, sequence(2, max_tokens=4))
        install_running(candidate, sequence(2, max_tokens=4))
        baseline.add(sequence(4, token_offset=100))
        candidate.add(sequence(4, token_offset=100))

        baseline_scheduled, baseline_prefill = baseline.schedule()
        candidate_scheduled, candidate_prefill = candidate.schedule()

        self.assertEqual(candidate_prefill, baseline_prefill)
        self.assertEqual(len(candidate_scheduled), len(baseline_scheduled))

    def test_high_pressure_drains_high_release_efficiency_first(self) -> None:
        scheduler = Scheduler(scheduler_config(num_blocks=5))
        efficient = install_running(
            scheduler,
            sequence(4, generated_tokens=4, max_tokens=5),
        )
        install_running(
            scheduler,
            sequence(4, generated_tokens=1, max_tokens=8, token_offset=100),
        )
        scheduler.add(sequence(4, token_offset=200))

        scheduled, is_prefill = scheduler.schedule()

        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, [efficient])
        decision = scheduler.observability_snapshot()["recompute_aware"]
        self.assertEqual(
            decision["last_decision"]["selected_decode_ids"],
            [efficient.seq_id],
        )

    def test_starvation_preempts_low_harm_victim(self) -> None:
        scheduler = Scheduler(
            scheduler_config(num_blocks=4, pressure_waiting_age_threshold=2)
        )
        cheap = install_running(scheduler, sequence(5, max_tokens=8))
        expensive = install_running(
            scheduler,
            sequence(4, generated_tokens=4, max_tokens=8, token_offset=100),
        )
        waiting = sequence(5, token_offset=200)
        scheduler.add(waiting)
        scheduler._waiting_age_steps[waiting.seq_id] = 2

        scheduled, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [waiting])
        self.assertEqual(cheap.status, SequenceStatus.WAITING)
        self.assertEqual(expensive.status, SequenceStatus.RUNNING)
        preemptions = scheduler.last_step_event.preemptions
        self.assertEqual([event.seq_id for event in preemptions], [cheap.seq_id])
        self.assertEqual(preemptions[0].reason, "waiting_prefill_allocation")

    def test_waiting_age_guard_provides_prefill_progress(self) -> None:
        scheduler = Scheduler(
            scheduler_config(num_blocks=8, pressure_waiting_age_threshold=2)
        )
        install_running(scheduler, sequence(2, max_tokens=8))
        waiting = sequence(4, token_offset=100)
        scheduler.add(waiting)
        scheduler._waiting_age_steps[waiting.seq_id] = 2

        scheduled, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [waiting])
        self.assertGreater(waiting.num_scheduled_tokens, 0)

    def test_pending_recompute_age_does_not_retrigger_starvation_prefill(
        self,
    ) -> None:
        scheduler = Scheduler(
            scheduler_config(num_blocks=4, pressure_waiting_age_threshold=2)
        )
        resident = install_running(
            scheduler,
            sequence(11, generated_tokens=1, max_tokens=4),
        )
        pending = sequence(5, token_offset=100)
        pending.pending_recompute = True
        scheduler.add(pending)
        scheduler._waiting_age_steps[pending.seq_id] = 2

        scheduled, is_prefill = scheduler.schedule()

        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, [resident])
        self.assertEqual(
            scheduler.last_step_event.mode,
            "drain_decode",
        )

    def test_high_pressure_drain_stops_at_policy_token_budget(
        self,
    ) -> None:
        scheduler = Scheduler(
            scheduler_config(
                num_blocks=5,
                pressure_decode_token_budget=2,
                pressure_waiting_age_threshold=4,
            )
        )
        install_running(
            scheduler,
            sequence(4, generated_tokens=1, max_tokens=8),
        )
        install_running(
            scheduler,
            sequence(4, generated_tokens=1, max_tokens=8, token_offset=100),
        )
        scheduler.add(sequence(4, token_offset=200))

        phases = []
        for token_id in (101, 102, 103):
            scheduled, is_prefill = scheduler.schedule()
            phases.append(is_prefill)
            scheduler.postprocess(
                scheduled,
                [token_id] * len(scheduled),
                is_prefill=is_prefill,
            )

        self.assertEqual(phases, [False, False, True])

    def test_completion_releases_all_resident_blocks(self) -> None:
        scheduler = Scheduler(scheduler_config(num_blocks=4))
        seq = install_running(scheduler, sequence(2, max_tokens=1))

        scheduled, is_prefill = scheduler.schedule()
        scheduler.postprocess(scheduled, [101], is_prefill=is_prefill)

        self.assertTrue(seq.is_finished)
        self.assertEqual(scheduler.block_manager.current_utilization(), 0.0)
        self.assertEqual(seq.block_table, [])

    def test_preempt_resume_counts_only_executed_recompute_tokens(self) -> None:
        scheduler = Scheduler(scheduler_config(num_blocks=4))
        seq = install_running(scheduler, sequence(3, max_tokens=4))
        scheduler.running.remove(seq)
        scheduler.preempt(seq, reason="unit_test", prepend=True)

        scheduled, is_prefill = scheduler.schedule()
        expected = scheduled[0].num_scheduled_tokens
        scheduler.postprocess(scheduled, [101], is_prefill=is_prefill)

        self.assertTrue(is_prefill)
        self.assertEqual(seq.actual_recompute_tokens, expected)
        self.assertEqual(seq.resume_count, 1)
        self.assertFalse(seq.pending_recompute)
        snapshot = scheduler.observability_snapshot()
        self.assertEqual(snapshot["actual_recompute_tokens"], expected)
        self.assertEqual(snapshot["resume_count"], 1)

    def test_equal_cost_victim_selection_is_deterministic(self) -> None:
        scheduler = Scheduler(scheduler_config(num_blocks=3))
        first = install_running(scheduler, sequence(4, max_tokens=8))
        second = install_running(
            scheduler,
            sequence(4, max_tokens=8, token_offset=100),
        )

        victim = scheduler.select_recompute_victim([second, first])

        self.assertEqual(victim.seq_id, min(first.seq_id, second.seq_id))

    def test_no_waiting_request_uses_normal_decode_batch(self) -> None:
        scheduler = Scheduler(scheduler_config(num_blocks=8))
        first = install_running(scheduler, sequence(2, max_tokens=4))
        second = install_running(
            scheduler,
            sequence(2, max_tokens=4, token_offset=100),
        )

        scheduled, is_prefill = scheduler.schedule()

        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, [first, second])
        decision = scheduler.observability_snapshot()["recompute_aware"]
        self.assertEqual(decision["last_decision"]["mode"], "normal_decode")

    def test_existing_prefill_first_behavior_is_unchanged(self) -> None:
        scheduler = Scheduler(
            scheduler_config(
                policy="prefill_first",
                decode_token_budget=0,
                pressure_decode_token_budget=1,
            )
        )
        running = install_running(scheduler, sequence(2, max_tokens=4))
        waiting = sequence(4, token_offset=100)
        scheduler.add(waiting)

        scheduled, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [waiting])
        self.assertIn(running, scheduler.running)

    def test_shared_prefix_blocks_are_not_counted_as_releasable(self) -> None:
        scheduler = Scheduler(scheduler_config(num_blocks=6))
        first = install_running(scheduler, sequence(8, max_tokens=4))
        first_block = scheduler.block_manager.blocks[first.block_table[0]]
        first_hash = scheduler.block_manager.compute_hash(first.block(0))
        first_block.update(first_hash, first.block(0))
        scheduler.block_manager.hash_to_block_id[first_hash] = (
            first_block.block_id
        )
        second = sequence(8, max_tokens=4)
        cached = scheduler.block_manager.can_allocate(second)
        self.assertEqual(cached, 1)
        scheduler.block_manager.allocate(second, cached)
        second.num_cached_tokens = second.num_tokens
        second.status = SequenceStatus.RUNNING
        second.is_prefill = False
        scheduler.running.append(second)

        cost = scheduler.resident_sequence_cost(first)

        self.assertEqual(cost.resident_kv_blocks, 2)
        self.assertEqual(cost.releasable_blocks, 1)
        self.assertEqual(cost.estimated_recompute_tokens, 4)

    def test_decode_expansion_uses_recompute_aware_victim(self) -> None:
        scheduler = Scheduler(scheduler_config(num_blocks=3))
        candidate = install_running(scheduler, sequence(1, max_tokens=4))
        cheap = install_running(
            scheduler,
            sequence(2, max_tokens=8, token_offset=100),
        )
        expensive = install_running(
            scheduler,
            sequence(4, max_tokens=8, token_offset=200),
        )

        scheduled, is_prefill = scheduler.schedule()

        self.assertFalse(is_prefill)
        self.assertIn(candidate, scheduled)
        self.assertEqual(cheap.status, SequenceStatus.WAITING)
        self.assertEqual(expensive.status, SequenceStatus.RUNNING)
        self.assertEqual(
            scheduler.last_step_event.preemptions[0].reason,
            "decode_kv_expansion",
        )

    def test_mixed_workload_drains_without_kv_leak(self) -> None:
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

        decisions = self.run_to_completion(scheduler, limit=4_000)

        self.assertLess(decisions, 4_000)
        self.assertTrue(scheduler.is_finished())
        self.assertEqual(scheduler.block_manager.current_utilization(), 0.0)
        self.assertTrue(
            all(block.ref_count == 0 for block in scheduler.block_manager.blocks)
        )


if __name__ == "__main__":
    unittest.main()
