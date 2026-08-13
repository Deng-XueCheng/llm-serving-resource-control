from __future__ import annotations

import unittest
from types import SimpleNamespace

from nanovllm import SamplingParams
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus


def scheduler_config(
    *,
    max_num_batched_tokens: int = 4,
    max_num_seqs: int = 4,
    block_size: int = 4,
    num_blocks: int = 32,
    scheduler_policy: str = "prefill_first",
    decode_token_budget: int = 0,
    decode_step_guard: int = 0,
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


class SchedulerBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        Sequence.block_size = 4

    def make_running_sequence(self, scheduler: Scheduler) -> Sequence:
        seq = sequence(2)
        scheduler.add(seq)
        scheduled, is_prefill = scheduler.schedule()
        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [seq])
        scheduler.postprocess(scheduled, [101], is_prefill=True)
        self.assertEqual(seq.status, SequenceStatus.RUNNING)
        return seq

    def install_running_sequence(self, scheduler: Scheduler) -> Sequence:
        seq = sequence(2)
        num_cached_blocks = scheduler.block_manager.can_allocate(seq)
        self.assertGreaterEqual(num_cached_blocks, 0)
        scheduler.block_manager.allocate(seq, num_cached_blocks)
        seq.num_cached_tokens = seq.num_tokens
        seq.status = SequenceStatus.RUNNING
        seq.is_prefill = False
        scheduler.running.append(seq)
        return seq

    def test_decode_batch_respects_global_token_budget(self) -> None:
        scheduler = Scheduler(
            scheduler_config(
                max_num_batched_tokens=2,
                max_num_seqs=4,
                scheduler_policy="decode_first_budgeted",
                decode_token_budget=4,
            )
        )
        running = [
            self.install_running_sequence(scheduler) for _ in range(4)
        ]

        scheduled, is_prefill = scheduler.schedule()

        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, running[:2])
        self.assertEqual(
            sum(seq.num_scheduled_tokens for seq in scheduled),
            2,
        )

    def test_decode_batch_respects_policy_token_budget(self) -> None:
        scheduler = Scheduler(
            scheduler_config(
                max_num_batched_tokens=4,
                max_num_seqs=4,
                scheduler_policy="decode_first_budgeted",
                decode_token_budget=1,
            )
        )
        running = [
            self.install_running_sequence(scheduler) for _ in range(4)
        ]

        scheduled, is_prefill = scheduler.schedule()

        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, running[:1])

    def test_prefill_first_takes_waiting_before_running_decode(self) -> None:
        scheduler = Scheduler(scheduler_config())
        running = self.make_running_sequence(scheduler)
        waiting = sequence(8)
        scheduler.add(waiting)

        scheduled, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [waiting])
        self.assertIn(running, scheduler.running)

    def test_first_waiting_sequence_can_use_chunked_prefill(self) -> None:
        scheduler = Scheduler(scheduler_config(max_num_batched_tokens=4))
        waiting = sequence(10)
        scheduler.add(waiting)

        first_chunk, is_prefill = scheduler.schedule()
        self.assertTrue(is_prefill)
        self.assertEqual(first_chunk, [waiting])
        self.assertEqual(waiting.num_scheduled_tokens, 4)
        self.assertEqual(waiting.status, SequenceStatus.WAITING)
        scheduler.postprocess(first_chunk, [101], is_prefill=True)
        self.assertEqual(waiting.num_cached_tokens, 4)

        second_chunk, is_prefill = scheduler.schedule()
        self.assertTrue(is_prefill)
        self.assertEqual(second_chunk, [waiting])
        self.assertEqual(waiting.num_scheduled_tokens, 4)
        self.assertEqual(waiting.status, SequenceStatus.WAITING)

    def test_decode_runs_when_no_prefill_is_waiting(self) -> None:
        scheduler = Scheduler(scheduler_config())
        running = self.make_running_sequence(scheduler)

        scheduled, is_prefill = scheduler.schedule()

        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, [running])
        self.assertEqual(running.num_scheduled_tokens, 1)

    def test_kv_usage_peak_and_reset_are_measured_from_allocations(self) -> None:
        manager = BlockManager(num_blocks=4, block_size=4)
        first = sequence(4)
        second = sequence(4)

        manager.allocate(first, num_cached_blocks=0)
        manager.allocate(second, num_cached_blocks=0)
        manager.deallocate(first)

        self.assertEqual(
            manager.observability_snapshot(),
            {
                "total_blocks": 4,
                "final_used_blocks": 1,
                "final_free_blocks": 3,
                "peak_used_blocks": 2,
                "peak_utilization": 0.5,
            },
        )

        manager.reset_observability()
        self.assertEqual(
            manager.observability_snapshot()["peak_used_blocks"],
            1,
        )

    def test_preemption_counter_counts_repeated_successful_preemptions(
        self,
    ) -> None:
        scheduler = Scheduler(scheduler_config(num_blocks=2))
        seq = sequence(2)

        for _ in range(2):
            scheduler.block_manager.allocate(seq, num_cached_blocks=0)
            scheduler.preempt(seq)

        self.assertEqual(
            scheduler.observability_snapshot()["preemption_count"],
            2,
        )
        scheduler.reset_observability()
        self.assertEqual(
            scheduler.observability_snapshot()["preemption_count"],
            0,
        )

    def test_budgeted_decode_runs_before_waiting_prefill(self) -> None:
        scheduler = Scheduler(
            scheduler_config(
                scheduler_policy="decode_first_budgeted",
                decode_token_budget=1,
            )
        )
        running = self.make_running_sequence(scheduler)
        waiting = sequence(8)
        scheduler.add(waiting)

        scheduled, is_prefill = scheduler.schedule()

        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, [running])
        self.assertEqual(running.num_scheduled_tokens, 1)
        self.assertIn(waiting, scheduler.waiting)

    def test_budget_exhaustion_forces_prefill_progress(self) -> None:
        scheduler = Scheduler(
            scheduler_config(
                scheduler_policy="decode_first_budgeted",
                decode_token_budget=1,
            )
        )
        running = self.make_running_sequence(scheduler)
        waiting = sequence(8)
        scheduler.add(waiting)

        decoded, is_prefill = scheduler.schedule()
        self.assertFalse(is_prefill)
        scheduler.postprocess(decoded, [102], is_prefill=False)
        scheduled, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [waiting])
        self.assertIn(running, scheduler.running)

    def test_decode_step_guard_bounds_waiting_to_two_decode_steps(self) -> None:
        scheduler = Scheduler(
            scheduler_config(
                scheduler_policy="decode_first_budgeted",
                decode_token_budget=8,
                decode_step_guard=2,
            )
        )
        first = sequence(2)
        second = sequence(2)
        scheduler.add(first)
        scheduler.add(second)
        running, is_prefill = scheduler.schedule()
        scheduler.postprocess(running, [101, 102], is_prefill=True)
        waiting = sequence(8)
        scheduler.add(waiting)

        for token_ids in ([103, 104], [105, 106]):
            decoded, is_prefill = scheduler.schedule()
            self.assertFalse(is_prefill)
            self.assertEqual(len(decoded), 2)
            scheduler.postprocess(decoded, token_ids, is_prefill=False)

        scheduled, is_prefill = scheduler.schedule()
        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [waiting])

    def test_disabled_decode_step_guard_preserves_token_budget(self) -> None:
        explicit = scheduler_config(
            scheduler_policy="decode_first_budgeted",
            decode_token_budget=3,
            decode_step_guard=0,
        )
        legacy = scheduler_config(
            scheduler_policy="decode_first_budgeted",
            decode_token_budget=3,
        )
        del legacy.decode_step_guard

        for config in (explicit, legacy):
            with self.subTest(has_guard=hasattr(config, "decode_step_guard")):
                scheduler = Scheduler(config)
                running = self.make_running_sequence(scheduler)
                waiting = sequence(8)
                scheduler.add(waiting)

                for token_id in (102, 103, 104):
                    decoded, is_prefill = scheduler.schedule()
                    self.assertFalse(is_prefill)
                    self.assertEqual(decoded, [running])
                    scheduler.postprocess(
                        decoded,
                        [token_id],
                        is_prefill=False,
                    )

                scheduled, is_prefill = scheduler.schedule()
                self.assertTrue(is_prefill)
                self.assertEqual(scheduled, [waiting])

    def test_token_budget_can_trigger_before_decode_step_guard(self) -> None:
        scheduler = Scheduler(
            scheduler_config(
                scheduler_policy="decode_first_budgeted",
                decode_token_budget=2,
                decode_step_guard=8,
            )
        )
        running = self.make_running_sequence(scheduler)
        waiting = sequence(8)
        scheduler.add(waiting)

        for token_id in (102, 103):
            decoded, is_prefill = scheduler.schedule()
            self.assertFalse(is_prefill)
            self.assertEqual(decoded, [running])
            scheduler.postprocess(decoded, [token_id], is_prefill=False)

        scheduled, is_prefill = scheduler.schedule()
        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [waiting])

    def test_dynamic_waiting_arrival_starts_a_fresh_step_guard(self) -> None:
        scheduler = Scheduler(
            scheduler_config(
                scheduler_policy="decode_first_budgeted",
                decode_token_budget=8,
                decode_step_guard=1,
            )
        )
        running = self.make_running_sequence(scheduler)
        for token_id in (102, 103, 104):
            decoded, is_prefill = scheduler.schedule()
            scheduler.postprocess(decoded, [token_id], is_prefill=False)
        waiting = sequence(8)
        scheduler.add(waiting)

        decoded, is_prefill = scheduler.schedule()
        self.assertFalse(is_prefill)
        self.assertEqual(decoded, [running])
        scheduler.postprocess(decoded, [105], is_prefill=False)
        scheduled, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [waiting])

    def test_decode_step_guard_resets_after_partial_prefill_progress(self) -> None:
        scheduler = Scheduler(
            scheduler_config(
                scheduler_policy="decode_first_budgeted",
                decode_token_budget=8,
                decode_step_guard=1,
            )
        )
        running = self.make_running_sequence(scheduler)
        waiting = sequence(10)
        scheduler.add(waiting)

        decoded, is_prefill = scheduler.schedule()
        scheduler.postprocess(decoded, [102], is_prefill=False)
        first_chunk, is_prefill = scheduler.schedule()
        self.assertTrue(is_prefill)
        scheduler.postprocess(first_chunk, [103], is_prefill=True)
        decoded_after_progress, is_prefill = scheduler.schedule()

        self.assertFalse(is_prefill)
        self.assertEqual(decoded_after_progress, [running])

    def test_partial_decode_budget_rotates_running_sequences(self) -> None:
        scheduler = Scheduler(
            scheduler_config(
                scheduler_policy="decode_first_budgeted",
                decode_token_budget=1,
            )
        )
        first = sequence(2)
        second = sequence(2)
        scheduler.add(first)
        scheduler.add(second)
        scheduled, is_prefill = scheduler.schedule()
        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [first, second])
        scheduler.postprocess(scheduled, [101, 102], is_prefill=True)
        waiting = sequence(8)
        scheduler.add(waiting)

        decoded, is_prefill = scheduler.schedule()
        self.assertFalse(is_prefill)
        self.assertEqual(decoded, [first])
        scheduler.postprocess(decoded, [103], is_prefill=False)
        prefilled, is_prefill = scheduler.schedule()
        self.assertTrue(is_prefill)
        scheduler.postprocess(prefilled, [104], is_prefill=True)
        next_decoded, is_prefill = scheduler.schedule()

        self.assertFalse(is_prefill)
        self.assertEqual(next_decoded[0], second)

    def test_scheduler_rejects_invalid_policy_budget_pairs(self) -> None:
        invalid_configs = (
            scheduler_config(
                scheduler_policy=[],
                decode_token_budget=0,
            ),
            scheduler_config(
                scheduler_policy="decode_first_budgeted",
                decode_token_budget=True,
            ),
            scheduler_config(
                scheduler_policy="prefill_first",
                decode_token_budget=1,
            ),
            scheduler_config(
                scheduler_policy="prefill_first",
                decode_step_guard=1,
            ),
            scheduler_config(
                scheduler_policy="decode_first_budgeted",
                decode_token_budget=1,
                decode_step_guard=True,
            ),
            scheduler_config(
                scheduler_policy="decode_first_budgeted",
                decode_token_budget=1,
                decode_step_guard=-1,
            ),
        )

        for config in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaisesRegex(
                    ValueError,
                    "policy|decode_token_budget|decode_step_guard",
                ):
                    Scheduler(config)

    def test_forced_prefill_releases_kv_instead_of_overrunning_budget(
        self,
    ) -> None:
        scheduler = Scheduler(
            scheduler_config(
                max_num_batched_tokens=8,
                num_blocks=2,
                scheduler_policy="decode_first_budgeted",
                decode_token_budget=1,
            )
        )
        running = self.make_running_sequence(scheduler)
        waiting = sequence(8)
        scheduler.add(waiting)

        decoded, is_prefill = scheduler.schedule()
        self.assertFalse(is_prefill)
        scheduler.postprocess(decoded, [102], is_prefill=False)
        scheduled, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [waiting])
        self.assertEqual(running.status, SequenceStatus.WAITING)

    def test_step_guard_forced_prefill_releases_kv(self) -> None:
        scheduler = Scheduler(
            scheduler_config(
                max_num_batched_tokens=8,
                num_blocks=2,
                scheduler_policy="decode_first_budgeted",
                decode_token_budget=8,
                decode_step_guard=1,
            )
        )
        running = self.make_running_sequence(scheduler)
        waiting = sequence(8)
        scheduler.add(waiting)

        decoded, is_prefill = scheduler.schedule()
        scheduler.postprocess(decoded, [102], is_prefill=False)
        scheduled, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [waiting])
        self.assertEqual(running.status, SequenceStatus.WAITING)

    def test_budgeted_decode_rotates_when_waiting_queue_is_empty(self) -> None:
        scheduler = Scheduler(
            scheduler_config(
                max_num_seqs=2,
                scheduler_policy="decode_first_budgeted",
                decode_token_budget=1,
            )
        )
        first = sequence(2)
        second = sequence(2)
        scheduler.add(first)
        scheduler.add(second)
        scheduled, is_prefill = scheduler.schedule()
        scheduler.postprocess(scheduled, [101, 102], is_prefill)
        third = sequence(2)
        scheduler.add(third)
        decoded, is_prefill = scheduler.schedule()
        scheduler.postprocess(decoded, [103], is_prefill)
        prefilled, is_prefill = scheduler.schedule()
        scheduler.postprocess(prefilled, [104], is_prefill)

        first_batch, is_prefill = scheduler.schedule()
        self.assertFalse(is_prefill)
        self.assertEqual(len(first_batch), 1)
        scheduler.postprocess(first_batch, [105], is_prefill)
        second_batch, is_prefill = scheduler.schedule()
        self.assertEqual(len(second_batch), 1)
        scheduler.postprocess(second_batch, [106], is_prefill)
        third_batch, is_prefill = scheduler.schedule()

        self.assertFalse(is_prefill)
        self.assertEqual(third_batch, [third])

    def test_kv_decode_preemption_preserves_existing_waiting_head(
        self,
    ) -> None:
        scheduler = Scheduler(
            scheduler_config(
                num_blocks=1,
                scheduler_policy="decode_first_budgeted",
                decode_token_budget=2,
            )
        )
        running = self.make_running_sequence(scheduler)
        for token_id in (102, 103):
            decoded, is_prefill = scheduler.schedule()
            self.assertFalse(is_prefill)
            scheduler.postprocess(decoded, [token_id], is_prefill)
        waiting = sequence(2)
        scheduler.add(waiting)

        scheduled, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [waiting])
        self.assertEqual(running.status, SequenceStatus.WAITING)

    def test_request_larger_than_kv_capacity_has_descriptive_error(
        self,
    ) -> None:
        scheduler = Scheduler(
            scheduler_config(
                num_blocks=1,
                scheduler_policy="decode_first_budgeted",
                decode_token_budget=1,
            )
        )
        scheduler.add(sequence(8))

        with self.assertRaisesRegex(RuntimeError, "cannot make progress"):
            scheduler.schedule()

    def test_block_manager_reuses_complete_prefix_block(self) -> None:
        manager = BlockManager(num_blocks=8, block_size=4)
        source = sequence(8)
        self.assertEqual(manager.can_allocate(source), 0)
        manager.allocate(source, num_cached_blocks=0)
        source.num_scheduled_tokens = source.num_tokens
        manager.hash_blocks(source)
        cached_block_id = source.block_table[0]
        manager.deallocate(source)

        repeated = sequence(8)
        cached_blocks = manager.can_allocate(repeated)
        manager.allocate(repeated, cached_blocks)

        self.assertEqual(cached_blocks, 1)
        self.assertEqual(repeated.block_table[0], cached_block_id)
        self.assertEqual(repeated.num_cached_tokens, 4)


if __name__ == "__main__":
    unittest.main()
