from __future__ import annotations

import unittest
from types import SimpleNamespace

from nanovllm import SamplingParams
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence


def config(*, budget=4, blocks=8):
    return SimpleNamespace(
        max_num_batched_tokens=budget,
        max_num_seqs=4,
        eos=-1,
        kvcache_block_size=4,
        num_kvcache_blocks=blocks,
        scheduler_policy="mixed_token_budget",
        decode_token_budget=budget,
        decode_step_guard=0,
        incremental_kv_allocation=True,
    )


def sequence(tokens, *, max_tokens=2, offset=0):
    return Sequence(
        list(range(offset + 1, offset + tokens + 1)),
        SamplingParams(temperature=1.0, max_tokens=max_tokens, ignore_eos=True),
    )


class IncrementalKvAllocationTests(unittest.TestCase):
    def setUp(self):
        Sequence.block_size = 4

    def execute(self, scheduler, batch):
        sampled = {
            item.sequence.seq_id: 1000 + item.sequence.seq_id
            for item in batch.items if item.requires_sampling
        }
        scheduler.postprocess_mixed(batch, sampled)

    def test_partial_prefill_block_table_grows_with_chunks(self):
        scheduler = Scheduler(config(budget=4))
        seq = sequence(10)
        scheduler.add(seq)

        first = scheduler.schedule()
        self.assertEqual(len(seq.block_table), 1)
        self.execute(scheduler, first)
        second = scheduler.schedule()
        self.assertEqual(len(seq.block_table), 2)
        self.execute(scheduler, second)
        third = scheduler.schedule()
        self.assertEqual(len(seq.block_table), 3)

    def test_small_chunk_is_admitted_when_full_prompt_does_not_fit(self):
        scheduler = Scheduler(config(budget=4, blocks=1))
        seq = sequence(12)
        scheduler.add(seq)

        batch = scheduler.schedule()

        self.assertEqual(batch.items[0].num_scheduled_tokens, 4)
        self.assertEqual(len(seq.block_table), 1)

    def test_prefix_hit_plus_partial_tail_allocates_incrementally(self):
        scheduler = Scheduler(config(budget=8, blocks=4))
        first = sequence(8, max_tokens=1)
        scheduler.add(first)
        self.execute(scheduler, scheduler.schedule())
        second = sequence(12, max_tokens=1)
        scheduler.add(second)

        batch = scheduler.schedule()

        self.assertEqual(second.num_cached_tokens, 8)
        self.assertEqual(len(second.block_table), 3)
        self.assertEqual(batch.items[0].num_scheduled_tokens, 4)

    def test_preempt_resume_reallocates_only_recompute_chunk(self):
        scheduler = Scheduler(config(budget=4, blocks=3))
        seq = sequence(10)
        scheduler.add(seq)
        self.execute(scheduler, scheduler.schedule())
        scheduler.preempt(seq, reason="unit_test")

        resumed = scheduler.schedule()

        self.assertEqual(seq.num_cached_tokens, 4)
        self.assertEqual(len(seq.block_table), 2)
        self.assertEqual(resumed.items[0].num_scheduled_tokens, 4)

    def test_decode_growth_allocates_next_block(self):
        scheduler = Scheduler(config(budget=4, blocks=3))
        seq = sequence(4, max_tokens=2)
        scheduler.add(seq)
        self.execute(scheduler, scheduler.schedule())
        self.assertEqual(len(seq.block_table), 1)

        decode = scheduler.schedule()

        self.assertEqual(len(seq.block_table), 2)
        self.execute(scheduler, decode)
        self.assertEqual(scheduler.block_manager.current_utilization(), 0.0)


if __name__ == "__main__":
    unittest.main()
