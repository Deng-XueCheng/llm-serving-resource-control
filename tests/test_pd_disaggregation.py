from __future__ import annotations

import unittest

import torch

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.pd_disaggregation import (
    install_materialized_kv,
    pack_materialized_kv,
    restore_decode_sequence,
)


class PdDisaggregationTests(unittest.TestCase):
    def test_destination_uses_new_block_ids_and_restores_decode_state(self):
        source = BlockManager(6, 4)
        seq = Sequence(list(range(6)), SamplingParams(max_tokens=3))
        source.allocate(seq, 0)
        seq.num_cached_tokens = 6
        source_ids = tuple(seq.block_table)

        destination = BlockManager(6, 4)
        destination.free_block_ids.rotate(-2)
        restored = restore_decode_sequence(
            prompt_token_ids=list(range(6)),
            first_token_id=99,
            sampling_params=SamplingParams(max_tokens=3),
        )
        destination.allocate(restored, 0)
        restored.num_cached_tokens = 6

        self.assertNotEqual(tuple(restored.block_table), source_ids)
        self.assertEqual(restored.last_token, 99)
        self.assertEqual(restored.num_prompt_tokens, 6)
        self.assertEqual(restored.num_completion_tokens, 1)
        self.assertEqual(restored.status, SequenceStatus.RUNNING)

    def test_logical_kv_pack_and_install_handles_full_and_tail_blocks(self):
        block_size = 4
        cache = torch.arange(2 * 2 * 5 * block_size * 1 * 2).reshape(
            2, 2, 5, block_size, 1, 2
        )
        packed = pack_materialized_kv(
            cache, block_table=[3, 1], materialized_tokens=6
        )
        destination = torch.zeros_like(cache)
        install_materialized_kv(
            destination, block_table=[0, 4], packed=packed
        )
        self.assertTrue(torch.equal(destination[:, :, 0], cache[:, :, 3]))
        self.assertTrue(torch.equal(destination[:, :, 4, :2], cache[:, :, 1, :2]))
        self.assertEqual(packed.numel() * packed.element_size(), 48 * 8)

    def test_source_and_destination_release_without_leak(self):
        source = BlockManager(4, 4)
        destination = BlockManager(4, 4)
        source_seq = Sequence([1, 2, 3, 4, 5])
        destination_seq = Sequence([1, 2, 3, 4, 5])
        source.allocate(source_seq, 0)
        destination.allocate(destination_seq, 0)
        source.deallocate(source_seq)
        destination.deallocate(destination_seq)
        self.assertEqual(len(source.free_block_ids), 4)
        self.assertEqual(len(destination.free_block_ids), 4)


if __name__ == "__main__":
    unittest.main()
