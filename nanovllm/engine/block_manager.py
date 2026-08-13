from collections import deque
from dataclasses import dataclass

import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


@dataclass(frozen=True, slots=True)
class PrefixCachePreview:
    full_required_blocks: int
    matched_prefix_blocks: int
    active_shared_blocks: int
    inactive_cached_blocks: int
    matched_prefix_tokens: int
    incremental_reservation_blocks: int
    matched_block_ids: tuple[int, ...] = ()
    active_block_ids: tuple[int, ...] = ()
    inactive_block_ids: tuple[int, ...] = ()


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        self.peak_used_blocks = 0

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        self._record_usage()
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def _cached_prefix(self, seq: Sequence) -> tuple[int, int]:
        h = -1
        num_cached_blocks = 0
        active_cached_blocks = 0
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                active_cached_blocks += 1
        return num_cached_blocks, active_cached_blocks

    def num_cached_blocks(self, seq: Sequence) -> int:
        return self._cached_prefix(seq)[0]

    def can_allocate(
        self,
        seq: Sequence,
        materialized_tokens: int | None = None,
    ) -> int:
        num_cached_blocks, active_cached_blocks = self._cached_prefix(seq)
        required_blocks = (
            seq.num_blocks
            if materialized_tokens is None
            else (materialized_tokens + self.block_size - 1) // self.block_size
        )
        num_cached_blocks = min(num_cached_blocks, required_blocks)
        num_new_blocks = required_blocks
        num_new_blocks -= min(active_cached_blocks, num_cached_blocks)
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def preview_prefix_cache(
        self,
        token_ids: list[int],
        max_tokens: int,
    ) -> PrefixCachePreview:
        if not token_ids:
            raise ValueError("Prefix cache preview requires prompt tokens")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
            raise ValueError("Prefix cache preview requires integer max_tokens")
        if max_tokens <= 0:
            raise ValueError("Prefix cache preview requires positive max_tokens")
        cache_tokens = len(token_ids) + max_tokens - 1
        full_required_blocks = (
            cache_tokens + self.block_size - 1
        ) // self.block_size
        matched_prefix_blocks = 0
        active_shared_blocks = 0
        inactive_cached_blocks = 0
        matched_block_ids: list[int] = []
        active_block_ids: list[int] = []
        inactive_block_ids: list[int] = []
        prefix_hash = -1
        # Prefix reuse is decided at admission time, when the sequence only
        # contains its prompt. Keep this upper bound identical to
        # ``can_allocate(Sequence(prompt, ...))``: its final prompt block is
        # deliberately excluded because generation may append into it.
        prompt_blocks = (
            len(token_ids) + self.block_size - 1
        ) // self.block_size
        for index in range(prompt_blocks - 1):
            start = index * self.block_size
            end = start + self.block_size
            if end > len(token_ids):
                break
            block_tokens = token_ids[start:end]
            prefix_hash = self.compute_hash(block_tokens, prefix_hash)
            block_id = self.hash_to_block_id.get(prefix_hash, -1)
            if (
                block_id == -1
                or self.blocks[block_id].token_ids != block_tokens
            ):
                break
            matched_prefix_blocks += 1
            matched_block_ids.append(block_id)
            if block_id in self.used_block_ids:
                active_shared_blocks += 1
                active_block_ids.append(block_id)
            else:
                inactive_cached_blocks += 1
                inactive_block_ids.append(block_id)
        return PrefixCachePreview(
            full_required_blocks=full_required_blocks,
            matched_prefix_blocks=matched_prefix_blocks,
            active_shared_blocks=active_shared_blocks,
            inactive_cached_blocks=inactive_cached_blocks,
            matched_prefix_tokens=matched_prefix_blocks * self.block_size,
            incremental_reservation_blocks=(
                full_required_blocks - active_shared_blocks
            ),
            matched_block_ids=tuple(matched_block_ids),
            active_block_ids=tuple(active_block_ids),
            inactive_block_ids=tuple(inactive_block_ids),
        )

    def cache_state_snapshot(self) -> list[dict[str, object]]:
        snapshots = []
        for block in self.blocks:
            if (
                block.hash != -1
                and self.hash_to_block_id.get(block.hash) == block.block_id
            ):
                snapshots.append(
                    {
                        "block_id": block.block_id,
                        "hash": block.hash,
                        "token_ids": list(block.token_ids),
                        "used": block.block_id in self.used_block_ids,
                        "ref_count": block.ref_count,
                    }
                )
        return snapshots

    def allocate(
        self,
        seq: Sequence,
        num_cached_blocks: int,
        materialized_tokens: int | None = None,
    ):
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        required_blocks = (
            seq.num_blocks
            if materialized_tokens is None
            else (materialized_tokens + self.block_size - 1) // self.block_size
        )
        for i in range(num_cached_blocks, required_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size
        self._record_usage()

    def can_materialize(self, seq: Sequence, end_tokens: int) -> bool:
        required = (end_tokens + self.block_size - 1) // self.block_size
        return required - len(seq.block_table) <= len(self.free_block_ids)

    def materialize(self, seq: Sequence, end_tokens: int) -> None:
        required = (end_tokens + self.block_size - 1) // self.block_size
        while len(seq.block_table) < required:
            seq.block_table.append(self._allocate_block())

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id

    def _record_usage(self):
        self.peak_used_blocks = max(
            self.peak_used_blocks,
            len(self.used_block_ids),
        )

    def current_utilization(self) -> float:
        total_blocks = len(self.blocks)
        return len(self.used_block_ids) / total_blocks

    def reset_observability(self):
        self.peak_used_blocks = len(self.used_block_ids)

    def observability_snapshot(self) -> dict[str, int | float]:
        total_blocks = len(self.blocks)
        return {
            "total_blocks": total_blocks,
            "final_used_blocks": len(self.used_block_ids),
            "final_free_blocks": len(self.free_block_ids),
            "peak_used_blocks": self.peak_used_blocks,
            "peak_utilization": self.peak_used_blocks / total_blocks,
        }
