from __future__ import annotations

from dataclasses import dataclass

import torch

from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


@dataclass(frozen=True, slots=True)
class PrefillTransferMetadata:
    prompt_token_ids: tuple[int, ...]
    first_token_id: int
    max_tokens: int
    temperature: float
    ignore_eos: bool
    materialized_tokens: int
    source_block_ids: tuple[int, ...]


def pack_materialized_kv(
    kv_cache: torch.Tensor,
    *,
    block_table: list[int],
    materialized_tokens: int,
) -> torch.Tensor:
    block_size = kv_cache.shape[3]
    pieces = []
    remaining = materialized_tokens
    for block_id in block_table:
        count = min(block_size, remaining)
        if count <= 0:
            break
        pieces.append(kv_cache[:, :, block_id, :count])
        remaining -= count
    if remaining != 0:
        raise ValueError("block_table does not cover materialized KV")
    return torch.cat(pieces, dim=2).contiguous()


def install_materialized_kv(
    kv_cache: torch.Tensor,
    *,
    block_table: list[int],
    packed: torch.Tensor,
) -> None:
    block_size = kv_cache.shape[3]
    offset = 0
    materialized_tokens = packed.shape[2]
    for block_id in block_table:
        count = min(block_size, materialized_tokens - offset)
        if count <= 0:
            break
        kv_cache[:, :, block_id, :count].copy_(packed[:, :, offset:offset + count])
        offset += count
    if offset != materialized_tokens:
        raise ValueError("destination block_table does not cover transferred KV")


def restore_decode_sequence(
    *,
    prompt_token_ids: list[int],
    first_token_id: int,
    sampling_params: SamplingParams,
) -> Sequence:
    seq = Sequence(prompt_token_ids, sampling_params)
    seq.append_token(first_token_id)
    seq.num_cached_tokens = len(prompt_token_ids)
    seq.is_prefill = False
    seq.status = SequenceStatus.RUNNING
    return seq
