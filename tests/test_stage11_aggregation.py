from __future__ import annotations

import copy
import unittest

from experiments.aggregate_stage11 import (
    load_trace_from_document,
    validate_events,
)
from experiments.benchmark.open_loop import prepare_requests
from nanovllm.engine.block_manager import BlockManager


def admission_event(
    index: int,
    action: str,
    request_id: str,
    arrival_at: float,
    observed_at: float,
    reservation: int,
    reserved_after: int,
) -> dict:
    return {
        "schema_version": 2,
        "event_index": index,
        "action": action,
        "request_id": request_id,
        "arrival_at": arrival_at,
        "observed_at": observed_at,
        "queue_wait_ms": None if action == "released" else (observed_at - arrival_at) * 1000,
        "required_blocks": 3,
        "reservation_blocks": reservation,
        "reserved_blocks_after": reserved_after,
        "reason": None,
    }


def cache_event(
    index: int,
    action: str,
    request_id: str,
    observed_at: float,
    matched: int,
    active: int,
    increment: int,
    reserved_after: int,
    state_index: int,
    matched_block_ids: list[int] | None = None,
) -> dict:
    matched_block_ids = [] if matched_block_ids is None else matched_block_ids
    return {
        "schema_version": 2,
        "event_index": index,
        "action": action,
        "request_id": request_id,
        "observed_at": observed_at,
        "full_required_blocks": 3,
        "matched_prefix_blocks": matched,
        "active_shared_blocks": active,
        "inactive_cached_blocks": matched - active,
        "matched_prefix_tokens": matched * 4,
        "incremental_reservation_blocks": increment,
        "matched_block_ids": matched_block_ids,
        "active_block_ids": matched_block_ids[:active],
        "inactive_block_ids": matched_block_ids[active:],
        "cache_state_index": state_index,
        "reserved_blocks_after": reserved_after,
        "reason": None,
    }


class Stage11AggregationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trace = {
            "schema_version": 2,
            "description": "aggregation replay fixture",
            "time_unit": "seconds",
            "requests": [
                {
                    "request_id": "long-000",
                    "request_class": "long",
                    "arrival_time": 0.0,
                    "prompt_length": 8,
                    "max_output_tokens": 2,
                    "seed": 1,
                    "prefix_group": "system-prefix",
                    "shared_prefix_length": 4,
                },
                {
                    "request_id": "interactive-000",
                    "request_class": "interactive",
                    "arrival_time": 0.1,
                    "prompt_length": 8,
                    "max_output_tokens": 2,
                    "seed": 2,
                    "prefix_group": "system-prefix",
                    "shared_prefix_length": 4,
                },
            ],
        }
        self.requests = [
            {
                "request_id": "long-000",
                "terminal_state": "Finished",
                "admitted_at": 0.0,
                "terminal_at": 0.3,
            },
            {
                "request_id": "interactive-000",
                "terminal_state": "Finished",
                "admitted_at": 0.1,
                "terminal_at": 0.2,
            },
        ]
        self.admission = {
            "schema_version": 2,
            "policy": "prefix_aware_fifo",
            "total_kv_blocks": 5,
            "max_queue_wait_seconds": 1.0,
            "observe_prefix_cache": True,
            "admitted_requests": 2,
            "rejected_requests": 0,
            "max_observed_queue_wait_ms": 0.0,
            "peak_reserved_blocks": 5,
            "final_reserved_blocks": 0,
        }
        self.events = [
            admission_event(0, "admitted", "long-000", 0.0, 0.0, 3, 3),
            admission_event(1, "admitted", "interactive-000", 0.1, 0.1, 2, 5),
            admission_event(2, "released", "interactive-000", 0.1, 0.2, 2, 3),
            admission_event(3, "released", "long-000", 0.0, 0.3, 3, 0),
        ]
        prepared = {
            request.spec.request_id: request.prompt_token_ids
            for request in prepare_requests(
                load_trace_from_document(self.trace), token_id_upper_bound=10000
            )
        }
        prefix_tokens = prepared["long-000"][:4]
        prefix_hash = BlockManager.compute_hash(prefix_tokens)
        self.cache_states = [
            {
                "schema_version": 1,
                "state_index": 0,
                "observed_at": 0.0,
                "blocks": [],
            },
            {
                "schema_version": 1,
                "state_index": 1,
                "observed_at": 0.1,
                "blocks": [
                    {
                        "block_id": 7,
                        "hash": prefix_hash,
                        "token_ids": prefix_tokens,
                        "used": True,
                        "ref_count": 1,
                    }
                ],
            },
        ]
        self.cache = [
            cache_event(0, "admitted", "long-000", 0.0, 0, 0, 3, 3, 0),
            cache_event(1, "admitted", "interactive-000", 0.1, 1, 1, 2, 5, 1, [7]),
            cache_event(2, "released", "interactive-000", 0.2, 1, 1, 2, 3, 1, [7]),
            cache_event(3, "released", "long-000", 0.3, 0, 0, 3, 0, 0),
        ]

    def test_trace_allows_variable_shared_prefix_lengths_in_one_group(self) -> None:
        trace = copy.deepcopy(self.trace)
        trace["requests"][1]["shared_prefix_length"] = 6
        prepared = prepare_requests(
            load_trace_from_document(trace), token_id_upper_bound=10000
        )
        self.assertEqual(
            prepared[0].prompt_token_ids[:4], prepared[1].prompt_token_ids[:4]
        )

    def replay(self, events: list[dict], cache: list[dict]) -> dict:
        return validate_events(
            trace=self.trace,
            requests=self.requests,
            admission_events=events,
            cache_events=cache,
            cache_states=self.cache_states,
            admission=self.admission,
            policy="prefix_aware_fifo",
            block_size=4,
        )

    def test_prefix_cache_and_admission_events_replay(self) -> None:
        self.assertEqual(
            self.replay(self.events, self.cache),
            {
                "admitted_requests": 2,
                "rejected_requests": 0,
                "peak_reserved_blocks": 5,
                "max_observed_queue_wait_ms": 0.0,
            },
        )

    def test_cache_tampering_is_rejected(self) -> None:
        bad_cache = copy.deepcopy(self.cache)
        bad_cache[1]["active_shared_blocks"] = 0
        bad_cache[1]["inactive_cached_blocks"] = 1
        bad_cache[1]["incremental_reservation_blocks"] = 3

        with self.assertRaises(ValueError):
            self.replay(self.events, bad_cache)

    def test_orphan_cache_event_is_rejected(self) -> None:
        bad_cache = copy.deepcopy(self.cache)
        bad_cache.pop()

        with self.assertRaises(ValueError):
            self.replay(self.events, bad_cache)

    def test_active_hit_without_used_cache_block_is_rejected(self) -> None:
        bad_states = copy.deepcopy(self.cache_states)
        bad_states[1]["blocks"][0]["used"] = False
        bad_states[1]["blocks"][0]["ref_count"] = 0

        with self.assertRaises(ValueError):
            validate_events(
                trace=self.trace,
                requests=self.requests,
                admission_events=self.events,
                cache_events=self.cache,
                cache_states=bad_states,
                admission=self.admission,
                policy="prefix_aware_fifo",
                block_size=4,
            )

    def test_omitted_real_prefix_hit_is_rejected(self) -> None:
        bad_cache = copy.deepcopy(self.cache)
        bad_events = copy.deepcopy(self.events)
        bad_admission = copy.deepcopy(self.admission)
        bad_admission["total_kv_blocks"] = 6
        for event, reserved_after in ((bad_events[1], 6), (bad_events[2], 3)):
            event["reservation_blocks"] = 3
            event["reserved_blocks_after"] = reserved_after
        for index in (1, 2):
            bad_cache[index].update(
                {
                    "matched_prefix_blocks": 0,
                    "active_shared_blocks": 0,
                    "inactive_cached_blocks": 0,
                    "matched_prefix_tokens": 0,
                    "incremental_reservation_blocks": 3,
                    "matched_block_ids": [],
                    "active_block_ids": [],
                    "inactive_block_ids": [],
                }
            )
            bad_cache[index]["reserved_blocks_after"] = (6 if index == 1 else 3)

        with self.assertRaises(ValueError):
            validate_events(
                trace=self.trace,
                requests=self.requests,
                admission_events=bad_events,
                cache_events=bad_cache,
                cache_states=self.cache_states,
                admission=bad_admission,
                policy="prefix_aware_fifo",
                block_size=4,
            )

    def test_release_must_reference_its_admission_snapshot(self) -> None:
        bad_states = copy.deepcopy(self.cache_states)
        duplicate = copy.deepcopy(bad_states[1])
        duplicate["state_index"] = 2
        duplicate["observed_at"] = 0.15
        bad_states.append(duplicate)
        bad_cache = copy.deepcopy(self.cache)
        bad_cache[2]["cache_state_index"] = 2

        with self.assertRaises(ValueError):
            validate_events(
                trace=self.trace,
                requests=self.requests,
                admission_events=self.events,
                cache_events=bad_cache,
                cache_states=bad_states,
                admission=self.admission,
                policy="prefix_aware_fifo",
                block_size=4,
            )


if __name__ == "__main__":
    unittest.main()
