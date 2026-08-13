from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
from collections import Counter
from math import isfinite
from pathlib import Path
from typing import Any

from experiments.aggregate_stage9 import validate_raw
from experiments.benchmark.open_loop import prepare_requests
from nanovllm.engine.block_manager import BlockManager
from experiments.generate_matrix import write_json_idempotent
from experiments.generate_stage11_configs import (
    CAPACITIES,
    CODE_PATHS,
    CONFIG_DIRECTORY,
    MANIFEST_PATH,
    POLICIES,
    REPO_ROOT,
    REUSE_RATIOS,
    SEEDS,
    build_trace,
    manifest,
    planned_configs,
)


RESULTS = REPO_ROOT / "experiments/results"
OUTPUT = RESULTS / "stage11_formal_r1_aggregate.json"
CODE_KEYS = tuple(CODE_PATHS)
ADMISSION_V1_KEYS = {
    "schema_version",
    "event_index",
    "action",
    "request_id",
    "arrival_at",
    "observed_at",
    "queue_wait_ms",
    "required_blocks",
    "reserved_blocks_after",
    "reason",
}
ADMISSION_V2_KEYS = ADMISSION_V1_KEYS | {"reservation_blocks"}
CACHE_KEYS = {
    "schema_version",
    "event_index",
    "action",
    "request_id",
    "observed_at",
    "full_required_blocks",
    "matched_prefix_blocks",
    "active_shared_blocks",
    "inactive_cached_blocks",
    "matched_prefix_tokens",
    "incremental_reservation_blocks",
    "matched_block_ids",
    "active_block_ids",
    "inactive_block_ids",
    "cache_state_index",
    "reserved_blocks_after",
    "reason",
}
CACHE_STATE_KEYS = {"schema_version", "state_index", "observed_at", "blocks"}
CACHE_BLOCK_KEYS = {"block_id", "hash", "token_ids", "used", "ref_count"}
SUMMARY_ADMISSION_V1_KEYS = {
    "schema_version",
    "policy",
    "total_kv_blocks",
    "max_queue_wait_seconds",
    "admitted_requests",
    "rejected_requests",
    "max_observed_queue_wait_ms",
    "peak_reserved_blocks",
    "final_reserved_blocks",
}
SUMMARY_ADMISSION_V2_KEYS = SUMMARY_ADMISSION_V1_KEYS | {
    "observe_prefix_cache"
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(float(value))
    )


def positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def required_blocks(spec: dict[str, Any], block_size: int) -> int:
    tokens = spec["prompt_length"] + spec["max_output_tokens"] - 1
    return (tokens + block_size - 1) // block_size


def validate_cache_event(
    event: dict[str, Any],
    *,
    index: int,
    action: str,
    request_id: str,
    observed_at: float,
    reserved_blocks_after: int,
    reason: str | None,
    expected_full_blocks: int,
    block_size: int,
    prompt_token_ids: list[int],
    cache_state: dict[str, Any],
    permit_prior_state: bool,
) -> None:
    if set(event) != CACHE_KEYS or event["schema_version"] != 2:
        raise ValueError("cache event schema differs")
    if event["event_index"] != index or event["action"] != action:
        raise ValueError("cache event ordering differs")
    if event["request_id"] != request_id or event["reason"] != reason:
        raise ValueError("cache event request binding differs")
    if not finite_number(event["observed_at"]) or (
        abs(event["observed_at"] - observed_at) > 1e-6
    ):
        raise ValueError("cache event timestamp differs")
    if event["cache_state_index"] != cache_state["state_index"] or (
        not permit_prior_state
        and abs(cache_state["observed_at"] - observed_at) > 1e-6
    ) or (permit_prior_state and cache_state["observed_at"] > observed_at):
        raise ValueError("cache state/event binding differs")
    integer_fields = (
        "full_required_blocks",
        "matched_prefix_blocks",
        "active_shared_blocks",
        "inactive_cached_blocks",
        "matched_prefix_tokens",
        "incremental_reservation_blocks",
        "reserved_blocks_after",
    )
    if any(not isinstance(event[field], int) or isinstance(event[field], bool) for field in integer_fields):
        raise ValueError("cache event integer field differs")
    if event["full_required_blocks"] != expected_full_blocks:
        raise ValueError("cache full footprint differs")
    matched = event["matched_prefix_blocks"]
    active = event["active_shared_blocks"]
    inactive = event["inactive_cached_blocks"]
    if (
        matched < 0
        or active < 0
        or inactive < 0
        or matched != active + inactive
        or matched >= expected_full_blocks
        or event["matched_prefix_tokens"] != matched * block_size
        or event["incremental_reservation_blocks"]
        != expected_full_blocks - active
        or event["incremental_reservation_blocks"] <= 0
        or event["reserved_blocks_after"] != reserved_blocks_after
    ):
        raise ValueError("cache event reservation contract differs")
    block_lists = (
        event["matched_block_ids"],
        event["active_block_ids"],
        event["inactive_block_ids"],
    )
    if any(
        not isinstance(value, list)
        or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
        for value in block_lists
    ):
        raise ValueError("cache block identifier schema differs")
    matched_ids, active_ids, inactive_ids = block_lists
    if (
        len(matched_ids) != matched
        or len(active_ids) != active
        or len(inactive_ids) != inactive
        or len(set(matched_ids)) != len(matched_ids)
        or set(active_ids) | set(inactive_ids) != set(matched_ids)
        or set(active_ids) & set(inactive_ids)
    ):
        raise ValueError("cache block identifier binding differs")
    state_hashes = {block["hash"]: block for block in cache_state["blocks"]}
    expected_matched_ids: list[int] = []
    expected_active_ids: list[int] = []
    expected_inactive_ids: list[int] = []
    prefix_hash = -1
    prompt_blocks = (len(prompt_token_ids) + block_size - 1) // block_size
    for index in range(prompt_blocks - 1):
        start = index * block_size
        end = start + block_size
        token_ids = prompt_token_ids[start:end]
        prefix_hash = BlockManager.compute_hash(token_ids, prefix_hash)
        block = state_hashes.get(prefix_hash)
        if block is None or block["token_ids"] != token_ids:
            break
        expected_matched_ids.append(block["block_id"])
        if block["used"]:
            expected_active_ids.append(block["block_id"])
        else:
            expected_inactive_ids.append(block["block_id"])
    if (
        matched_ids != expected_matched_ids
        or active_ids != expected_active_ids
        or inactive_ids != expected_inactive_ids
    ):
        raise ValueError("cache preview does not match state snapshot")


def validate_events(
    *,
    trace: dict[str, Any],
    requests: list[dict[str, Any]],
    admission_events: list[dict[str, Any]],
    cache_events: list[dict[str, Any]],
    cache_states: list[dict[str, Any]],
    admission: dict[str, Any],
    policy: str,
    block_size: int,
) -> dict[str, int | float]:
    if len(admission_events) != len(cache_events):
        raise ValueError("admission/cache event counts differ")
    specs = {item["request_id"]: item for item in trace["requests"]}
    raw = {item["request_id"]: item for item in requests}
    if set(specs) != set(raw) or len(specs) != len(trace["requests"]):
        raise ValueError("trace/raw request identifiers differ")
    prepared = {
        item.spec.request_id: item.prompt_token_ids
        for item in prepare_requests(
            load_trace_from_document(trace), token_id_upper_bound=10000
        )
    }
    arrivals = sorted(
        trace["requests"], key=lambda item: (item["arrival_time"], item["request_id"])
    )
    queue: list[str] = []
    active: set[str] = set()
    rejected: set[str] = set()
    cursor = 0
    reserved = 0
    peak_reserved = 0
    previous_time = -float("inf")
    waits: list[float] = []
    per_request: Counter[tuple[str, str]] = Counter()
    admission_preview: dict[str, dict[str, Any]] = {}
    state_by_index: dict[int, dict[str, Any]] = {}
    previous_state_time = -float("inf")
    for index, state in enumerate(cache_states):
        if set(state) != CACHE_STATE_KEYS or state.get("schema_version") != 1:
            raise ValueError("cache state schema differs")
        if state.get("state_index") != index or not finite_number(state.get("observed_at")):
            raise ValueError("cache state ordering differs")
        if state["observed_at"] < previous_state_time or not isinstance(state["blocks"], list):
            raise ValueError("cache state timestamp differs")
        previous_state_time = state["observed_at"]
        block_ids = set()
        for block in state["blocks"]:
            if set(block) != CACHE_BLOCK_KEYS or not isinstance(block["block_id"], int) or isinstance(block["block_id"], bool) or block["block_id"] < 0 or block["block_id"] in block_ids or not isinstance(block["hash"], int) or isinstance(block["hash"], bool) or type(block["used"]) is not bool or not isinstance(block["ref_count"], int) or isinstance(block["ref_count"], bool) or block["ref_count"] < 0 or not isinstance(block["token_ids"], list) or not block["token_ids"] or any(not isinstance(token, int) or isinstance(token, bool) for token in block["token_ids"]):
                raise ValueError("cache state block schema differs")
            if (block["used"] and block["ref_count"] <= 0) or (not block["used"] and block["ref_count"] != 0):
                raise ValueError("cache state block lifecycle differs")
            block_ids.add(block["block_id"])
        state_by_index[index] = state
    for index, (event, cache) in enumerate(
        zip(admission_events, cache_events, strict=True)
    ):
        expected_keys = (
            ADMISSION_V2_KEYS if policy == "prefix_aware_fifo" else ADMISSION_V1_KEYS
        )
        if set(event) != expected_keys or event["schema_version"] != (
            2 if policy == "prefix_aware_fifo" else 1
        ):
            raise ValueError("admission event schema differs")
        request_id = event["request_id"]
        if request_id not in specs or event["event_index"] != index:
            raise ValueError("admission event identity differs")
        if event["action"] not in {"admitted", "released", "rejected"}:
            raise ValueError("unknown admission action")
        if not finite_number(event["arrival_at"]) or not finite_number(event["observed_at"]):
            raise ValueError("admission event timestamps differ")
        if (
            event["observed_at"] < previous_time
            or event["observed_at"] < event["arrival_at"]
            or abs(event["arrival_at"] - specs[request_id]["arrival_time"]) > 1e-9
        ):
            raise ValueError("admission event time ordering differs")
        if not positive_int(event["required_blocks"]) or not isinstance(
            event["reserved_blocks_after"], int
        ) or isinstance(event["reserved_blocks_after"], bool):
            raise ValueError("admission event numeric schema differs")
        full_blocks = required_blocks(specs[request_id], block_size)
        if event["required_blocks"] != full_blocks:
            raise ValueError("admission event footprint differs")
        while (
            cursor < len(arrivals)
            and arrivals[cursor]["arrival_time"] <= event["observed_at"]
        ):
            queue.append(arrivals[cursor]["request_id"])
            cursor += 1
        action = event["action"]
        if action == "released":
            if event["queue_wait_ms"] is not None or request_id not in active:
                raise ValueError("invalid release event")
            reservation = admission_preview[request_id]["incremental_reservation_blocks"]
            if policy != "prefix_aware_fifo":
                reservation = full_blocks
            active.remove(request_id)
            reserved -= reservation
            # A release repeats the immutable preview captured at admission;
            # its validity is checked against that prior event below rather
            # than against the post-release cache state.
        else:
            if not finite_number(event["queue_wait_ms"]):
                raise ValueError("admission event queue wait differs")
            expected_wait = (event["observed_at"] - event["arrival_at"]) * 1000
            if abs(event["queue_wait_ms"] - expected_wait) > 1e-6:
                raise ValueError("admission event queue wait differs")
            waits.append(event["queue_wait_ms"])
            if not queue or queue[0] != request_id:
                raise ValueError("admission FIFO ordering differs")
            if action == "admitted":
                if event["reason"] is not None or request_id in active or request_id in rejected:
                    raise ValueError("invalid admitted event")
            elif action == "rejected":
                if event["reason"] != "kv_reservation_timeout" or request_id in active:
                    raise ValueError("invalid rejected event")
            else:
                raise ValueError("unknown admission action")
        cache_reason = event["reason"]
        cache_reservation = (
            event["reservation_blocks"]
            if policy == "prefix_aware_fifo"
            else full_blocks
        )
        if policy == "prefix_aware_fifo":
            if not positive_int(cache_reservation):
                raise ValueError("prefix admission reservation differs")
        if action == "admitted":
            if event["observed_at"] - event["arrival_at"] >= admission["max_queue_wait_seconds"]:
                raise ValueError("admission occurred after timeout")
            if reserved + cache_reservation > admission["total_kv_blocks"]:
                raise ValueError("admission exceeds KV reservation capacity")
            queue.pop(0)
            active.add(request_id)
            reserved += cache_reservation
        elif action == "rejected":
            if event["observed_at"] - event["arrival_at"] < admission["max_queue_wait_seconds"]:
                raise ValueError("request rejected before admission timeout")
            queue.pop(0)
            rejected.add(request_id)
        if event["reserved_blocks_after"] != reserved or reserved < 0:
            raise ValueError("admission reservation replay differs")
        state_index = cache.get("cache_state_index")
        if not isinstance(state_index, int) or isinstance(state_index, bool) or state_index not in state_by_index:
            raise ValueError("cache event references an unknown state")
        validate_cache_event(
            cache,
            index=index,
            action=action,
            request_id=request_id,
            observed_at=event["observed_at"],
            reserved_blocks_after=reserved,
            reason=cache_reason,
            expected_full_blocks=full_blocks,
            block_size=block_size,
            prompt_token_ids=prepared[request_id],
            cache_state=state_by_index[state_index],
            permit_prior_state=action == "released",
        )
        if action == "admitted":
            admission_preview[request_id] = cache
            if policy == "prefix_aware_fifo" and (
                cache["incremental_reservation_blocks"] != cache_reservation
            ):
                raise ValueError("cache/admission reservation differs")
        elif action == "released":
            prior = admission_preview[request_id]
            for field in (
                "full_required_blocks",
                "matched_prefix_blocks",
                "active_shared_blocks",
                "inactive_cached_blocks",
                "matched_prefix_tokens",
                "incremental_reservation_blocks",
                "matched_block_ids",
                "active_block_ids",
                "inactive_block_ids",
                "cache_state_index",
            ):
                if cache[field] != prior[field]:
                    raise ValueError("release cache preview differs from admission")
            del admission_preview[request_id]
        previous_time = event["observed_at"]
        peak_reserved = max(peak_reserved, reserved)
        per_request[(request_id, action)] += 1
    if active or queue or cursor != len(arrivals) or reserved or admission_preview:
        raise ValueError("admission lifecycle did not return to zero")
    for request_id, item in raw.items():
        actions = [
            event["action"]
            for event in admission_events
            if event["request_id"] == request_id
        ]
        if item["terminal_state"] == "Finished":
            if actions != ["admitted", "released"]:
                raise ValueError("finished request admission lifecycle differs")
            admitted = next(
                event
                for event in admission_events
                if event["request_id"] == request_id
                and event["action"] == "admitted"
            )
            if abs(admitted["observed_at"] - item["admitted_at"]) > 1e-6:
                raise ValueError("finished request admission timestamp differs")
        elif item["terminal_state"] == "Rejected":
            if actions != ["rejected"]:
                raise ValueError("rejected request admission lifecycle differs")
            rejected_event = next(
                event
                for event in admission_events
                if event["request_id"] == request_id
                and event["action"] == "rejected"
            )
            if abs(rejected_event["observed_at"] - item["terminal_at"]) > 1e-6:
                raise ValueError("rejected request timestamp differs")
        else:
            raise ValueError("formal run has a non-terminal request")
    return {
        "admitted_requests": sum(per_request[(key, "admitted")] for key in raw),
        "rejected_requests": len(rejected),
        "peak_reserved_blocks": peak_reserved,
        "max_observed_queue_wait_ms": max(waits, default=0.0),
    }


def load_trace_from_document(document: dict[str, Any]):
    # Avoid a mutable temp trace: ``load_trace`` only performs schema checks,
    # so mirror its strict v2 contract locally for immutable raw evidence.
    if set(document) != {"schema_version", "description", "time_unit", "requests"}:
        raise ValueError("trace schema differs")
    if document["schema_version"] != 2 or document["time_unit"] != "seconds":
        raise ValueError("Stage 11 trace version differs")
    from experiments.benchmark.open_loop import RequestSpec

    specs = []
    ids = set()
    keys = {
        "request_id", "request_class", "arrival_time", "prompt_length",
        "max_output_tokens", "seed", "prefix_group", "shared_prefix_length",
    }
    previous_arrival = -float("inf")
    for item in document["requests"]:
        if not isinstance(item, dict) or set(item) != keys:
            raise ValueError("Stage 11 trace request schema differs")
        if (
            not isinstance(item["request_id"], str)
            or not item["request_id"]
            or item["request_id"] in ids
            or item["request_class"] not in {"interactive", "long"}
            or not finite_number(item["arrival_time"])
            or item["arrival_time"] < 0
            or item["arrival_time"] < previous_arrival
        ):
            raise ValueError("Stage 11 trace request identifier differs")
        previous_arrival = item["arrival_time"]
        for field in ("prompt_length", "max_output_tokens"):
            if not positive_int(item[field]):
                raise ValueError("Stage 11 trace request footprint differs")
        if not isinstance(item["seed"], int) or isinstance(item["seed"], bool):
            raise ValueError("Stage 11 trace request seed differs")
        group = item["prefix_group"]
        length = item["shared_prefix_length"]
        if group is None:
            if length != 0:
                raise ValueError("Stage 11 null prefix contract differs")
        elif (
            not isinstance(group, str)
            or not group
            or not positive_int(length)
            or length >= item["prompt_length"]
        ):
            raise ValueError("Stage 11 shared prefix contract differs")
        # ``prepare_requests`` derives a group stream from ``prefix_group``;
        # requests with 256 and 512 shared tokens from that stream legitimately
        # share the first 256 tokens.  Do not incorrectly require one length
        # per group here.
        ids.add(item["request_id"])
        specs.append(RequestSpec(**item))
    # The generator and runner validate the remaining scalar restrictions.
    # Reusing the public prompt constructor then binds raw cache evidence to
    # exactly the same deterministic token construction.
    return specs


def validate_pressure(summary: dict[str, Any], capacity: int, rejected: int) -> None:
    pressure = summary.get("pressure")
    if not isinstance(pressure, dict):
        raise ValueError("pressure evidence is missing")
    if (
        pressure.get("schema_version") != 1
        or pressure.get("observation_active") is not True
        or pressure.get("observation_complete") is not True
        or pressure.get("oom_detected") is not False
        or pressure.get("collection_error") is not None
        or pressure.get("rejected_requests") != rejected
        or pressure.get("admission_rejection_supported") is not True
    ):
        raise ValueError("pressure observation differs")
    kv = pressure.get("kv_cache")
    if not isinstance(kv, dict) or (
        kv.get("total_blocks") != capacity
        or kv.get("final_used_blocks") != 0
        or kv.get("final_free_blocks") != capacity
        or not isinstance(kv.get("peak_used_blocks"), int)
        or kv["peak_used_blocks"] < 0
        or kv["peak_used_blocks"] > capacity
    ):
        raise ValueError("KV pressure observation differs")


def normalized_pair(config: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(config)
    value["admission"]["policy"] = "<policy>"
    value["output"]["run_id"] = "<run_id>"
    return value


def mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def nullable_mean_std(values: list[float | None]) -> dict[str, float | None]:
    if all(value is None for value in values):
        return {"mean": None, "sample_std": None}
    if any(value is None or not finite_number(value) for value in values):
        raise ValueError("matched metric has inconsistent null values")
    return mean_std([float(value) for value in values])


def validate_record(path: Path, expected: dict[str, Any], frozen: dict[str, Any]) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    run_id = expected["output"]["run_id"]
    if summary.get("status") != "passed" or summary["summary"]["runtime"]["timed_out"]:
        raise ValueError(f"run did not complete: {run_id}")
    config_path = REPO_ROOT / CONFIG_DIRECTORY / f"{run_id}.json"
    if sha256(config_path) != summary["provenance"].get("config_sha256"):
        raise ValueError(f"config hash differs: {run_id}")
    if json.loads(config_path.read_text(encoding="utf-8")) != expected:
        raise ValueError(f"config contents differ: {run_id}")
    trace_path = REPO_ROOT / expected["workload"]["trace_path"]
    if sha256(trace_path) != summary["provenance"].get("trace_sha256"):
        raise ValueError(f"trace hash differs: {run_id}")
    if Path(summary["provenance"].get("trace_path", "")).name != trace_path.name:
        raise ValueError(f"trace path differs: {run_id}")
    if {key: summary["provenance"].get(key) for key in CODE_KEYS} != frozen["execution_code"]:
        raise ValueError(f"execution code differs: {run_id}")
    expected_engine = copy.deepcopy(expected["engine"])
    expected_engine.setdefault("cuda_event_timing", False)
    for field, value in (
        ("engine", expected_engine),
        ("sampling", expected["sampling"]),
        ("slo", expected["slo"]),
        ("measurement", expected["measurement"]),
    ):
        if summary.get(field) != value:
            raise ValueError(f"summary {field} differs: {run_id}")
    if summary.get("upstream_commit") != expected["upstream_commit"]:
        raise ValueError(f"upstream binding differs: {run_id}")
    if any(summary["model"].get(key) != expected["model"][key] for key in ("repo_id", "revision", "sha256")):
        raise ValueError(f"model binding differs: {run_id}")
    if summary["model"].get("verified_sha256") != expected["model"]["sha256"]:
        raise ValueError(f"model verification differs: {run_id}")
    artifacts = summary.get("artifacts", {})
    requests_path = RESULTS / f"{run_id}.requests.jsonl"
    steps_path = RESULTS / f"{run_id}.steps.jsonl"
    admission_path = RESULTS / f"{run_id}.admission.jsonl"
    cache_path = RESULTS / f"{run_id}.cache.jsonl"
    cache_states_path = RESULTS / f"{run_id}.cache_states.jsonl"
    bound = (
        (requests_path, "requests_sha256"),
        (steps_path, "steps_sha256"),
        (admission_path, "admission_events_sha256"),
        (cache_path, "cache_events_sha256"),
        (cache_states_path, "cache_states_sha256"),
    )
    if any(not path.exists() or sha256(path) != artifacts.get(key) for path, key in bound):
        raise ValueError(f"raw artifact hash differs: {run_id}")
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    trace_key = next(
        (
            (reuse_ratio, seed)
            for reuse_ratio in REUSE_RATIOS
            for seed in SEEDS
            if trace_path.name
            == f"stage11_formal_r1_reuse{reuse_ratio}_seed{seed}.json"
        ),
        None,
    )
    if trace_key is None or trace != build_trace(*trace_key):
        raise ValueError(f"formal trace differs from generator: {run_id}")
    requests = read_jsonl(requests_path)
    steps = read_jsonl(steps_path)
    validate_raw(trace, requests, steps, summary)
    terminal = summary["summary"]["terminal_counts"]
    counts = Counter(item["terminal_state"] for item in requests)
    states = ("Finished", "Rejected", "Failed", "Cancelled", "Unfinished")
    if (
        len(requests) != terminal["submitted"]
        or set(counts) - set(states)
        or not terminal["reconciled"]
        or any(counts[name] != terminal[name] for name in states)
        or counts["Failed"]
        or counts["Cancelled"]
        or counts["Unfinished"]
    ):
        raise ValueError(f"terminal reconciliation differs: {run_id}")
    policy = expected["admission"]["policy"]
    admission = summary.get("admission")
    expected_keys = (
        SUMMARY_ADMISSION_V2_KEYS
        if policy == "prefix_aware_fifo"
        else SUMMARY_ADMISSION_V1_KEYS
    )
    if not isinstance(admission, dict) or set(admission) != expected_keys:
        raise ValueError(f"admission summary schema differs: {run_id}")
    if (
        admission["schema_version"] != (2 if policy == "prefix_aware_fifo" else 1)
        or admission["policy"] != policy
        or admission["total_kv_blocks"] != expected["engine"]["num_kvcache_blocks"]
        or admission["max_queue_wait_seconds"] != expected["admission"]["max_queue_wait_seconds"]
        or admission["final_reserved_blocks"] != 0
    ):
        raise ValueError(f"admission summary binding differs: {run_id}")
    if policy == "prefix_aware_fifo" and admission["observe_prefix_cache"] is not True:
        raise ValueError(f"prefix observation differs: {run_id}")
    replay = validate_events(
        trace=trace,
        requests=requests,
        admission_events=read_jsonl(admission_path),
        cache_events=read_jsonl(cache_path),
        cache_states=read_jsonl(cache_states_path),
        admission=admission,
        policy=policy,
        block_size=expected["engine"]["kvcache_block_size"],
    )
    if any(admission[key] != value for key, value in replay.items()):
        raise ValueError(f"admission replay differs: {run_id}")
    validate_pressure(summary, expected["engine"]["num_kvcache_blocks"], counts["Rejected"])
    return summary


def aggregate() -> dict[str, Any]:
    frozen = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if frozen != manifest():
        raise ValueError("Stage 11 manifest is not frozen")
    records = []
    for config_path, expected in planned_configs():
        summary = validate_record(
            RESULTS / f"{expected['output']['run_id']}.summary.json",
            expected,
            frozen,
        )
        records.append(
            {
                "config": str(config_path.relative_to(REPO_ROOT)),
                "run_id": expected["output"]["run_id"],
                "policy": expected["admission"]["policy"],
                "reuse_ratio": next(
                    value for value in REUSE_RATIOS if f"reuse{value}_" in expected["output"]["run_id"]
                ),
                "capacity": expected["engine"]["num_kvcache_blocks"],
                "seed": expected["sampling"]["seed"] - 20260811,
                "terminal_counts": summary["summary"]["terminal_counts"],
                "admission": summary["admission"],
                "metrics": {
                    "interactive_slo_goodput_rps": summary["summary"]["interactive"]["slo_goodput_rps"],
                    "interactive_ttft_p99_ms": summary["summary"]["interactive"]["ttft_ms"]["p99"],
                    "interactive_itl_p99_ms": summary["summary"]["interactive"]["itl_ms"]["p99"],
                    "long_window_token_goodput_tps": summary["summary"]["long"]["token_goodput_tps"],
                    "elapsed_seconds": summary["summary"]["runtime"]["elapsed_seconds"],
                },
            }
        )
    pairs: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for _path, config in planned_configs():
        key = (
            next(value for value in REUSE_RATIOS if f"reuse{value}_" in config["output"]["run_id"]),
            config["engine"]["num_kvcache_blocks"],
            config["sampling"]["seed"] - 20260811,
        )
        pairs.setdefault(key, []).append(config)
    if any(
        len(pair) != 2 or normalized_pair(pair[0]) != normalized_pair(pair[1])
        for pair in pairs.values()
    ):
        raise ValueError("matched policy-pair config differs")
    by_cell = {}
    for reuse_ratio in REUSE_RATIOS:
        for capacity in CAPACITIES:
            for policy in POLICIES.values():
                selected = [
                    record
                    for record in records
                    if record["reuse_ratio"] == reuse_ratio
                    and record["capacity"] == capacity
                    and record["policy"] == policy["policy"]
                ]
                if len(selected) != len(SEEDS):
                    raise ValueError("formal cell count differs")
                by_cell[
                    f"reuse{reuse_ratio}_kv{capacity}_{policy['policy']}"
                ] = {
                    key: nullable_mean_std(
                        [record["metrics"][key] for record in selected]
                    )
                    for key in selected[0]["metrics"]
                } | {
                    "mean_rejected_requests": statistics.mean(
                        record["terminal_counts"]["Rejected"] for record in selected
                    )
                }
    return {
        "schema_version": 1,
        "revision": "stage11_formal_r1",
        "required_cells": frozen["required_cells"],
        "validated_cells": len(records),
        "execution_code": frozen["execution_code"],
        "runs": records,
        "by_cell": by_cell,
        "decision": (
            "Prefix-aware FIFO is evaluated only as a matched KV reservation "
            "trade-off; all latency, rejection, fairness, and cache evidence "
            "must be interpreted together."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(RESULTS.resolve()):
        raise ValueError("aggregate output must be inside experiments/results")
    value = aggregate()
    value["aggregate_sha256"] = sha256(Path(__file__))
    write_json_idempotent(args.output, value)


if __name__ == "__main__":
    main()
