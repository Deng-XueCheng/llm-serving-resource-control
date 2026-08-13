from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
from math import isfinite
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.generate_stage9_configs import POLICIES, SEEDS, manifest, planned_configs
from experiments.benchmark.lifecycle import RequestRecord, TerminalState, summarize_requests


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "experiments/results"
MANIFEST = REPO_ROOT / "experiments/configs/stage9_formal_r1_manifest.json"
OUTPUT = RESULTS / "stage9_formal_r1_strict_r6_aggregate_20260810.json"
CODE_KEYS = (
    "runner_sha256", "open_loop_sha256", "lifecycle_sha256", "llm_engine_sha256",
    "scheduler_sha256", "block_manager_sha256", "model_runner_sha256", "config_module_sha256",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(float(value))


def mean_std(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0}


def normalized(config: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(config)
    value["admission"] = "<policy>"
    value["output"]["run_id"] = "<run_id>"
    return value


def validate_raw(trace: dict[str, Any], requests: list[dict[str, Any]], steps: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    planned = {item["request_id"]: item for item in trace["requests"]}
    raw = {item["request_id"]: item for item in requests}
    if len(planned) != len(trace["requests"]) or len(raw) != len(requests) or set(planned) != set(raw):
        raise ValueError("trace/raw request identifiers differ")
    rebuilt = []; emitted = Counter(); finished = Counter()
    step_keys = {"step_index", "started_at", "finished_at", "duration_ms", "phase", "num_scheduled_tokens", "events"}
    event_keys = {"request_id", "seq_id", "phase", "num_scheduled_tokens", "emitted_token", "finished"}
    for step_index, item in enumerate(steps):
        if set(item) != step_keys or not isinstance(item["step_index"], int) or isinstance(item["step_index"], bool) or item["step_index"] != step_index or not isinstance(item["num_scheduled_tokens"], int) or isinstance(item["num_scheduled_tokens"], bool) or item["events"] is None or item["phase"] not in {"prefill", "decode"} or any(not finite_number(item[key]) for key in ("started_at", "finished_at", "duration_ms")):
            raise ValueError("invalid step schema")
        for event in item["events"]:
            if set(event) != event_keys or type(event["emitted_token"]) is not bool or type(event["finished"]) is not bool or not isinstance(event["seq_id"], int) or isinstance(event["seq_id"], bool) or not isinstance(event["num_scheduled_tokens"], int) or isinstance(event["num_scheduled_tokens"], bool): raise ValueError("invalid step event schema")
            request_id = event["request_id"]
            if request_id not in raw or event["phase"] != item["phase"]: raise ValueError("step event binding differs")
            emitted[request_id] += event["emitted_token"]; finished[request_id] += event["finished"]
    for request_id, item in raw.items():
        spec = planned[request_id]; state = item["terminal_state"]
        if not finite_number(item["arrival_at"]) or item["request_class"] != spec["request_class"] or abs(item["arrival_at"] - spec["arrival_time"]) > 1e-9 or state not in {state.value for state in TerminalState}:
            raise ValueError("raw request trace binding differs")
        record = RequestRecord(request_id=request_id, request_class=item["request_class"], arrival_at=item["arrival_at"])
        if state == "Finished":
            timestamps = item["token_timestamps"]
            if item["terminal_reason"] is not None or item["admitted_at"] is None or len(timestamps) != spec["max_output_tokens"] or timestamps != sorted(timestamps) or item["terminal_at"] < timestamps[-1] or emitted[request_id] != len(timestamps) or finished[request_id] != 1:
                raise ValueError("finished raw lifecycle differs")
            if any(not finite_number(value) or value < item["arrival_at"] for value in [item["admitted_at"], *timestamps, item["terminal_at"]]): raise ValueError("raw timestamp is invalid")
            record.mark_admitted(item["admitted_at"])
            for timestamp in timestamps: record.record_token(timestamp)
            record.mark_terminal(TerminalState.FINISHED, item["terminal_at"])
        elif state == "Rejected":
            if item["admitted_at"] is not None or item["token_timestamps"] or emitted[request_id] or finished[request_id] or item["terminal_reason"] != "kv_reservation_timeout":
                raise ValueError("rejected raw lifecycle differs")
            if not finite_number(item["terminal_at"]) or item["terminal_at"] < item["arrival_at"]: raise ValueError("rejected terminal timestamp is invalid")
            record.mark_terminal(TerminalState.REJECTED, item["terminal_at"], reason=item["terminal_reason"])
        else:
            raise ValueError("formal run contains invalid terminal state")
        rebuilt.append(record)
    recomputed = summarize_requests(rebuilt, measurement_start=summary["measurement"]["start_seconds"], measurement_end=summary["measurement"]["end_seconds"], ttft_slo_ms=summary["slo"]["ttft_slo_ms"], itl_slo_ms=summary["slo"]["itl_slo_ms"], require_itl=summary["slo"]["require_itl"])
    if recomputed != summary["summary"] | {"runtime": summary["summary"]["runtime"]}:
        for key, value in recomputed.items():
            if summary["summary"].get(key) != value: raise ValueError(f"raw metric recomputation differs: {key}")


def validate_events(events: list[dict[str, Any]], requests: list[dict[str, Any]], trace: dict[str, Any], admission: dict[str, Any]) -> dict[str, float | int]:
    reserved = 0
    active: set[str] = set()
    rejected: set[str] = set()
    raw = {item["request_id"]: item for item in requests}
    planned = {item["request_id"]: item for item in trace["requests"]}
    queue: list[str] = []
    arrivals = sorted(trace["requests"], key=lambda item: (item["arrival_time"], item["request_id"]))
    cursor = 0
    previous_time = -float("inf")
    per_request = Counter()
    peak_reserved = 0; waits: list[float] = []
    for index, event in enumerate(events):
        if event.get("schema_version") != 1 or event.get("event_index") != index:
            raise ValueError("admission event sequence is invalid")
        request_id = event.get("request_id")
        if request_id not in raw or request_id not in planned:
            raise ValueError("admission event request binding is invalid")
        if any(not finite_number(event.get(key)) for key in ("arrival_at", "observed_at", "required_blocks", "reserved_blocks_after")) or (event.get("action") != "released" and not finite_number(event.get("queue_wait_ms"))):
            raise ValueError("admission event numeric schema differs")
        while cursor < len(arrivals) and arrivals[cursor]["arrival_time"] <= event["observed_at"]:
            queue.append(arrivals[cursor]["request_id"]); cursor += 1
        expected_blocks = (planned[request_id]["prompt_length"] + planned[request_id]["max_output_tokens"] - 1 + 255) // 256
        action = event.get("action")
        if event["observed_at"] < previous_time or event["observed_at"] < event["arrival_at"]:
            raise ValueError("admission event time differs")
        if action != "released" and abs(event["queue_wait_ms"] - (event["observed_at"] - event["arrival_at"]) * 1000) > 1e-6:
            raise ValueError("admission queue wait differs")
        if action != "released": waits.append(event["queue_wait_ms"])
        previous_time = event["observed_at"]; per_request[(request_id, action)] += 1
        if event.get("required_blocks") != expected_blocks or abs(event["arrival_at"] - planned[request_id]["arrival_time"]) > 1e-9:
            raise ValueError("admission event trace footprint differs")
        blocks = expected_blocks
        if action == "admitted":
            if not queue or queue[0] != request_id or request_id in active or request_id in rejected or event.get("reason") is not None:
                raise ValueError("invalid admitted event")
            if event["observed_at"] - event["arrival_at"] >= admission["max_queue_wait_seconds"] or reserved + blocks > admission["total_kv_blocks"]:
                raise ValueError("FIFO admission violated timeout or capacity")
            queue.pop(0)
            active.add(request_id); reserved += blocks
        elif action == "released":
            if request_id not in active:
                raise ValueError("release without admission")
            active.remove(request_id); reserved -= blocks
        elif action == "rejected":
            if not queue or queue[0] != request_id or request_id in active or event.get("reason") != "kv_reservation_timeout":
                raise ValueError("invalid rejected event")
            if event["observed_at"] - event["arrival_at"] < admission["max_queue_wait_seconds"]:
                raise ValueError("FIFO rejected before timeout")
            queue.pop(0)
            rejected.add(request_id)
        else:
            raise ValueError("unknown admission action")
        if reserved != event.get("reserved_blocks_after") or reserved < 0:
            raise ValueError("admission reservation replay differs")
        peak_reserved = max(peak_reserved, reserved)
    if active or queue or cursor != len(arrivals) or reserved or reserved != admission["final_reserved_blocks"]:
        raise ValueError("admission reservation did not return to zero")
    for request_id, item in raw.items():
        if item["terminal_state"] == "Rejected":
            matching = [event for event in events if event["request_id"] == request_id]
            if request_id not in rejected or per_request[(request_id, "rejected")] != 1 or [event["action"] for event in matching] != ["rejected"] or matching[0]["reason"] != item["terminal_reason"] or abs(matching[0]["observed_at"] - item["terminal_at"]) > 1e-6 or item["admitted_at"] is not None or item["token_timestamps"]:
                raise ValueError("rejected request lifecycle differs")
        elif item["terminal_state"] == "Finished":
            matching = [event for event in events if event["request_id"] == request_id]
            if request_id in rejected or item["admitted_at"] is None or [event["action"] for event in matching] != ["admitted", "released"] or matching[1]["reason"] is not None or abs(matching[0]["observed_at"] - item["admitted_at"]) > 1e-6 or abs(matching[1]["observed_at"] - item["terminal_at"]) > 1e-6:
                raise ValueError("finished request lifecycle differs")
    return {"admitted_requests": sum(per_request[(request_id, "admitted")] for request_id in raw), "rejected_requests": len(rejected), "peak_reserved_blocks": peak_reserved, "max_observed_queue_wait_ms": max(waits, default=0.0)}


def validate_record(path: Path, expected: dict[str, Any], frozen_code: dict[str, str]) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    run_id = expected["output"]["run_id"]
    if summary["status"] != "passed" or summary["summary"]["runtime"]["timed_out"]:
        raise ValueError(f"run did not complete: {run_id}")
    config_path = REPO_ROOT / "experiments/configs/stage9_formal_r1" / f"{run_id}.json"
    if sha256(config_path) != summary["provenance"]["config_sha256"] or json.loads(config_path.read_text(encoding="utf-8")) != expected:
        raise ValueError(f"config binding differs: {run_id}")
    requests_path = RESULTS / f"{run_id}.requests.jsonl"; steps_path = RESULTS / f"{run_id}.steps.jsonl"
    if sha256(requests_path) != summary["artifacts"]["requests_sha256"] or sha256(steps_path) != summary["artifacts"]["steps_sha256"]:
        raise ValueError(f"raw hash differs: {run_id}")
    trace_path = REPO_ROOT / expected["workload"]["trace_path"]
    if sha256(trace_path) != summary["provenance"]["trace_sha256"] or Path(summary["provenance"]["trace_path"]).name != trace_path.name:
        raise ValueError(f"trace binding differs: {run_id}")
    expected_engine = copy.deepcopy(expected["engine"])
    expected_engine.setdefault("cuda_event_timing", False)
    for field, planned_value in (("engine", expected_engine), ("sampling", expected["sampling"]), ("slo", expected["slo"]), ("measurement", expected["measurement"])):
        if summary[field] != planned_value: raise ValueError(f"summary {field} differs: {run_id}")
    if {key: summary["provenance"][key] for key in CODE_KEYS} != frozen_code:
            raise ValueError(f"execution code differs: {run_id}")
    if summary["upstream_commit"] != expected["upstream_commit"] or any(summary["model"][key] != expected["model"][key] for key in ("repo_id", "revision", "sha256")) or summary["model"]["verified_sha256"] != expected["model"]["sha256"]:
        raise ValueError(f"model or upstream binding differs: {run_id}")
    requests = read_jsonl(requests_path); trace = json.loads(trace_path.read_text(encoding="utf-8")); steps = read_jsonl(steps_path)
    validate_raw(trace, requests, steps, summary)
    counts = Counter(item["terminal_state"] for item in requests)
    terminal = summary["summary"]["terminal_counts"]
    states = ("Finished", "Rejected", "Failed", "Cancelled", "Unfinished")
    if len(requests) != terminal["submitted"] or set(counts) - set(states) or not terminal["reconciled"] or any(counts[name] != terminal[name] for name in states):
        raise ValueError(f"terminal reconciliation differs: {run_id}")
    if counts["Failed"] or counts["Cancelled"] or counts["Unfinished"] or summary["pressure"]["oom_detected"] or summary["pressure"]["kv_cache"]["final_used_blocks"] != 0:
        raise ValueError(f"completion gate differs: {run_id}")
    policy = expected["admission"]["policy"]
    if summary["admission"]["policy"] != policy or summary["admission"]["max_queue_wait_seconds"] != expected["admission"]["max_queue_wait_seconds"] or summary["admission"]["final_reserved_blocks"] != 0:
        raise ValueError(f"admission summary differs: {run_id}")
    if policy == "kv_aware_fifo":
        if summary["admission"]["total_kv_blocks"] != expected["engine"]["num_kvcache_blocks"]: raise ValueError("FIFO admission capacity differs")
        events_path = RESULTS / f"{run_id}.admission.jsonl"
        if sha256(events_path) != summary["artifacts"].get("admission_events_sha256"):
            raise ValueError(f"admission hash differs: {run_id}")
        replay = validate_events(read_jsonl(events_path), requests, trace, summary["admission"])
        if any(abs(float(summary["admission"][key]) - value) > 1e-6 for key, value in replay.items()): raise ValueError("admission summary replay differs")
    elif counts["Rejected"]:
        raise ValueError("disabled admission rejected requests")
    else:
        if summary["admission"]["total_kv_blocks"] != expected["engine"]["num_kvcache_blocks"]: raise ValueError("disabled admission capacity differs")
        derived_wait = max((item["admitted_at"] - item["arrival_at"] for item in requests), default=0.0) * 1000
        if summary["admission"]["admitted_requests"] != counts["Finished"] or summary["admission"]["rejected_requests"] != 0 or summary["admission"]["peak_reserved_blocks"] != 0 or abs(summary["admission"]["max_observed_queue_wait_ms"] - derived_wait) > 1e-6: raise ValueError("disabled admission summary differs")
    if summary["pressure"]["rejected_requests"] != counts["Rejected"]: raise ValueError("pressure rejected count differs")
    elapsed = summary["summary"]["runtime"]["elapsed_seconds"]; last_terminal = max(item["terminal_at"] for item in requests)
    if not isfinite(elapsed) or elapsed < last_terminal or elapsed > expected["measurement"]["max_run_seconds"]: raise ValueError("runtime elapsed differs")
    return summary


def aggregate() -> dict[str, Any]:
    frozen = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if frozen != manifest():
        raise ValueError("Stage 9 manifest is not frozen")
    records = []
    fingerprints: set[tuple[str, ...]] = set()
    for config_path, expected in planned_configs():
        summary_path = RESULTS / f"{expected['output']['run_id']}.summary.json"
        summary = validate_record(summary_path, expected, frozen["execution_code"])
        fingerprints.add(tuple(summary["provenance"][key] for key in CODE_KEYS))
        records.append({"config": str(config_path.relative_to(REPO_ROOT)), "run_id": expected["output"]["run_id"], "policy": expected["admission"]["policy"], "seed": expected["sampling"]["seed"], "terminal_counts": summary["summary"]["terminal_counts"], "admission": summary["admission"], "metrics": {"interactive_slo_goodput_rps": summary["summary"]["interactive"]["slo_goodput_rps"], "output_throughput_tps": summary["summary"]["output_throughput_tps"], "elapsed_seconds": summary["summary"]["runtime"]["elapsed_seconds"]}})
    if len(fingerprints) != 1:
        raise ValueError("execution code fingerprint differs")
    pairs = {item["seed"]: [] for item in records}
    for config_path, expected in planned_configs(): pairs[expected["sampling"]["seed"]].append(expected)
    if any(len(pair) != 2 or normalized(pair[0]) != normalized(pair[1]) for pair in pairs.values()):
        raise ValueError("matched policy-pair config differs")
    by_policy = {}
    for name, entry in POLICIES.items():
        selected = [item for item in records if item["policy"] == entry["policy"]]
        by_policy[name] = {key: mean_std([float(item["metrics"][key]) for item in selected]) for key in selected[0]["metrics"]} | {"mean_rejected_requests": statistics.mean(item["terminal_counts"]["Rejected"] for item in selected), "mean_max_queue_wait_ms": statistics.mean(item["admission"]["max_observed_queue_wait_ms"] for item in selected)}
    return {"schema_version": 1, "revision": "stage9_formal_r1", "required_cells": 6, "validated_cells": len(records), "execution_code": dict(zip(CODE_KEYS, next(iter(fingerprints)), strict=True)), "runs": records, "by_policy": by_policy, "decision": "admission_rejection_tradeoff"}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=OUTPUT); args = parser.parse_args()
    if not args.output.resolve().is_relative_to(RESULTS.resolve()): raise ValueError("aggregate output must be inside experiments/results")
    value = aggregate(); value["aggregate_sha256"] = sha256(Path(__file__))
    encoded = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if args.output.exists() and args.output.read_text(encoding="utf-8") != encoded: raise FileExistsError("aggregate output differs")
    args.output.write_text(encoded, encoding="utf-8")
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
