from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from math import ceil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from experiments.benchmark.lifecycle import RequestRecord, TerminalState, summarize_requests
from experiments.benchmark.open_loop import load_trace, prepare_requests
from experiments.gpu_provenance import (
    binding_fix_sha,
    git_head,
    nvidia_compute_apps,
    require_valid_gpu_provenance,
    sha256_config,
    sha256_file,
    validate_gpu_preflight,
    validate_gpu_provenance,
)
from experiments.multi_replica import ProcessReplicaEndpoint, replica_worker
from experiments.run_multi_replica import start_process_on_gpu
from experiments.snapshot_preflight import (
    verify_model_identity,
    verify_source_snapshot,
)


def launch_worker(config: dict, *, role: str, gpu_id: int, port: int):
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    engine = dict(config["engine"])
    engine["distributed_init_port"] = port
    process = ctx.Process(
        target=replica_worker,
        kwargs={
            "connection": child,
            "replica_id": 0 if role == "prefill" else 1,
            "gpu_id": gpu_id,
            "model": config["model_path"],
            "engine_kwargs": engine,
            "temperature": config["sampling"]["temperature"],
            "ignore_eos": config["sampling"]["ignore_eos"],
            "sampling_seed": config["sampling"].get("seed", 0),
        },
    )
    start_process_on_gpu(process, gpu_id)
    child.close()
    endpoint = ProcessReplicaEndpoint(parent, process, 0 if role == "prefill" else 1)
    endpoint.snapshot()
    return endpoint


def run(config: dict) -> dict:
    source_snapshot = verify_source_snapshot()
    model_provenance = verify_model_identity(config)
    requests = prepare_requests(
        load_trace(Path(config["trace_path"])),
        token_id_upper_bound=config["token_id_upper_bound"],
    )
    prefill = launch_worker(
        config, role="prefill", gpu_id=0,
        port=config["distributed_init_port_base"],
    )
    decode = launch_worker(
        config, role="decode", gpu_id=1,
        port=config["distributed_init_port_base"] + 1,
    )
    runtime = [prefill.runtime_info(), decode.runtime_info()]
    residency = nvidia_compute_apps()
    base_port = config["distributed_init_port_base"]
    preflight = validate_gpu_preflight(
        runtime, residency, expected_worker_count=2,
        requested_gpu_ids=[0, 1], distributed_ports=[base_port, base_port + 1],
    )
    require_valid_gpu_provenance(preflight)
    records = {
        item.spec.request_id: RequestRecord(
            item.spec.request_id, item.spec.request_class, item.spec.arrival_time
        ) for item in requests
    }
    pending = list(requests)
    decode_ids = {}
    active_prefill = None
    transfer_queue = []
    transfer_events = []
    step_events = []
    execution_intervals = []
    started = time.perf_counter()
    executor = ThreadPoolExecutor(max_workers=1)
    error = None
    try:
        while pending or active_prefill is not None or transfer_queue or not decode.is_finished():
            now = time.perf_counter() - started
            if now >= config["max_run_seconds"]:
                break
            if active_prefill is None and pending and pending[0].spec.arrival_time <= now:
                item = pending.pop(0)
                records[item.spec.request_id].mark_admitted(now)
                active_prefill = (
                    item,
                    now,
                    executor.submit(
                        prefill.audit_prefill_export,
                        item.prompt_token_ids,
                        item.spec.max_output_tokens,
                    ),
                )
            if active_prefill is not None and active_prefill[2].done():
                item, prefill_started, future = active_prefill
                audited = future.result()
                execution_intervals.append({
                    key: audited[key] for key in (
                        "replica_id", "worker_pid", "step_started", "step_finished"
                        , "cuda_elapsed_ms"
                    )
                })
                transfer = audited["transfer"]
                host_ready_at = time.perf_counter() - started
                record = records[item.spec.request_id]
                record.record_schedule(prefill_started)
                record.record_token(
                    prefill_started + transfer["prefill_compute_ms"] / 1000
                )
                payload_bytes = len(transfer["kv"]["data"])
                event = {
                    "schema_version": 1,
                    "request_id": item.spec.request_id,
                    "arrival_at": item.spec.arrival_time,
                    "prefill_queue_time_ms": (prefill_started - item.spec.arrival_time) * 1000,
                    "prefill_compute_time_ms": transfer["prefill_compute_ms"],
                    "kv_transfer_bytes": payload_bytes,
                    "kv_transfer_started_at": (
                        prefill_started + transfer["prefill_compute_ms"] / 1000
                    ),
                    "host_payload_ready_at": host_ready_at,
                    "device_to_host_ms": transfer["kv"]["device_to_host_ms"],
                    "first_token_id": transfer["metadata"]["first_token_id"],
                    "source_block_ids": transfer["metadata"]["source_block_ids"],
                    "transport": "pinned_host_staging",
                }
                transfer_queue.append((item, transfer, event))
                active_prefill = None
            if transfer_queue:
                item, transfer, event = transfer_queue[0]
                required = ceil(
                    transfer["metadata"]["materialized_tokens"]
                    / config["engine"]["kvcache_block_size"]
                )
                if decode.snapshot().free_kv_blocks >= required:
                    imported = decode.import_decode(transfer)
                    transfer_finished = time.perf_counter() - started
                    seq_id = imported["seq_id"]
                    decode_ids[seq_id] = item.spec.request_id
                    event.update({
                        "kv_transfer_finished_at": transfer_finished,
                        "kv_transfer_latency_ms": (
                            transfer_finished - event["kv_transfer_started_at"]
                        ) * 1000,
                        "host_to_device_ms": imported["host_to_device_ms"],
                        "decode_queue_started_at": transfer_finished,
                    })
                    transfer_events.append(event)
                    transfer_queue.pop(0)
            if not decode.is_finished():
                audited = decode.audit_step()
                execution_intervals.append({
                    key: audited[key] for key in (
                        "replica_id", "worker_pid", "step_started", "step_finished"
                        , "cuda_elapsed_ms"
                    )
                })
                events = audited["events"]
                observed = time.perf_counter() - started
                for event in events:
                    request_id = decode_ids[event["seq_id"]]
                    record = records[request_id]
                    record.record_schedule(observed)
                    if event["emitted_token_id"] is not None:
                        record.record_token(observed)
                    if event["finished"]:
                        record.mark_terminal(TerminalState.FINISHED, observed)
                step_events.append({"observed_at": observed, "events": events})
            elif active_prefill is None and pending:
                delay = pending[0].spec.arrival_time - (time.perf_counter() - started)
                if delay > 0:
                    time.sleep(min(delay, 0.001))
    except Exception as exception:
        error = {"type": type(exception).__name__, "message": str(exception)}
    finally:
        ended = min(time.perf_counter() - started, config["max_run_seconds"])
        executor.shutdown(wait=True)
        final = []
        for endpoint in (prefill, decode):
            snapshot = endpoint.snapshot()
            final.append({name: getattr(snapshot, name) for name in snapshot.__slots__})
            endpoint.close()
        for record in records.values():
            if record.terminal_state is None:
                record.mark_terminal(
                    TerminalState.FAILED if error else TerminalState.UNFINISHED,
                    ended,
                    reason="runner_error" if error else "timeout",
                )
    summary = summarize_requests(
        list(records.values()), measurement_start=config["measurement_start"],
        measurement_end=config["measurement_end"], ttft_slo_ms=config["ttft_slo_ms"],
        itl_slo_ms=config["itl_slo_ms"], require_itl=config.get("require_itl", True),
    )
    for event in transfer_events:
        record = records[event["request_id"]]
        event["decode_start_time"] = record.token_timestamps[1] if len(record.token_timestamps) > 1 else None
        event["decode_queue_time_ms"] = (
            None if event["decode_start_time"] is None else
            (event["decode_start_time"] - event["decode_queue_started_at"]) * 1000
        )
        event["end_to_end_latency_ms"] = None if record.e2e_s is None else record.e2e_s * 1000
        event["transfer_overhead_ratio"] = (
            None if record.e2e_s is None or record.e2e_s == 0 else
            event["kv_transfer_latency_ms"] / (record.e2e_s * 1000)
        )
    provenance = validate_gpu_provenance(
        runtime, residency, execution_intervals, require_overlap=False,
        expected_pids=[int(item["worker_pid"]) for item in runtime],
        final_snapshots=final,
    )
    result = {
        "schema_version": 1, "status": "ok" if error is None else "failed",
        "evidence_generation": "POST_GPU_BINDING_FIX",
        "gpu_binding_fix_sha": binding_fix_sha(),
        "code_sha": git_head(),
        "benchmark_config_hash": sha256_config(config),
        "trace_hash": sha256_file(Path(config["trace_path"])),
        "source_snapshot": source_snapshot,
        "model": model_provenance,
        "seed": config.get("sampling", {}).get("seed"),
        "transport": "pinned_host_staging", "error": error, "summary": summary,
        "requests": [record.to_dict() for record in records.values()],
        "transfer_events": transfer_events, "step_events": step_events,
        "final_worker_snapshots": final, "gpu_preflight": preflight,
        "gpu_provenance": provenance,
    }
    require_valid_gpu_provenance(provenance)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.model_path is not None:
        config["model_path"] = str(args.model_path.resolve())
    result = run(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
