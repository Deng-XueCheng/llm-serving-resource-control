from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from dataclasses import asdict, replace
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
from experiments.multi_replica import (
    MultiReplicaCoordinator,
    ProcessReplicaEndpoint,
    ResourceAwareRouter,
    RoundRobinRouter,
    replica_worker,
)
from experiments.snapshot_preflight import (
    verify_model_identity,
    verify_source_snapshot,
)


def start_process_on_gpu(process, gpu_id: int) -> None:
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        process.start()
    finally:
        if previous is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous


def launch_replicas(config: dict) -> list[ProcessReplicaEndpoint]:
    ctx = mp.get_context("spawn")
    endpoints = []
    base_port = config["replicas"]["distributed_init_port_base"]
    for replica_id, gpu_id in enumerate(config["replicas"]["gpu_ids"]):
        parent, child = ctx.Pipe()
        engine = dict(config["engine"])
        engine["distributed_init_port"] = base_port + replica_id
        process = ctx.Process(
            target=replica_worker,
            kwargs={
                "connection": child,
                "replica_id": replica_id,
                "gpu_id": gpu_id,
                "model": config["model_path"],
                "engine_kwargs": engine,
                "temperature": config["sampling"]["temperature"],
                "ignore_eos": config["sampling"]["ignore_eos"],
                "sampling_seed": config["sampling"].get("seed", 0),
            },
        )
        # A spawned interpreter imports the target module before entering
        # replica_worker, so device visibility must be set before start().
        start_process_on_gpu(process, gpu_id)
        child.close()
        endpoints.append(ProcessReplicaEndpoint(parent, process, replica_id))
    # Force initialization errors to surface before arrivals begin.
    try:
        for endpoint in endpoints:
            endpoint.snapshot()
    except Exception:
        for endpoint in endpoints:
            if endpoint.process.is_alive():
                endpoint.process.terminate()
            endpoint.process.join(timeout=10)
            endpoint.connection.close()
        raise
    return endpoints


def run(config: dict) -> dict:
    source_snapshot = verify_source_snapshot()
    model_provenance = verify_model_identity(config)
    specs = load_trace(Path(config["trace_path"]))
    seed_offset = config.get("seed_offset", 0)
    arrival_scale = config.get("arrival_scale", 1.0)
    specs = [
        replace(
            spec,
            seed=spec.seed + seed_offset,
            arrival_time=spec.arrival_time * arrival_scale,
        )
        for spec in specs
    ]
    requests = prepare_requests(
        specs, token_id_upper_bound=config["token_id_upper_bound"]
    )
    router_name = config["router"]
    router = (
        RoundRobinRouter(len(config["replicas"]["gpu_ids"]))
        if router_name == "round_robin"
        else ResourceAwareRouter(prefix_aware=config.get("prefix_aware", False))
    )
    replicas = launch_replicas(config)
    runtime = [endpoint.runtime_info() for endpoint in replicas]
    residency = nvidia_compute_apps()
    base_port = config["replicas"]["distributed_init_port_base"]
    preflight = validate_gpu_preflight(
        runtime,
        residency,
        expected_worker_count=len(replicas),
        requested_gpu_ids=list(config["replicas"]["gpu_ids"]),
        distributed_ports=[base_port + index for index in range(len(replicas))],
    )
    require_valid_gpu_provenance(preflight)
    coordinator = MultiReplicaCoordinator(
        replicas, router, block_size=config["engine"]["kvcache_block_size"]
    )
    records = {
        item.spec.request_id: RequestRecord(
            item.spec.request_id, item.spec.request_class, item.spec.arrival_time
        )
        for item in requests
    }
    pending = list(requests)
    started = time.perf_counter()
    step_events = []
    error = None
    try:
        while pending or not coordinator.is_finished():
            elapsed = time.perf_counter() - started
            if elapsed >= config["max_run_seconds"]:
                break
            while pending and pending[0].spec.arrival_time <= elapsed:
                item = pending.pop(0)
                coordinator.route_request(
                    item.spec.request_id,
                    item.prompt_token_ids,
                    item.spec.max_output_tokens,
                    observed_at=elapsed,
                )
                records[item.spec.request_id].mark_admitted(elapsed)
            if coordinator.is_finished():
                if pending:
                    time.sleep(max(0, pending[0].spec.arrival_time - elapsed))
                continue
            events = coordinator.step(observed_at=elapsed)
            observed = time.perf_counter() - started
            for event in events:
                record = records[event["request_id"]]
                record.record_schedule(observed)
                if event["emitted_token_id"] is not None:
                    record.record_token(observed)
                if event["finished"]:
                    record.mark_terminal(TerminalState.FINISHED, observed)
            step_events.append({"observed_at": observed, "events": events})
    except Exception as exception:
        error = {"type": type(exception).__name__, "message": str(exception)}
    finally:
        ended = min(time.perf_counter() - started, config["max_run_seconds"])
        for record in records.values():
            if record.terminal_state is None:
                record.mark_terminal(
                    TerminalState.FAILED if error else TerminalState.UNFINISHED,
                    ended,
                    reason="runner_error" if error else "timeout",
                )
        final_snapshots = []
        for endpoint in replicas:
            try:
                final_snapshots.append(asdict(endpoint.snapshot()))
            except Exception:
                pass
        for endpoint in replicas:
            endpoint.close()
    summary = summarize_requests(
        list(records.values()),
        measurement_start=config["measurement_start"],
        measurement_end=config["measurement_end"],
        ttft_slo_ms=config["ttft_slo_ms"],
        itl_slo_ms=config["itl_slo_ms"],
        require_itl=config.get("require_itl", True),
    )
    routing = routing_summary(coordinator.route_events, coordinator.resource_events)
    provenance = validate_gpu_provenance(
        runtime,
        residency,
        coordinator.step_execution_intervals,
        require_overlap=True,
        expected_pids=[int(item["worker_pid"]) for item in runtime],
        requests_per_replica=routing["routed_requests_by_replica"],
        require_all_replicas_routed=True,
        final_snapshots=final_snapshots,
    )
    result = {
        "schema_version": 1,
        "evidence_generation": "POST_GPU_BINDING_FIX",
        "gpu_binding_fix_sha": binding_fix_sha(),
        "code_sha": git_head(),
        "benchmark_config_hash": sha256_config(config),
        "trace_hash": sha256_file(Path(config["trace_path"])),
        "source_snapshot": source_snapshot,
        "model": model_provenance,
        "seed": config.get("seed_offset", config.get("sampling", {}).get("seed")),
        "status": "ok" if error is None else "failed",
        "router": router_name,
        "error": error,
        "summary": summary,
        "requests": [record.to_dict() for record in records.values()],
        "route_events": coordinator.route_events,
        "resource_events": coordinator.resource_events,
        "step_events": step_events,
        "final_replica_snapshots": final_snapshots,
        "routing_summary": routing,
        "gpu_preflight": preflight,
        "gpu_provenance": provenance,
    }
    require_valid_gpu_provenance(provenance)
    return result


def routing_summary(route_events, resource_events):
    def distribution(values):
        ordered = sorted(values)
        if not ordered:
            return {"mean": 0.0, "p99": 0.0, "max": 0.0}
        index = min(len(ordered) - 1, round((len(ordered) - 1) * 0.99))
        return {
            "mean": sum(ordered) / len(ordered),
            "p99": ordered[index],
            "max": ordered[-1],
        }

    routed = {}
    prefix_reuse = 0
    for event in route_events:
        replica_id = str(event["replica_id"])
        routed[replica_id] = routed.get(replica_id, 0) + 1
        selected = next(
            item for item in event["snapshots"]
            if item["replica_id"] == event["replica_id"]
        )
        prefix_reuse += selected["matched_prefix_blocks"]
    return {
        "routed_requests_by_replica": routed,
        "matched_prefix_blocks_at_route": prefix_reuse,
        "queue_imbalance": distribution(
            [event["queue_imbalance"] for event in resource_events]
        ),
        "kv_imbalance": distribution(
            [event["kv_imbalance"] for event in resource_events]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as file:
        config = json.load(file)
    if args.model_path is not None:
        config["model_path"] = str(args.model_path.resolve())
    result = run(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
