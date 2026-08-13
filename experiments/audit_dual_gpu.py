from __future__ import annotations

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from experiments.multi_replica import (
    MultiReplicaCoordinator,
    ResourceAwareRouter,
    RoundRobinRouter,
)
from experiments.run_multi_replica import launch_replicas


def nvidia_compute_apps() -> list[dict[str, object]]:
    output = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
        "--format=csv,noheader,nounits",
    ], text=True)
    rows = []
    for line in output.splitlines():
        if not line.strip():
            continue
        pid, uuid, memory = (item.strip() for item in line.split(","))
        rows.append({"pid": int(pid), "gpu_uuid": uuid, "used_memory_mib": int(memory)})
    return rows


def utilization_samples(duration_s: float = 2.0) -> list[dict[str, object]]:
    samples = []
    deadline = time.perf_counter() + duration_s
    while time.perf_counter() < deadline:
        output = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ], text=True)
        samples.append({"observed_at": time.perf_counter(), "gpus": [
            {"index": int(parts[0]), "uuid": parts[1], "utilization_percent": int(parts[2]), "memory_used_mib": int(parts[3])}
            for parts in ([item.strip() for item in line.split(",")] for line in output.splitlines())
        ]})
        time.sleep(0.1)
    return samples


def overlap(left: dict, right: dict) -> float:
    return max(0.0, min(left["step_finished"], right["step_finished"]) - max(left["step_started"], right["step_started"]))


def run_smoke(config: dict, router: str) -> dict:
    run_config = deepcopy(config)
    run_config["router"] = router
    replicas = launch_replicas(run_config)
    try:
        runtime = [replica.runtime_info() for replica in replicas]
        residency = nvidia_compute_apps()
        selected_router = (
            RoundRobinRouter(len(replicas))
            if router == "round_robin"
            else ResourceAwareRouter(prefix_aware=False)
        )
        coordinator = MultiReplicaCoordinator(
            replicas,
            selected_router,
            block_size=run_config["engine"]["kvcache_block_size"],
        )
        requests = []
        for index in range(config.get("audit_request_count", 4)):
            request_id = f"{router}-{index}"
            routed = coordinator.route_request(
                request_id,
                [((index + token) % 9000) + 1 for token in range(512)],
                32,
                observed_at=time.perf_counter(),
            )
            requests.append({
                "request_id": request_id,
                "replica_id": routed.replica_id,
                "worker_pid": runtime[routed.replica_id]["worker_pid"],
                "physical_gpu_uuid": runtime[routed.replica_id]["device_uuid"],
                "local_seq_id": routed.local_seq_id,
            })
        step_rounds = []
        utilization = []
        while not all(replica.is_finished() for replica in replicas):
            active = [replica for replica in replicas if not replica.is_finished()]
            with ThreadPoolExecutor(max_workers=len(active) + 1) as executor:
                sample_future = executor.submit(utilization_samples, 0.5)
                futures = [executor.submit(replica.audit_step) for replica in active]
                step_intervals = [future.result() for future in futures]
                utilization.extend(sample_future.result())
            step_rounds.append(step_intervals)
        overlaps = [
            overlap(step_round[0], step_round[1])
            for step_round in step_rounds
            if len(step_round) == 2
        ]
        final = [asdict(replica.snapshot()) for replica in replicas]
        return {
            "router": router, "runtime": runtime, "residency": residency,
            "requests": requests, "route_events": coordinator.route_events,
            "step_rounds": step_rounds,
            "positive_overlap_count": sum(value > 0 for value in overlaps),
            "max_overlap_ms": max(overlaps, default=0.0) * 1000,
            "utilization_samples": utilization, "final_snapshots": final,
        }
    finally:
        for replica in replicas:
            replica.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = {"round_robin": run_smoke(config, "round_robin")}
    config["replicas"]["distributed_init_port_base"] += 10
    output["resource_aware"] = run_smoke(config, "resource_aware")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
