from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def nvidia_compute_apps() -> list[dict[str, object]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = []
    for line in output.splitlines():
        if not line.strip():
            continue
        pid, uuid, memory = (item.strip() for item in line.split(","))
        rows.append(
            {"pid": int(pid), "gpu_uuid": uuid, "used_memory_mib": int(memory)}
        )
    return rows


def binding_fix_sha() -> str:
    return subprocess.check_output(
        [
            "git", "log", "-1", "--format=%H", "-Sstart_process_on_gpu", "--",
            "experiments/run_multi_replica.py",
        ],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_config(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_gpu_preflight(
    runtime: list[dict[str, object]],
    residency: list[dict[str, object]],
    *,
    expected_worker_count: int,
    requested_gpu_ids: list[int],
    distributed_ports: list[int],
) -> dict[str, object]:
    errors = []
    pids = [int(item["worker_pid"]) for item in runtime]
    torch_uuids = [str(item["device_uuid"]).removeprefix("GPU-") for item in runtime]
    residency_by_pid = {int(item["pid"]): item for item in residency}
    nvml_uuids = [
        str(residency_by_pid.get(pid, {}).get("gpu_uuid", "")).removeprefix("GPU-")
        for pid in pids
    ]
    if len(pids) != expected_worker_count or len(set(pids)) != expected_worker_count:
        errors.append("unexpected_worker_pid_count")
    if len(requested_gpu_ids) != expected_worker_count or len(set(requested_gpu_ids)) != expected_worker_count:
        errors.append("requested_gpu_ids_not_unique")
    if len(distributed_ports) != expected_worker_count or len(set(distributed_ports)) != expected_worker_count:
        errors.append("distributed_ports_not_unique")
    if len(torch_uuids) != expected_worker_count or len(set(torch_uuids)) != expected_worker_count:
        errors.append("worker_physical_gpu_uuids_not_distinct")
    if torch_uuids != nvml_uuids:
        errors.append("torch_nvml_uuid_mismatch")
    if any(int(residency_by_pid.get(pid, {}).get("used_memory_mib", 0)) <= 0 for pid in pids):
        errors.append("missing_model_residency")
    return {
        "status": "PREFLIGHT_PROVENANCE_PASS" if not errors else "INVALID_GPU_PROVENANCE",
        "errors": errors,
        "requested_gpu_ids": requested_gpu_ids,
        "distributed_ports": distributed_ports,
        "physical_gpu_uuid_list": [f"GPU-{uuid}" for uuid in torch_uuids],
        "worker_runtime": runtime,
        "model_residency": [residency_by_pid.get(pid) for pid in pids],
    }


def validate_gpu_provenance(
    runtime: list[dict[str, object]],
    residency: list[dict[str, object]],
    intervals: list[dict[str, object]],
    *,
    require_overlap: bool,
    expected_pids: list[int] | None = None,
    requests_per_replica: dict[str, int] | None = None,
    require_all_replicas_routed: bool = False,
    final_snapshots: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    errors = []
    pids = [int(item["worker_pid"]) for item in runtime]
    torch_uuids = [str(item["device_uuid"]).removeprefix("GPU-") for item in runtime]
    residency_by_pid = {int(item["pid"]): item for item in residency}
    nvml_uuids = [
        str(residency_by_pid.get(pid, {}).get("gpu_uuid", "")).removeprefix("GPU-")
        for pid in pids
    ]
    if len(pids) != 2 or len(set(pids)) != 2:
        errors.append("expected_two_distinct_worker_pids")
    if expected_pids is not None and pids != expected_pids:
        errors.append("worker_pid_changed_after_preflight")
    if len(set(torch_uuids)) != len(torch_uuids):
        errors.append("worker_physical_gpu_uuids_not_distinct")
    if torch_uuids != nvml_uuids:
        errors.append("torch_nvml_uuid_mismatch")
    if any(int(residency_by_pid.get(pid, {}).get("used_memory_mib", 0)) <= 0 for pid in pids):
        errors.append("missing_model_residency")
    interval_replicas = {int(item["replica_id"]) for item in intervals}
    if interval_replicas != {0, 1}:
        errors.append("both_workers_did_not_execute_steps")
    cuda_replicas = {
        int(item["replica_id"])
        for item in intervals
        if float(item.get("cuda_elapsed_ms", 0.0)) > 0
    }
    if cuda_replicas != {0, 1}:
        errors.append("both_workers_did_not_execute_cuda_work")
    overlap_observed = any(
        max(0.0, min(left["step_finished"], right["step_finished"])
            - max(left["step_started"], right["step_started"])) > 0
        for index, left in enumerate(intervals)
        for right in intervals[index + 1:]
        if left["replica_id"] != right["replica_id"]
    )
    if require_overlap and not overlap_observed:
        errors.append("dual_gpu_execution_overlap_not_observed")
    if require_all_replicas_routed and (
        requests_per_replica is None
        or any(int(requests_per_replica.get(str(replica_id), 0)) == 0 for replica_id in (0, 1))
    ):
        errors.append("requests_not_routed_to_both_replicas")
    if final_snapshots is not None and any(
        int(item.get("used_kv_blocks", -1)) != 0
        or int(item.get("free_kv_blocks", -1)) != int(item.get("total_kv_blocks", -2))
        for item in final_snapshots
    ):
        errors.append("final_kv_state_not_released")
    return {
        "cell_status": "VALID_GPU_PROVENANCE" if not errors else "INVALID_GPU_PROVENANCE",
        "errors": errors,
        "worker_runtime": runtime,
        "model_residency": [residency_by_pid.get(pid) for pid in pids],
        "step_execution_intervals": intervals,
        "dual_gpu_overlap_observed": overlap_observed,
        "requests_per_replica": requests_per_replica,
        "final_free_kv_blocks": (
            [item["free_kv_blocks"] for item in final_snapshots]
            if final_snapshots is not None else None
        ),
    }


def require_valid_gpu_provenance(provenance: dict[str, object]) -> None:
    status = provenance.get("cell_status", provenance.get("status"))
    if status not in {"VALID_GPU_PROVENANCE", "PREFLIGHT_PROVENANCE_PASS"}:
        raise RuntimeError(f"Invalid GPU provenance: {provenance['errors']}")
