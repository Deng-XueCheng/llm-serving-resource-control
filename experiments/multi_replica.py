from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from concurrent.futures import ThreadPoolExecutor
from math import ceil
import time
from typing import Protocol

from nanovllm.engine.block_manager import PrefixCachePreview


@dataclass(frozen=True, slots=True)
class ReplicaStateSnapshot:
    replica_id: int
    total_kv_blocks: int
    free_kv_blocks: int
    used_kv_blocks: int
    kv_utilization: float
    waiting_queue_length: int
    running_requests: int
    oldest_waiting_age: int
    matched_prefix_blocks: int = 0
    active_prefix_blocks: int = 0

    def with_prefix(
        self,
        preview: PrefixCachePreview | None,
    ) -> "ReplicaStateSnapshot":
        if preview is None:
            return self
        return replace(
            self,
            matched_prefix_blocks=preview.matched_prefix_blocks,
            active_prefix_blocks=preview.active_shared_blocks,
        )


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    replica_id: int
    reason: str
    snapshots: tuple[ReplicaStateSnapshot, ...]
    estimated_kv_blocks: int


class RoundRobinRouter:
    def __init__(self, num_replicas: int):
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        self.num_replicas = num_replicas
        self._next = 0

    def route(
        self,
        snapshots: list[ReplicaStateSnapshot],
        *,
        estimated_kv_blocks: int,
    ) -> RoutingDecision:
        by_id = {snapshot.replica_id: snapshot for snapshot in snapshots}
        replica_id = self._next
        self._next = (self._next + 1) % self.num_replicas
        if replica_id not in by_id:
            raise ValueError("Missing round-robin replica snapshot")
        return RoutingDecision(
            replica_id,
            "round_robin",
            tuple(sorted(snapshots, key=lambda item: item.replica_id)),
            estimated_kv_blocks,
        )


class ResourceAwareRouter:
    def __init__(self, *, prefix_aware: bool = False):
        self.prefix_aware = prefix_aware

    def route(
        self,
        snapshots: list[ReplicaStateSnapshot],
        *,
        estimated_kv_blocks: int,
    ) -> RoutingDecision:
        if not snapshots:
            raise ValueError("At least one replica snapshot is required")
        feasible = [
            item
            for item in snapshots
            if item.free_kv_blocks + item.active_prefix_blocks
            >= estimated_kv_blocks
        ]
        candidates = feasible or snapshots
        selected = min(
            candidates,
            key=lambda item: (
                -item.matched_prefix_blocks if self.prefix_aware else 0,
                item.waiting_queue_length,
                item.kv_utilization,
                item.running_requests,
                item.oldest_waiting_age,
                item.replica_id,
            ),
        )
        if self.prefix_aware and selected.matched_prefix_blocks > 0:
            reason = "prefix_affinity"
        elif len(feasible) != len(snapshots):
            reason = "capacity_redirect"
        elif len({(
            item.total_kv_blocks,
            item.free_kv_blocks,
            item.used_kv_blocks,
            item.waiting_queue_length,
            item.running_requests,
            item.oldest_waiting_age,
        ) for item in snapshots}) == 1:
            reason = "deterministic_tie_break"
        else:
            reason = "resource_pressure"
        return RoutingDecision(
            selected.replica_id,
            reason,
            tuple(sorted(snapshots, key=lambda item: item.replica_id)),
            estimated_kv_blocks,
        )


class ReplicaEndpoint(Protocol):
    def snapshot(self) -> ReplicaStateSnapshot: ...

    def preview(self, prompt: list[int], max_tokens: int) -> PrefixCachePreview | None: ...

    def add(self, prompt: list[int], max_tokens: int) -> int: ...

    def step(self) -> list[dict[str, object]]: ...

    def is_finished(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class RoutedRequest:
    request_id: str
    replica_id: int
    local_seq_id: int


class MultiReplicaCoordinator:
    """Thin global control plane; all execution state remains replica-local."""

    def __init__(self, replicas: list[ReplicaEndpoint], router, *, block_size: int):
        if not replicas or block_size <= 0:
            raise ValueError("replicas and positive block_size are required")
        self.replicas = replicas
        self.router = router
        self.block_size = block_size
        self.route_events: list[dict[str, object]] = []
        self.resource_events: list[dict[str, object]] = []
        self.step_execution_intervals: list[dict[str, object]] = []
        self.requests: dict[tuple[int, int], str] = {}

    def route_request(
        self,
        request_id: str,
        prompt: list[int],
        max_tokens: int,
        *,
        observed_at: float,
    ) -> RoutedRequest:
        if not request_id or request_id in self.requests.values():
            raise ValueError("request_id must be unique and non-empty")
        snapshots = [replica.snapshot() for replica in self.replicas]
        enriched = [
            snapshot.with_prefix(
                self.replicas[snapshot.replica_id].preview(prompt, max_tokens)
            )
            for snapshot in snapshots
        ]
        estimated = ceil((len(prompt) + max_tokens - 1) / self.block_size)
        decision = self.router.route(enriched, estimated_kv_blocks=estimated)
        local_seq_id = self.replicas[decision.replica_id].add(prompt, max_tokens)
        self.requests[(decision.replica_id, local_seq_id)] = request_id
        self.route_events.append(
            {
                "schema_version": 1,
                "request_id": request_id,
                "observed_at": observed_at,
                "replica_id": decision.replica_id,
                "local_seq_id": local_seq_id,
                "reason": decision.reason,
                "estimated_kv_blocks": estimated,
                "snapshots": [asdict(item) for item in decision.snapshots],
                "queue_imbalance": _queue_imbalance(decision.snapshots),
                "kv_imbalance": _kv_imbalance(decision.snapshots),
            }
        )
        return RoutedRequest(request_id, decision.replica_id, local_seq_id)

    def step(self, *, observed_at: float) -> list[dict[str, object]]:
        events = []
        active = [
            (replica_id, replica)
            for replica_id, replica in enumerate(self.replicas)
            if not replica.is_finished()
        ]
        with ThreadPoolExecutor(max_workers=len(active) or 1) as executor:
            results = []
            for replica_id, replica in active:
                operation = getattr(replica, "audit_step", None)
                results.append((replica_id, executor.submit(operation or replica.step)))
            for replica_id, future in results:
                result = future.result()
                if isinstance(result, dict) and "step_started" in result:
                    self.step_execution_intervals.append({
                        key: result[key] for key in (
                            "replica_id", "worker_pid", "step_started", "step_finished",
                            "cuda_elapsed_ms",
                        )
                    })
                    step_events = result["events"]
                else:
                    step_events = result
                for event in step_events:
                    item = dict(event)
                    item["replica_id"] = replica_id
                    seq_id = int(item["seq_id"])
                    item["request_id"] = self.requests[(replica_id, seq_id)]
                    events.append(item)
        snapshots = tuple(replica.snapshot() for replica in self.replicas)
        self.resource_events.append(
            {
                "schema_version": 1,
                "observed_at": observed_at,
                "snapshots": [asdict(item) for item in snapshots],
                "queue_imbalance": _queue_imbalance(snapshots),
                "kv_imbalance": _kv_imbalance(snapshots),
            }
        )
        return events

    def is_finished(self) -> bool:
        return all(replica.is_finished() for replica in self.replicas)


def _queue_imbalance(snapshots) -> int:
    loads = [item.waiting_queue_length + item.running_requests for item in snapshots]
    return max(loads) - min(loads) if loads else 0


def _kv_imbalance(snapshots) -> float:
    values = [item.kv_utilization for item in snapshots]
    return max(values) - min(values) if values else 0.0


class ProcessReplicaEndpoint:
    def __init__(self, connection, process, replica_id: int):
        self.connection = connection
        self.process = process
        self.replica_id = replica_id

    def _call(self, operation: str, **payload):
        self.connection.send({"operation": operation, **payload})
        response = self.connection.recv()
        if not response["ok"]:
            raise RuntimeError(
                f"Replica {self.replica_id} {operation} failed: {response['error']}"
            )
        return response.get("value")

    def snapshot(self) -> ReplicaStateSnapshot:
        return ReplicaStateSnapshot(**self._call("snapshot"))

    def preview(self, prompt: list[int], max_tokens: int):
        value = self._call("preview", prompt=prompt, max_tokens=max_tokens)
        return PrefixCachePreview(**value)

    def add(self, prompt: list[int], max_tokens: int) -> int:
        return self._call("add", prompt=prompt, max_tokens=max_tokens)

    def step(self) -> list[dict[str, object]]:
        return self._call("step")

    def is_finished(self) -> bool:
        return self._call("is_finished")

    def runtime_info(self) -> dict[str, object]:
        return self._call("runtime_info")

    def audit_step(self) -> dict[str, object]:
        return self._call("audit_step")

    def audit_prefill_export(self, prompt: list[int], max_tokens: int) -> dict:
        return self._call(
            "audit_prefill_export", prompt=prompt, max_tokens=max_tokens
        )

    def prefill_export(self, prompt: list[int], max_tokens: int) -> dict:
        return self._call(
            "prefill_export", prompt=prompt, max_tokens=max_tokens
        )

    def import_decode(self, transfer: dict) -> dict:
        return self._call("import_decode", transfer=transfer)

    def close(self) -> None:
        try:
            self._call("close")
        finally:
            self.process.join(timeout=30)
            self.connection.close()


def replica_worker(
    connection,
    *,
    replica_id: int,
    gpu_id: int,
    model: str,
    engine_kwargs: dict,
    temperature: float,
    ignore_eos: bool,
    sampling_seed: int = 0,
) -> None:
    import os
    import traceback

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    engine = None
    try:
        import torch

        torch.manual_seed(sampling_seed)
        from nanovllm.engine.llm_engine import LLMEngine
        from nanovllm.sampling_params import SamplingParams

        engine = LLMEngine(model, **engine_kwargs)
        while True:
            request = connection.recv()
            operation = request["operation"]
            try:
                if operation == "snapshot":
                    value = engine.replica_state_snapshot(replica_id)
                elif operation == "preview":
                    value = asdict(engine.preview_prefix_cache(
                        request["prompt"], request["max_tokens"]
                    ))
                elif operation == "add":
                    value = engine.add_request(
                        request["prompt"],
                        SamplingParams(
                            temperature=temperature,
                            max_tokens=request["max_tokens"],
                            ignore_eos=ignore_eos,
                        ),
                    )
                elif operation == "step":
                    value = [asdict(item) for item in engine.step_with_events().events]
                elif operation == "audit_step":
                    step_started = time.perf_counter()
                    cuda_started = torch.cuda.Event(enable_timing=True)
                    cuda_finished = torch.cuda.Event(enable_timing=True)
                    cuda_started.record()
                    events = [asdict(item) for item in engine.step_with_events().events]
                    cuda_finished.record()
                    cuda_finished.synchronize()
                    value = {
                        "replica_id": replica_id,
                        "worker_pid": os.getpid(),
                        "step_started": step_started,
                        "step_finished": time.perf_counter(),
                        "cuda_elapsed_ms": cuda_started.elapsed_time(cuda_finished),
                        "events": events,
                    }
                elif operation == "runtime_info":
                    properties = torch.cuda.get_device_properties(
                        torch.cuda.current_device()
                    )
                    value = {
                        "replica_id": replica_id,
                        "worker_pid": os.getpid(),
                        "configured_gpu_id": gpu_id,
                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "torch_cuda_current_device": torch.cuda.current_device(),
                        "torch_cuda_device_count": torch.cuda.device_count(),
                        "device_name": properties.name,
                        "device_uuid": str(properties.uuid),
                        "memory_allocated_bytes": torch.cuda.memory_allocated(),
                        "memory_reserved_bytes": torch.cuda.memory_reserved(),
                    }
                elif operation == "prefill_export":
                    value = engine.prefill_export(
                        request["prompt"],
                        SamplingParams(
                            temperature=temperature,
                            max_tokens=request["max_tokens"],
                            ignore_eos=ignore_eos,
                        ),
                    )
                elif operation == "audit_prefill_export":
                    step_started = time.perf_counter()
                    cuda_started = torch.cuda.Event(enable_timing=True)
                    cuda_finished = torch.cuda.Event(enable_timing=True)
                    cuda_started.record()
                    transfer = engine.prefill_export(
                        request["prompt"],
                        SamplingParams(
                            temperature=temperature,
                            max_tokens=request["max_tokens"],
                            ignore_eos=ignore_eos,
                        ),
                    )
                    cuda_finished.record()
                    cuda_finished.synchronize()
                    value = {
                        "replica_id": replica_id,
                        "worker_pid": os.getpid(),
                        "step_started": step_started,
                        "step_finished": time.perf_counter(),
                        "cuda_elapsed_ms": cuda_started.elapsed_time(cuda_finished),
                        "transfer": transfer,
                    }
                elif operation == "import_decode":
                    value = engine.import_decode(request["transfer"])
                elif operation == "is_finished":
                    value = engine.is_finished()
                elif operation == "close":
                    engine.exit()
                    engine = None
                    connection.send({"ok": True, "value": None})
                    return
                else:
                    raise ValueError(f"Unknown replica operation: {operation}")
                connection.send({"ok": True, "value": value})
            except Exception:
                connection.send({"ok": False, "error": traceback.format_exc()})
    except Exception:
        connection.send({"ok": False, "error": traceback.format_exc()})
    finally:
        if engine is not None:
            engine.exit()
        connection.close()
