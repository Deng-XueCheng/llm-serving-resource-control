from __future__ import annotations

import os
import unittest

from experiments.multi_replica import (
    MultiReplicaCoordinator,
    ReplicaStateSnapshot,
    ResourceAwareRouter,
    RoundRobinRouter,
)
from experiments.run_multi_replica import start_process_on_gpu
from experiments.gpu_provenance import (
    validate_gpu_preflight,
    validate_gpu_provenance,
)


class FakeReplica:
    def __init__(self, replica_id: int, capacity: int = 8):
        self.replica_id = replica_id
        self.capacity = capacity
        self.active = {}
        self.next_id = 0

    def snapshot(self):
        used = len(self.active)
        return state(
            self.replica_id,
            free=self.capacity - used,
            used=used,
            waiting=used,
        )

    def preview(self, prompt, max_tokens):
        return None

    def add(self, prompt, max_tokens):
        seq_id = self.next_id
        self.next_id += 1
        self.active[seq_id] = max_tokens
        return seq_id

    def step(self):
        events = []
        for seq_id in tuple(self.active):
            self.active[seq_id] -= 1
            finished = self.active[seq_id] == 0
            events.append({"seq_id": seq_id, "finished": finished})
            if finished:
                del self.active[seq_id]
        return events

    def is_finished(self):
        return not self.active


def state(
    replica_id: int,
    *,
    free: int = 8,
    used: int = 0,
    waiting: int = 0,
    running: int = 0,
    age: int = 0,
    prefix: int = 0,
) -> ReplicaStateSnapshot:
    return ReplicaStateSnapshot(
        replica_id=replica_id,
        total_kv_blocks=free + used,
        free_kv_blocks=free,
        used_kv_blocks=used,
        kv_utilization=used / (free + used),
        waiting_queue_length=waiting,
        running_requests=running,
        oldest_waiting_age=age,
        matched_prefix_blocks=prefix,
        active_prefix_blocks=prefix,
    )


class MultiReplicaRouterTests(unittest.TestCase):
    def test_gpu_preflight_accepts_distinct_pid_uuid_and_ports(self):
        runtime = [
            {"worker_pid": 10, "device_uuid": "aaa"},
            {"worker_pid": 11, "device_uuid": "bbb"},
        ]
        residency = [
            {"pid": 10, "gpu_uuid": "GPU-aaa", "used_memory_mib": 100},
            {"pid": 11, "gpu_uuid": "GPU-bbb", "used_memory_mib": 100},
        ]
        result = validate_gpu_preflight(
            runtime, residency, expected_worker_count=2,
            requested_gpu_ids=[0, 1], distributed_ports=[20000, 20001],
        )
        self.assertEqual(result["status"], "PREFLIGHT_PROVENANCE_PASS")

    def test_runtime_provenance_requires_actual_cuda_execution(self):
        runtime = [
            {"worker_pid": 10, "device_uuid": "aaa"},
            {"worker_pid": 11, "device_uuid": "bbb"},
        ]
        residency = [
            {"pid": 10, "gpu_uuid": "GPU-aaa", "used_memory_mib": 100},
            {"pid": 11, "gpu_uuid": "GPU-bbb", "used_memory_mib": 100},
        ]
        intervals = [
            {"replica_id": 0, "step_started": 1.0, "step_finished": 2.0,
             "cuda_elapsed_ms": 0.0},
            {"replica_id": 1, "step_started": 1.5, "step_finished": 2.5,
             "cuda_elapsed_ms": 0.0},
        ]
        result = validate_gpu_provenance(
            runtime, residency, intervals, require_overlap=True
        )
        self.assertEqual(result["cell_status"], "INVALID_GPU_PROVENANCE")
        self.assertIn("both_workers_did_not_execute_cuda_work", result["errors"])

    def test_gpu_provenance_fails_closed_for_same_physical_gpu(self):
        runtime = [
            {"worker_pid": 10, "device_uuid": "aaa"},
            {"worker_pid": 11, "device_uuid": "aaa"},
        ]
        residency = [
            {"pid": 10, "gpu_uuid": "GPU-aaa", "used_memory_mib": 100},
            {"pid": 11, "gpu_uuid": "GPU-aaa", "used_memory_mib": 100},
        ]
        intervals = [
            {"replica_id": 0, "step_started": 1.0, "step_finished": 2.0},
            {"replica_id": 1, "step_started": 1.5, "step_finished": 2.5},
        ]
        result = validate_gpu_provenance(
            runtime, residency, intervals, require_overlap=True
        )
        self.assertEqual(result["cell_status"], "INVALID_GPU_PROVENANCE")
        self.assertIn("worker_physical_gpu_uuids_not_distinct", result["errors"])

    def test_worker_device_visibility_is_set_before_spawn_and_restored(self):
        observed = []

        class FakeProcess:
            def start(self):
                observed.append(os.environ.get("CUDA_VISIBLE_DEVICES"))

        previous = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        try:
            start_process_on_gpu(FakeProcess(), 0)
            start_process_on_gpu(FakeProcess(), 1)
            self.assertEqual(observed, ["0", "1"])
            self.assertNotIn("CUDA_VISIBLE_DEVICES", os.environ)
        finally:
            if previous is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous

    def test_round_robin_routes_in_order(self):
        router = RoundRobinRouter(2)
        snapshots = [state(0), state(1)]
        self.assertEqual(
            [router.route(snapshots, estimated_kv_blocks=1).replica_id for _ in range(5)],
            [0, 1, 0, 1, 0],
        )

    def test_resource_router_prefers_lower_queue_and_kv_pressure(self):
        decision = ResourceAwareRouter().route(
            [state(0, free=2, used=6, waiting=3), state(1, free=7, used=1)],
            estimated_kv_blocks=2,
        )
        self.assertEqual(decision.replica_id, 1)
        self.assertEqual(decision.reason, "resource_pressure")

    def test_equal_state_uses_deterministic_replica_id(self):
        decision = ResourceAwareRouter().route(
            [state(1), state(0)], estimated_kv_blocks=1
        )
        self.assertEqual(decision.replica_id, 0)
        self.assertEqual(decision.reason, "deterministic_tie_break")

    def test_prefix_affinity_prefers_more_active_reuse(self):
        decision = ResourceAwareRouter(prefix_aware=True).route(
            [state(0, prefix=0), state(1, prefix=2)], estimated_kv_blocks=3
        )
        self.assertEqual(decision.replica_id, 1)
        self.assertEqual(decision.reason, "prefix_affinity")

    def test_saturated_replica_redirects_to_feasible_replica(self):
        decision = ResourceAwareRouter().route(
            [state(0, free=1, used=7), state(1, free=4, used=4)],
            estimated_kv_blocks=3,
        )
        self.assertEqual(decision.replica_id, 1)

    def test_snapshot_values_are_independent(self):
        first = state(0, free=1, used=7, waiting=2)
        second = state(1, free=8, used=0)
        self.assertNotEqual(first.free_kv_blocks, second.free_kv_blocks)
        self.assertNotEqual(first.waiting_queue_length, second.waiting_queue_length)

    def test_coordinator_keeps_local_ids_and_state_isolated_and_drains(self):
        replicas = [FakeReplica(0), FakeReplica(1)]
        coordinator = MultiReplicaCoordinator(
            replicas, RoundRobinRouter(2), block_size=2
        )
        routed = [
            coordinator.route_request(f"r{i}", [1, 2], 2, observed_at=0.0)
            for i in range(4)
        ]
        self.assertEqual([item.replica_id for item in routed], [0, 1, 0, 1])
        self.assertEqual([item.local_seq_id for item in routed], [0, 0, 1, 1])
        while not coordinator.is_finished():
            coordinator.step(observed_at=1.0)
        self.assertEqual([replica.snapshot().used_kv_blocks for replica in replicas], [0, 0])
        self.assertTrue(all("snapshots" in event for event in coordinator.route_events))
        event = coordinator.resource_events[-1]
        snapshots = event["snapshots"]
        expected = abs(snapshots[0]["kv_utilization"] - snapshots[1]["kv_utilization"])
        self.assertEqual(event["kv_imbalance"], expected)


if __name__ == "__main__":
    unittest.main()
