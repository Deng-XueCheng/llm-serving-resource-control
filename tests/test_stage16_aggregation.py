from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from experiments.aggregate_stage16 import (
    fairness_metrics,
    validate_bounded_exit_active_set,
    validate_manifest,
)
from experiments.benchmark.calibration_evidence import validate_steps
from experiments.generate_stage16_configs import execution_fingerprint


REPO_ROOT = Path(__file__).resolve().parents[1]


def frozen_manifest() -> dict:
    return json.loads(
        (
            REPO_ROOT
            / "experiments/configs/stage16_diagnostic_r2_matrix.json"
        ).read_text(encoding="utf-8")
    )


def validation_manifest() -> dict:
    manifest = frozen_manifest()
    # The frozen fingerprint belongs to the legacy execution commit.  Unit
    # tests validate the same contract against the clean snapshot source
    # without rewriting the formal artifact on disk.
    manifest["execution_fingerprint"] = execution_fingerprint()
    return manifest


def fairness_state(
    seq_id: int,
    request_id: str,
    *,
    last_progress_step: int | None,
    current_step: int,
) -> dict:
    return {
        "seq_id": seq_id,
        "location": "running",
        "pending_recompute": False,
        "had_emitted_token": True,
        "waiting_age_steps": 0,
        "last_progress_step": last_progress_step,
        "steps_since_last_progress": (
            current_step - last_progress_step
            if last_progress_step is not None
            else None
        ),
        "slo_deadline_at": 10.0,
        "slo_deadline_kind": "itl_slo_deadline",
        "request_id": request_id,
    }


def decode_event(
    request_id: str,
    seq_id: int,
    *,
    step: int,
    previous: int,
) -> dict:
    return {
        "request_id": request_id,
        "seq_id": seq_id,
        "phase": "decode",
        "num_scheduled_tokens": 1,
        "emitted_token": True,
        "finished": False,
        "scheduler_step_index": step,
        "previous_progress_step": previous,
        "progress_gap_steps": step - previous,
        "had_emitted_token_before": True,
    }


def scheduler_snapshot(
    *,
    step: int,
    mode: str,
    selected: list[int],
    states: list[dict],
    episode_step: int = 0,
    exit_reason: str | None = None,
    resource_pressure: bool = True,
) -> dict:
    return {
        "policy": "recompute_aware_bounded",
        "state": "critical",
        "mode": mode,
        "kv_total_blocks": 8,
        "kv_used_blocks_before": 8,
        "kv_free_blocks_before": 0,
        "kv_used_blocks_after": 8,
        "kv_free_blocks_after": 0,
        "running_ids_before": [1, 2],
        "waiting_ids_before": [],
        "running_ids_after": [1, 2],
        "waiting_ids_after": [],
        "selected_decode_ids": selected,
        "selected_prefill_ids": [],
        "oldest_waiting_age": 0,
        "resident_costs": [],
        "preemptions": [],
        "selected_decode_request_ids": [
            "interactive-000" if seq_id == 1 else "interactive-001"
            for seq_id in selected
        ],
        "selected_prefill_request_ids": [],
        "scheduler_step_index": step,
        "active_progress_states": states,
        "drain_slo_watch": (
            [
                {
                    "seq_id": state["seq_id"],
                    "entry_progress_step": state["last_progress_step"],
                    "deadline_at": state["slo_deadline_at"],
                    "deadline_kind": state["slo_deadline_kind"],
                    "request_id": state["request_id"],
                }
                for state in states
                if (episode_step or exit_reason)
            ]
        ),
        "drain_episode_id": 1 if episode_step or exit_reason else None,
        "drain_episode_step": episode_step,
        "drain_tokens": episode_step,
        "drain_episode_started_at": 1.0 if episode_step or exit_reason else None,
        "drain_exit_reason": exit_reason,
        "fairness_trigger_reason": None,
        "resource_pressure": resource_pressure,
        "slo_guard_seq_id": None,
        "slo_guard_entry_progress_step": None,
        "slo_guard_deadline_at": None,
        "slo_guard_triggered_at": None,
    }


def synthetic_steps() -> list[dict]:
    prefill_events = []
    prefill_states = []
    for seq_id, request_id in ((1, "interactive-000"), (2, "interactive-001")):
        prefill_events.append(
            {
                "request_id": request_id,
                "seq_id": seq_id,
                "phase": "prefill",
                "num_scheduled_tokens": 4,
                "emitted_token": True,
                "finished": False,
                "prefill_kind": "initial_prefill",
                "actual_recompute_tokens": 0,
                "resumed": False,
                "scheduler_step_index": 0,
                "previous_progress_step": None,
                "progress_gap_steps": None,
                "had_emitted_token_before": False,
            }
        )
        prefill_states.append(
            {
                "seq_id": seq_id,
                "location": "waiting",
                "pending_recompute": False,
                "had_emitted_token": False,
                "waiting_age_steps": 0,
                "last_progress_step": None,
                "steps_since_last_progress": None,
                "slo_deadline_at": 10.0,
                "slo_deadline_kind": "ttft_slo_deadline",
                "request_id": request_id,
            }
        )
    steps = [
        {
            "step_index": 0,
            "started_at": 0.0,
            "finished_at": 0.01,
            "duration_ms": 10.0,
            "phase": "prefill",
            "num_scheduled_tokens": 8,
            "events": prefill_events,
            "schema_version": 3,
            "scheduler": {
                **scheduler_snapshot(
                    step=0,
                    mode="prefill",
                    selected=[],
                    states=prefill_states,
                ),
                "running_ids_before": [],
                "waiting_ids_before": [1, 2],
                "running_ids_after": [1, 2],
                "selected_prefill_ids": [1, 2],
                "selected_prefill_request_ids": [
                    "interactive-000",
                    "interactive-001",
                ],
            },
        }
    ]
    prior = {1: 0, 2: 0}
    for step_index in (1, 2, 3):
        selected = [2] if step_index < 3 else [1, 2]
        states = [
            fairness_state(
                seq_id,
                request_id,
                last_progress_step=prior[seq_id],
                current_step=step_index,
            )
            for seq_id, request_id in (
                (1, "interactive-000"),
                (2, "interactive-001"),
            )
        ]
        events = [
            decode_event(
                "interactive-000" if seq_id == 1 else "interactive-001",
                seq_id,
                step=step_index,
                previous=prior[seq_id],
            )
            for seq_id in selected
        ]
        mode = "drain_decode" if step_index < 3 else "baseline_decode"
        steps.append(
            {
                "step_index": step_index,
                "started_at": step_index * 0.01,
                "finished_at": (step_index + 1) * 0.01,
                "duration_ms": 10.0,
                "phase": "decode",
                "num_scheduled_tokens": len(events),
                "events": events,
                "schema_version": 3,
                "scheduler": scheduler_snapshot(
                    step=step_index,
                    mode=mode,
                    selected=selected,
                    states=states,
                    episode_step=(step_index if step_index < 3 else 2),
                    exit_reason=(
                        "resource_released" if step_index == 3 else None
                    ),
                    resource_pressure=step_index < 3,
                ),
            }
        )
        if step_index >= 1:
            steps[-1]["scheduler"]["drain_slo_watch"] = [
                {
                    "seq_id": seq_id,
                    "entry_progress_step": 0,
                    "deadline_at": 10.0,
                    "deadline_kind": "itl_slo_deadline",
                    "request_id": request_id,
                }
                for seq_id, request_id in (
                    (1, "interactive-000"),
                    (2, "interactive-001"),
                )
            ]
        for seq_id in selected:
            prior[seq_id] = step_index
    return steps


class Stage16AggregationTests(unittest.TestCase):
    def test_fairness_metrics_recompute_progress_and_episode(self) -> None:
        metrics = fairness_metrics(synthetic_steps(), bounded=True)

        self.assertEqual(
            metrics["post_token_progress_gap_steps"]["max"],
            3.0,
        )
        self.assertEqual(
            metrics["per_request_max_post_token_progress_gap_steps"]["max"],
            3.0,
        )
        self.assertEqual(metrics["drain_episode_count"], 1)
        self.assertEqual(
            metrics["drain_episode_length_steps"]["max"],
            2.0,
        )
        self.assertEqual(
            metrics["drain_exit_reasons"],
            {"resource_released": 1},
        )

    def test_progress_or_fairness_tamper_fails_closed(self) -> None:
        steps = synthetic_steps()
        steps[3]["events"][0]["progress_gap_steps"] = 2
        with self.assertRaisesRegex(ValueError, "progress event"):
            fairness_metrics(steps, bounded=True)

        steps = synthetic_steps()
        steps[2]["scheduler"]["active_progress_states"][0][
            "last_progress_step"
        ] = 1
        with self.assertRaisesRegex(ValueError, "fairness state"):
            fairness_metrics(steps, bounded=True)

    def test_bounded_drain_invariant_rejects_over_budget_step(self) -> None:
        steps = synthetic_steps()
        steps[2]["scheduler"]["drain_episode_step"] = 17
        with self.assertRaisesRegex(ValueError, "drain invariant"):
            fairness_metrics(steps, bounded=True)

    def test_manifest_threshold_tamper_fails_closed(self) -> None:
        manifest = validation_manifest()
        validate_manifest(manifest)

        tampered = copy.deepcopy(manifest)
        tampered["acceptance"]["itl_p99_reduction_min"] = 0.0
        with self.assertRaisesRegex(ValueError, "manifest contract"):
            validate_manifest(tampered)

    def test_manifest_execution_or_trace_tamper_fails_closed(self) -> None:
        manifest = validation_manifest()

        tampered = copy.deepcopy(manifest)
        tampered["execution_fingerprint"]["scheduler_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "execution fingerprint"):
            validate_manifest(tampered)

        tampered = copy.deepcopy(manifest)
        tampered["traces"][0]["trace_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "trace hash"):
            validate_manifest(tampered)

    def test_emitted_event_timestamp_must_match_request_artifact(self) -> None:
        steps = synthetic_steps()
        steps[-1]["events"][0]["finished"] = True
        steps[-1]["events"][1]["finished"] = True
        requests = {
            "interactive-000": {
                "token_timestamps": [0.01, 0.03],
                "terminal_at": 0.04,
            },
            "interactive-001": {
                "token_timestamps": [0.01, 0.02, 0.03, 0.04],
                "terminal_at": 0.04,
            },
        }

        with self.assertRaisesRegex(ValueError, "timestamp"):
            validate_steps(steps, requests, len(steps))

    def test_cross_step_active_set_tamper_fails_closed(self) -> None:
        steps = synthetic_steps()
        steps[0]["scheduler"]["running_ids_after"] = [1]

        with self.assertRaisesRegex(ValueError, "transition"):
            fairness_metrics(steps, bounded=True)

    def test_partial_prefill_may_remain_waiting(self) -> None:
        steps = synthetic_steps()[:1]
        step = steps[0]
        step["events"] = [step["events"][0]]
        step["events"][0]["emitted_token"] = False
        step["events"][0]["finished"] = False
        step["num_scheduled_tokens"] = 4
        scheduler = step["scheduler"]
        scheduler["running_ids_after"] = []
        scheduler["waiting_ids_before"] = [1]
        scheduler["waiting_ids_after"] = [1]
        scheduler["selected_prefill_ids"] = [1]
        scheduler["selected_prefill_request_ids"] = ["interactive-000"]
        scheduler["active_progress_states"] = [
            scheduler["active_progress_states"][0]
        ]

        metrics = fairness_metrics(steps, bounded=False)

        self.assertEqual(metrics["drain_episode_count"], 0)

    def test_schema_v2_unbounded_drain_does_not_require_bounded_fields(
        self,
    ) -> None:
        steps = synthetic_steps()
        bounded_scheduler_fields = (
            "scheduler_step_index",
            "active_progress_states",
            "drain_slo_watch",
            "drain_episode_id",
            "drain_episode_step",
            "drain_tokens",
            "drain_episode_started_at",
            "drain_exit_reason",
            "fairness_trigger_reason",
            "resource_pressure",
            "slo_guard_seq_id",
            "slo_guard_entry_progress_step",
            "slo_guard_deadline_at",
            "slo_guard_triggered_at",
        )
        bounded_event_fields = (
            "scheduler_step_index",
            "previous_progress_step",
            "progress_gap_steps",
            "had_emitted_token_before",
        )
        for step in steps:
            step["schema_version"] = 2
            for field in bounded_scheduler_fields:
                step["scheduler"].pop(field)
            for event in step["events"]:
                for field in bounded_event_fields:
                    event.pop(field)

        metrics = fairness_metrics(steps, bounded=False)

        self.assertEqual(metrics["drain_episode_count"], 1)

    def test_initial_waiting_age_cannot_be_episode_exit(self) -> None:
        steps = synthetic_steps()
        steps[3]["scheduler"]["drain_exit_reason"] = "initial_waiting_age"

        with self.assertRaisesRegex(ValueError, "exit reason"):
            fairness_metrics(steps, bounded=True)

    def test_slo_exit_must_match_episode_entry_watch(self) -> None:
        steps = synthetic_steps()
        scheduler = steps[3]["scheduler"]
        scheduler["drain_exit_reason"] = "itl_slo_deadline"
        scheduler["slo_guard_seq_id"] = 1
        scheduler["slo_guard_entry_progress_step"] = 1
        scheduler["slo_guard_deadline_at"] = 11.0
        scheduler["slo_guard_triggered_at"] = 11.0

        with self.assertRaisesRegex(ValueError, "SLO guard evidence"):
            fairness_metrics(steps, bounded=True)

    def test_empty_exit_reason_requires_matching_active_set(self) -> None:
        with self.assertRaisesRegex(ValueError, "resident-empty"):
            validate_bounded_exit_active_set(
                "resident_empty",
                {1},
                set(),
            )

        with self.assertRaisesRegex(ValueError, "waiting-empty"):
            validate_bounded_exit_active_set(
                "waiting_empty",
                set(),
                {1},
            )


if __name__ == "__main__":
    unittest.main()
