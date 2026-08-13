from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.aggregate_stage15 import (
    CODE_FINGERPRINT_KEYS,
    load_validated_stage15_run,
)
from experiments.benchmark.calibration_evidence import distribution, sha256_file
from experiments.generate_stage16_configs import (
    CAPACITIES,
    MAX_DRAIN_STEPS,
    POLICIES,
    REPO_ROOT,
    REVISION,
    SEEDS,
    WAITING_AGE_LIMIT,
    EXECUTION_FINGERPRINT_KEYS,
    execution_fingerprint,
    normalized_config,
    run_id as expected_run_id,
    source_config_path,
    trace_binding,
)


COMPLETION_GATES = {
    "status": "passed",
    "all_requests_finished": True,
    "terminal_reconciled": True,
    "timed_out": False,
    "oom_detected": False,
    "final_kv_used_blocks": 0,
    "raw_summary_counters_match": True,
}
ACCEPTANCE = {
    "fairness_reduction_min": 0.50,
    "itl_p99_reduction_min": 0.30,
    "resource_reduction_min": 0.50,
    "ttft_p99_regression_max": 0.10,
    "improved_matched_pairs_min": 6,
    "matched_pairs": 9,
}
CONTROLLED_DIFFERENCE = [
    "engine.scheduler_policy",
    "engine.max_drain_steps",
    "engine.waiting_age_limit",
    "engine.ttft_slo_ms",
    "engine.itl_slo_ms",
    "output.run_id",
]


def validate_manifest(manifest: dict[str, Any]) -> None:
    if (
        manifest.get("schema_version") != 2
        or manifest.get("revision") != REVISION
        or tuple(manifest.get("capacities", ())) != CAPACITIES
        or tuple(manifest.get("seeds", ())) != SEEDS
        or manifest.get("policies") != list(POLICIES)
        or manifest.get("required_cells") != 27
        or manifest.get("controlled_difference") != CONTROLLED_DIFFERENCE
        or manifest.get("bounded_contract")
        != {
            "max_drain_steps": MAX_DRAIN_STEPS,
            "waiting_age_limit": WAITING_AGE_LIMIT,
            "ttft_slo_source": "slo.ttft_slo_ms",
            "itl_slo_source": "slo.itl_slo_ms",
            "episode_entry_positive_slack_only": True,
        }
        or manifest.get("completion_gates") != COMPLETION_GATES
        or manifest.get("acceptance") != ACCEPTANCE
        or set(manifest.get("execution_fingerprint", {}))
        != set(EXECUTION_FINGERPRINT_KEYS)
        or manifest.get("execution_fingerprint") != execution_fingerprint()
        or len(manifest.get("cells", ())) != 27
    ):
        if manifest.get("execution_fingerprint") != execution_fingerprint():
            raise ValueError("Stage 16 execution fingerprint differs")
        raise ValueError("Stage 16 manifest contract differs")
    expected_traces = [
        trace_binding(
            json.loads(
                source_config_path(CAPACITIES[0], seed).read_text(
                    encoding="utf-8"
                )
            ),
            seed,
        )
        for seed in SEEDS
    ]
    if manifest.get("traces") != expected_traces:
        raise ValueError("Stage 16 trace hash binding differs")
    expected_cells = {
        (capacity, seed, policy)
        for capacity in CAPACITIES
        for seed in SEEDS
        for policy in POLICIES
    }
    actual_cells = {
        (cell.get("capacity"), cell.get("seed"), cell.get("policy"))
        for cell in manifest["cells"]
    }
    if actual_cells != expected_cells:
        raise ValueError("Stage 16 manifest cells differ")
    expected_keys = {
        "capacity",
        "seed",
        "policy",
        "run_id",
        "config_path",
        "source_config_path",
        "config_sha256",
        "source_config_sha256",
        "trace_path",
        "trace_sha256",
    }
    if any(set(cell) != expected_keys for cell in manifest["cells"]):
        raise ValueError("Stage 16 manifest cell schema differs")
    traces_by_seed = {trace["seed"]: trace for trace in expected_traces}
    for cell in manifest["cells"]:
        run_id = expected_run_id(
            cell["capacity"], cell["policy"], cell["seed"]
        )
        expected_config_path = (
            Path("experiments/configs") / REVISION / f"{run_id}.json"
        ).as_posix()
        expected_source_path = source_config_path(
            cell["capacity"], cell["seed"]
        ).relative_to(REPO_ROOT).as_posix()
        trace = traces_by_seed[cell["seed"]]
        if (
            cell["run_id"] != run_id
            or Path(cell["config_path"]).as_posix() != expected_config_path
            or Path(cell["source_config_path"]).as_posix()
            != expected_source_path
            or cell["trace_path"] != trace["trace_path"]
            or cell["trace_sha256"] != trace["trace_sha256"]
        ):
            raise ValueError("Stage 16 manifest cell binding differs")


def expected_config_for_cell(
    source: dict[str, Any],
    *,
    policy_name: str,
    run_id: str,
) -> dict[str, Any]:
    expected = copy.deepcopy(source)
    expected["engine"]["scheduler_policy"] = POLICIES[policy_name]
    if policy_name == "bounded":
        expected["engine"].update(
            {
                "max_drain_steps": MAX_DRAIN_STEPS,
                "waiting_age_limit": WAITING_AGE_LIMIT,
                "ttft_slo_ms": expected["slo"]["ttft_slo_ms"],
                "itl_slo_ms": expected["slo"]["itl_slo_ms"],
            }
        )
    expected["output"]["run_id"] = run_id
    return expected


def validate_bounded_exit_active_set(
    reason: str,
    running_before: set[int],
    waiting_before: set[int],
) -> None:
    if reason == "resident_empty" and running_before:
        raise ValueError("Stage 16 resident-empty exit condition differs")
    if reason == "waiting_empty" and waiting_before:
        raise ValueError("Stage 16 waiting-empty exit condition differs")


def fairness_metrics(
    steps: list[dict[str, Any]],
    *,
    bounded: bool,
) -> dict[str, Any]:
    last_progress: dict[str, int] = {}
    last_token: dict[str, int] = {}
    seen_token: set[str] = set()
    waiting_age: dict[int, int] = {}
    progress_gaps: list[float] = []
    post_token_progress_gaps: list[float] = []
    token_gaps: list[float] = []
    waiting_age_samples: list[float] = []
    oldest_waiting_ages: list[float] = []
    per_request_max_progress: dict[str, int] = {}
    per_request_max_post_token_progress: dict[str, int] = {}
    per_request_max_token_gap: dict[str, int] = {}
    exit_reasons: Counter[str] = Counter()
    episodes: list[dict[str, float | int]] = []
    active_episode: dict[str, float | int] | None = None
    request_id_by_seq_id: dict[int, str] = {}
    pending_recompute: set[int] = set()
    previous_running_after: set[int] | None = None
    previous_waiting_after: set[int] | None = None
    previous_finished: set[int] = set()
    ever_active: set[int] = set()
    for step in steps:
        for event in step["events"]:
            previous = request_id_by_seq_id.setdefault(
                event["seq_id"], event["request_id"]
            )
            if previous != event["request_id"]:
                raise ValueError("Stage 16 seq/request mapping differs")

    for expected_index, step in enumerate(steps):
        if step["step_index"] != expected_index:
            raise ValueError("Stage 16 step indexes are not contiguous")
        scheduler = step["scheduler"]
        running_before = set(scheduler["running_ids_before"])
        waiting_before_set = set(scheduler["waiting_ids_before"])
        running_after = set(scheduler["running_ids_after"])
        waiting_after_set = set(scheduler["waiting_ids_after"])
        for field in (
            "running_ids_before",
            "waiting_ids_before",
            "running_ids_after",
            "waiting_ids_after",
            "selected_decode_ids",
            "selected_prefill_ids",
        ):
            if len(scheduler[field]) != len(set(scheduler[field])):
                raise ValueError("Stage 16 scheduler active-set duplicates")
        if (
            running_before & waiting_before_set
            or running_after & waiting_after_set
        ):
            raise ValueError("Stage 16 scheduler active sets overlap")
        if previous_running_after is not None:
            if running_before != previous_running_after - previous_finished:
                raise ValueError("Stage 16 running transition differs")
            if not previous_waiting_after.issubset(waiting_before_set):
                raise ValueError("Stage 16 waiting transition differs")
            newly_admitted = waiting_before_set - previous_waiting_after
            if newly_admitted & ever_active:
                raise ValueError("Stage 16 waiting transition re-admits request")
        selected_decode = set(scheduler["selected_decode_ids"])
        selected_prefill = set(scheduler["selected_prefill_ids"])
        preempted = {item["seq_id"] for item in scheduler["preemptions"]}
        prefill_origins = waiting_before_set | preempted
        if (
            not selected_decode.issubset(running_before)
            or not selected_prefill.issubset(prefill_origins)
            or not preempted.issubset(running_before)
            or running_after - selected_prefill
            != running_before - preempted
            or not waiting_after_set.issubset(prefill_origins)
            or not (prefill_origins - selected_prefill).issubset(
                waiting_after_set
            )
            or not selected_prefill.issubset(
                running_after | waiting_after_set
            )
        ):
            raise ValueError("Stage 16 scheduler state transition differs")
        before_waiting = scheduler["waiting_ids_before"]
        waiting_age = {
            seq_id: waiting_age.get(seq_id, 0)
            for seq_id in before_waiting
        }
        reconstructed_oldest = max(waiting_age.values(), default=0)
        if reconstructed_oldest != scheduler["oldest_waiting_age"]:
            raise ValueError("Stage 16 waiting-age reconstruction differs")
        oldest_waiting_ages.append(float(reconstructed_oldest))
        waiting_age_samples.extend(float(value) for value in waiting_age.values())

        if bounded:
            if step.get("schema_version") != 3:
                raise ValueError("Stage 16 bounded step must use schema v3")
            if scheduler["scheduler_step_index"] != expected_index:
                raise ValueError("Stage 16 scheduler step index differs")
            fairness = scheduler["active_progress_states"]
            expected_locations = {
                **{seq_id: "running" for seq_id in scheduler["running_ids_before"]},
                **{seq_id: "waiting" for seq_id in before_waiting},
            }
            if (
                len(fairness) != len(expected_locations)
                or {state["seq_id"] for state in fairness}
                != set(expected_locations)
            ):
                raise ValueError("Stage 16 active fairness membership differs")
            for state in fairness:
                seq_id = state["seq_id"]
                request_id = request_id_by_seq_id.get(seq_id)
                expected_last = (
                    last_progress.get(request_id)
                    if request_id is not None
                    else None
                )
                expected_since = (
                    expected_index - expected_last
                    if expected_last is not None
                    else None
                )
                if (
                    state["location"] != expected_locations[seq_id]
                    or state["request_id"] != request_id
                    or state["pending_recompute"]
                    != (seq_id in pending_recompute)
                    or state["had_emitted_token"]
                    != (
                        request_id in seen_token
                        if request_id is not None
                        else False
                    )
                    or state["last_progress_step"] != expected_last
                    or state["steps_since_last_progress"] != expected_since
                    or state["waiting_age_steps"]
                    != (
                        waiting_age.get(seq_id, 0)
                        if state["location"] == "waiting"
                        else 0
                    )
                ):
                    raise ValueError("Stage 16 fairness state differs")

        for event in step["events"]:
            request_id = event["request_id"]
            previous_progress = last_progress.get(request_id)
            gap = (
                expected_index - previous_progress
                if previous_progress is not None
                else None
            )
            had_token = request_id in seen_token
            if bounded and (
                event["scheduler_step_index"] != expected_index
                or event["previous_progress_step"] != previous_progress
                or event["progress_gap_steps"] != gap
                or event["had_emitted_token_before"] != had_token
            ):
                raise ValueError("Stage 16 progress event differs")
            if gap is not None:
                progress_gaps.append(float(gap))
                per_request_max_progress[request_id] = max(
                    per_request_max_progress.get(request_id, 0), gap
                )
                if had_token:
                    post_token_progress_gaps.append(float(gap))
                    per_request_max_post_token_progress[request_id] = max(
                        per_request_max_post_token_progress.get(request_id, 0),
                        gap,
                    )
            last_progress[request_id] = expected_index
            if event["emitted_token"]:
                previous_token = last_token.get(request_id)
                if previous_token is not None:
                    token_gap = expected_index - previous_token
                    token_gaps.append(float(token_gap))
                    per_request_max_token_gap[request_id] = max(
                        per_request_max_token_gap.get(request_id, 0),
                        token_gap,
                    )
                last_token[request_id] = expected_index
                seen_token.add(request_id)

        next_waiting_age: dict[int, int] = {}
        for seq_id in scheduler["waiting_ids_after"]:
            if seq_id in selected_prefill:
                next_waiting_age[seq_id] = 0
            elif seq_id in waiting_age:
                next_waiting_age[seq_id] = waiting_age[seq_id] + 1
            else:
                next_waiting_age[seq_id] = 0
        waiting_age = next_waiting_age

        if scheduler["mode"] == "drain_decode":
            if bounded and (
                scheduler["drain_episode_id"] is None
                or scheduler["drain_episode_started_at"] is None
                or scheduler["drain_episode_step"] <= 0
                or scheduler["drain_episode_step"] > MAX_DRAIN_STEPS
                or scheduler["oldest_waiting_age"] >= WAITING_AGE_LIMIT
                or scheduler["drain_exit_reason"] is not None
                or scheduler["fairness_trigger_reason"] is not None
                or scheduler["resource_pressure"] is not True
            ):
                raise ValueError("Stage 16 bounded drain invariant differs")
            if active_episode is None:
                if bounded and (
                    scheduler["drain_episode_step"] != 1
                    or scheduler["drain_tokens"]
                    != len(scheduler["selected_decode_ids"])
                    or scheduler["drain_episode_id"]
                    != len(episodes) + 1
                ):
                    raise ValueError("Stage 16 drain episode start differs")
                active_episode = {
                    "id": scheduler.get("drain_episode_id"),
                    "steps": 0,
                    "tokens": 0,
                    "scheduler_started_at": scheduler.get(
                        "drain_episode_started_at"
                    ),
                    "started_at": float(step["started_at"]),
                    "finished_at": float(step["finished_at"]),
                }
                if bounded:
                    fairness_by_seq = {
                        state["seq_id"]: state
                        for state in scheduler["active_progress_states"]
                    }
                    episode_started_at = scheduler[
                        "drain_episode_started_at"
                    ]
                    expected_watch = {
                        seq_id: (
                            state["last_progress_step"],
                            state["slo_deadline_at"],
                            state["slo_deadline_kind"],
                            state["request_id"],
                        )
                        for seq_id, state in fairness_by_seq.items()
                        if state["slo_deadline_at"] is not None
                        and state["slo_deadline_at"] > episode_started_at
                    }
                    actual_watch = {
                        watch["seq_id"]: (
                            watch["entry_progress_step"],
                            watch["deadline_at"],
                            watch["deadline_kind"],
                            watch["request_id"],
                        )
                        for watch in scheduler["drain_slo_watch"]
                    }
                    if (
                        len(actual_watch)
                        != len(scheduler["drain_slo_watch"])
                        or actual_watch != expected_watch
                    ):
                        raise ValueError("Stage 16 drain SLO watch differs")
                    active_episode["slo_watch"] = actual_watch
            elif bounded and (
                scheduler["drain_episode_id"] != active_episode["id"]
                or scheduler["drain_episode_started_at"]
                != active_episode["scheduler_started_at"]
                or scheduler["drain_episode_step"]
                != int(active_episode["steps"]) + 1
                or scheduler["drain_tokens"]
                != int(active_episode["tokens"])
                + len(scheduler["selected_decode_ids"])
                or {
                    watch["seq_id"]: (
                        watch["entry_progress_step"],
                        watch["deadline_at"],
                        watch["deadline_kind"],
                        watch["request_id"],
                    )
                    for watch in scheduler["drain_slo_watch"]
                }
                != active_episode["slo_watch"]
            ):
                raise ValueError("Stage 16 drain episode progression differs")
            active_episode["steps"] = int(active_episode["steps"]) + 1
            active_episode["tokens"] = int(active_episode["tokens"]) + len(
                scheduler["selected_decode_ids"]
            )
            active_episode["finished_at"] = float(step["finished_at"])
        elif active_episode is not None:
            if bounded and (
                scheduler["drain_exit_reason"] is None
                or scheduler["drain_episode_id"] != active_episode["id"]
                or scheduler["drain_episode_step"] != active_episode["steps"]
                or scheduler["drain_tokens"] != active_episode["tokens"]
                or scheduler["drain_episode_started_at"]
                != active_episode["scheduler_started_at"]
                or {
                    watch["seq_id"]: (
                        watch["entry_progress_step"],
                        watch["deadline_at"],
                        watch["deadline_kind"],
                        watch["request_id"],
                    )
                    for watch in scheduler["drain_slo_watch"]
                }
                != active_episode["slo_watch"]
                or scheduler["fairness_trigger_reason"] is not None
            ):
                raise ValueError("Stage 16 drain episode close differs")
            if bounded:
                reason = scheduler["drain_exit_reason"]
                allowed_exit_reasons = {
                    "max_drain_steps",
                    "waiting_age_limit",
                    "resource_released",
                    "ttft_slo_deadline",
                    "itl_slo_deadline",
                    "resident_empty",
                    "waiting_empty",
                }
                if reason not in allowed_exit_reasons:
                    raise ValueError("Stage 16 drain exit reason differs")
                validate_bounded_exit_active_set(
                    reason,
                    running_before,
                    waiting_before_set,
                )
                if reason == "max_drain_steps" and int(
                    active_episode["steps"]
                ) < MAX_DRAIN_STEPS:
                    raise ValueError("Stage 16 max-drain exit condition differs")
                if (
                    reason == "waiting_age_limit"
                    and scheduler["oldest_waiting_age"] < WAITING_AGE_LIMIT
                ):
                    raise ValueError("Stage 16 waiting-age exit condition differs")
                if (
                    reason == "resource_released"
                    and scheduler["resource_pressure"] is not False
                ):
                    raise ValueError("Stage 16 resource exit condition differs")
                if reason in {"ttft_slo_deadline", "itl_slo_deadline"}:
                    seq_id = scheduler["slo_guard_seq_id"]
                    watch = active_episode["slo_watch"].get(seq_id)
                    states = {
                        state["seq_id"]: state
                        for state in scheduler["active_progress_states"]
                    }
                    state = states.get(seq_id)
                    if (
                        watch is None
                        or state is None
                        or scheduler["slo_guard_entry_progress_step"]
                        != watch[0]
                        or scheduler["slo_guard_deadline_at"] is None
                        or scheduler["slo_guard_deadline_at"] != watch[1]
                        or scheduler["slo_guard_triggered_at"] is None
                        or scheduler["drain_episode_started_at"] is None
                        or scheduler["slo_guard_deadline_at"]
                        <= scheduler["drain_episode_started_at"]
                        or scheduler["slo_guard_triggered_at"]
                        < scheduler["slo_guard_deadline_at"]
                        or reason != watch[2]
                        or state["request_id"] != watch[3]
                        or state["last_progress_step"] != watch[0]
                        or state["slo_deadline_at"] != watch[1]
                        or state["slo_deadline_kind"] != watch[2]
                        or state["slo_deadline_kind"]
                        != (
                            "itl_slo_deadline"
                            if state["had_emitted_token"]
                            else "ttft_slo_deadline"
                        )
                    ):
                        raise ValueError("Stage 16 SLO guard evidence differs")
            episodes.append(
                {
                    "steps": int(active_episode["steps"]),
                    "duration_ms": (
                        float(active_episode["finished_at"])
                        - float(active_episode["started_at"])
                    )
                    * 1000.0,
                }
            )
            active_episode = None
        if bounded and scheduler["drain_exit_reason"] is not None:
            exit_reasons[scheduler["drain_exit_reason"]] += 1
        if bounded and scheduler["fairness_trigger_reason"] is not None:
            if (
                scheduler["drain_episode_id"] is not None
                or scheduler["drain_exit_reason"] is not None
                or scheduler["fairness_trigger_reason"]
                not in {"initial_waiting_age", "waiting_age_limit"}
            ):
                raise ValueError("Stage 16 fairness trigger evidence differs")
            if (
                scheduler["fairness_trigger_reason"] == "waiting_age_limit"
                and scheduler["oldest_waiting_age"] < WAITING_AGE_LIMIT
            ):
                raise ValueError("Stage 16 fairness trigger condition differs")
        if (
            bounded
            and active_episode is None
            and scheduler["drain_episode_id"] is None
            and scheduler["drain_slo_watch"]
        ):
            raise ValueError("Stage 16 inactive drain SLO watch differs")

        pending_recompute.update(preempted)
        for event in step["events"]:
            if event.get("resumed"):
                pending_recompute.discard(event["seq_id"])
        previous_running_after = running_after
        previous_waiting_after = waiting_after_set
        previous_finished = {
            event["seq_id"] for event in step["events"] if event["finished"]
        }
        ever_active.update(running_before | waiting_before_set)
        ever_active.update(running_after | waiting_after_set)
    if active_episode is not None:
        if bounded:
            raise ValueError("Stage 16 drain episode did not close")
        episodes.append(
            {
                "steps": int(active_episode["steps"]),
                "duration_ms": (
                    float(active_episode["finished_at"])
                    - float(active_episode["started_at"])
                )
                * 1000.0,
            }
        )
    if bounded and sum(exit_reasons.values()) != len(episodes):
        raise ValueError("Stage 16 drain episode count differs")

    return {
        "oldest_waiting_age": distribution(oldest_waiting_ages),
        "request_waiting_age": distribution(waiting_age_samples),
        "request_progress_gap_steps": distribution(progress_gaps),
        "post_token_progress_gap_steps": distribution(
            post_token_progress_gaps
        ),
        "token_gap_steps": distribution(token_gaps),
        "per_request_max_progress_gap_steps": distribution(
            [float(value) for value in per_request_max_progress.values()]
        ),
        "per_request_max_post_token_progress_gap_steps": distribution(
            [
                float(value)
                for value in per_request_max_post_token_progress.values()
            ]
        ),
        "per_request_max_token_gap_steps": distribution(
            [float(value) for value in per_request_max_token_gap.values()]
        ),
        "drain_episode_length_steps": distribution(
            [float(episode["steps"]) for episode in episodes]
        ),
        "drain_episode_duration_ms": distribution(
            [float(episode["duration_ms"]) for episode in episodes]
        ),
        "drain_episode_count": len(episodes),
        "drain_exit_reasons": dict(sorted(exit_reasons.items())),
    }


def run_metrics(record: dict[str, Any], *, bounded: bool) -> dict[str, Any]:
    serving = record["serving_metrics"]
    recompute = record["recompute_metrics"]
    summary = record["summary"]
    fairness = fairness_metrics(record["steps"], bounded=bounded)
    return {
        "actual_recompute_tokens": recompute["actual_recompute_tokens"],
        "recompute_amplification": recompute["recompute_amplification"],
        "preemption_count": recompute["preemption_count"],
        "resume_count": recompute["resume_count"],
        "ttft_p50_ms": serving["interactive_ttft"]["p50"],
        "ttft_p99_ms": serving["interactive_ttft"]["p99"],
        "itl_p50_ms": serving["interactive_itl"]["p50"],
        "itl_p99_ms": serving["interactive_itl"]["p99"],
        "interactive_slo_goodput_rps": summary["summary"]["interactive"][
            "slo_goodput_rps"
        ],
        "elapsed_seconds": record["elapsed_seconds"],
        "steps": summary["summary"]["runtime"]["steps"],
        "fairness": fairness,
    }


def relative_reduction(candidate: float, baseline: float) -> float | None:
    if baseline == 0:
        return None
    return (baseline - candidate) / baseline


def aggregate_stage16(
    results_directory: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    records: dict[tuple[int, int, str], dict[str, Any]] = {}
    code_fingerprints = set()
    repository_heads = set()
    runtime_fingerprints = set()
    model_fingerprints = set()
    manifest_fingerprint = manifest["execution_fingerprint"]
    if tuple(CODE_FINGERPRINT_KEYS) != tuple(
        EXECUTION_FINGERPRINT_KEYS
    ):
        raise ValueError("Stage 16 execution fingerprint schema differs")
    for cell in manifest["cells"]:
        capacity = cell["capacity"]
        seed = cell["seed"]
        policy_name = cell["policy"]
        run_id = cell["run_id"]
        config_path = (REPO_ROOT / cell["config_path"]).resolve()
        source_path = (REPO_ROOT / cell["source_config_path"]).resolve()
        if sha256_file(config_path) != cell["config_sha256"]:
            raise ValueError("Stage 16 frozen config hash differs")
        if sha256_file(source_path) != cell["source_config_sha256"]:
            raise ValueError("Stage 16 source config hash differs")
        trace_path = (REPO_ROOT / cell["trace_path"]).resolve()
        if sha256_file(trace_path) != cell["trace_sha256"]:
            raise ValueError("Stage 16 frozen trace hash differs")
        record = load_validated_stage15_run(
            (results_directory / f"{run_id}.summary.json").resolve(),
            expected_policy=POLICIES[policy_name],
            expected_capacity=capacity,
        )
        if record["paths"]["config"] != config_path:
            raise ValueError("Stage 16 config path differs from manifest")
        source = json.loads(source_path.read_text(encoding="utf-8"))
        expected = expected_config_for_cell(
            source,
            policy_name=policy_name,
            run_id=run_id,
        )
        if record["config"] != expected:
            raise ValueError("Stage 16 config differs from frozen source")
        if policy_name == "bounded":
            bounded_summary = record["summary"].get(
                "recompute_aware_bounded"
            )
            if not isinstance(bounded_summary, dict):
                raise ValueError("Stage 16 bounded summary is missing")
            for field in ("actual_recompute_tokens", "resume_count"):
                if bounded_summary.get(field) != record[
                    "recompute_metrics"
                ][field]:
                    raise ValueError(
                        f"Stage 16 bounded summary {field} differs"
                    )
        provenance = record["summary"]["provenance"]
        if record["paths"]["trace"] != trace_path:
            raise ValueError("Stage 16 trace path differs from manifest")
        if provenance["trace_sha256"] != cell["trace_sha256"]:
            raise ValueError("Stage 16 summary trace hash differs")
        if any(
            provenance.get(key) != manifest_fingerprint[key]
            for key in EXECUTION_FINGERPRINT_KEYS
        ):
            raise ValueError("Stage 16 run execution fingerprint differs")
        code_fingerprints.add(
            tuple(provenance[key] for key in CODE_FINGERPRINT_KEYS)
        )
        repository_heads.add(record["summary"]["repository_head"])
        runtime_fingerprints.add(
            json.dumps(record["summary"]["runtime"], sort_keys=True)
        )
        model_fingerprints.add(
            json.dumps(record["summary"]["model"], sort_keys=True)
        )
        metrics = run_metrics(
            record,
            bounded=policy_name == "bounded",
        )
        if policy_name == "bounded" and (
            bounded_summary.get("drain_active") is not False
            or bounded_summary.get("drain_episode_count")
            != metrics["fairness"]["drain_episode_count"]
        ):
            raise ValueError("Stage 16 bounded episode summary differs")
        records[(capacity, seed, policy_name)] = {
            **record,
            "metrics": metrics,
        }
    if (
        len(records) != 27
        or len(code_fingerprints) != 1
        or len(repository_heads) != 1
        or len(runtime_fingerprints) != 1
        or len(model_fingerprints) != 1
    ):
        raise ValueError("Stage 16 matrix code/cell consistency differs")

    triples = []
    for capacity in CAPACITIES:
        for seed in SEEDS:
            triple = {
                policy: records[(capacity, seed, policy)]
                for policy in POLICIES
            }
            if len(
                {
                    json.dumps(
                        normalized_config(item["config"]),
                        sort_keys=True,
                    )
                    for item in triple.values()
                }
            ) != 1 or len(
                {
                    item["summary"]["provenance"]["trace_sha256"]
                    for item in triple.values()
                }
            ) != 1:
                raise ValueError("Stage 16 matched triple contract differs")
            triples.append(
                {
                    "capacity": capacity,
                    "seed": seed,
                    **{
                        policy: triple[policy]["metrics"]
                        for policy in POLICIES
                    },
                }
            )

    def aggregate_gate(
        metric_getter,
        *,
        baseline_policy: str,
        candidate_policy: str,
        reduction_min: float,
        regression_max: float | None = None,
    ) -> dict[str, Any]:
        baseline_values = [
            float(metric_getter(item[baseline_policy])) for item in triples
        ]
        candidate_values = [
            float(metric_getter(item[candidate_policy])) for item in triples
        ]
        baseline_total = sum(baseline_values)
        candidate_total = sum(candidate_values)
        reduction = relative_reduction(candidate_total, baseline_total)
        improved = sum(
            candidate < baseline
            for baseline, candidate in zip(
                baseline_values,
                candidate_values,
                strict=True,
            )
        )
        passed = (
            reduction is not None
            and (
                reduction >= reduction_min
                if regression_max is None
                else reduction >= -regression_max
            )
            and (
                improved >= ACCEPTANCE["improved_matched_pairs_min"]
                if regression_max is None
                else True
            )
        )
        return {
            "baseline_policy": baseline_policy,
            "candidate_policy": candidate_policy,
            "baseline_total": baseline_total,
            "candidate_total": candidate_total,
            "reduction": reduction,
            "improved_pairs": improved,
            "passed": passed,
        }

    gates = {
        "post_token_progress_gap_p99": aggregate_gate(
            lambda metrics: metrics["fairness"][
                "post_token_progress_gap_steps"
            ]["p99"],
            baseline_policy="recompute",
            candidate_policy="bounded",
            reduction_min=ACCEPTANCE["fairness_reduction_min"],
        ),
        "max_waiting_age": aggregate_gate(
            lambda metrics: metrics["fairness"]["oldest_waiting_age"][
                "max"
            ],
            baseline_policy="recompute",
            candidate_policy="bounded",
            reduction_min=ACCEPTANCE["fairness_reduction_min"],
        ),
        "itl_p99_ms": aggregate_gate(
            lambda metrics: metrics["itl_p99_ms"],
            baseline_policy="recompute",
            candidate_policy="bounded",
            reduction_min=ACCEPTANCE["itl_p99_reduction_min"],
        ),
        "actual_recompute_tokens": aggregate_gate(
            lambda metrics: metrics["actual_recompute_tokens"],
            baseline_policy="pressure",
            candidate_policy="bounded",
            reduction_min=ACCEPTANCE["resource_reduction_min"],
        ),
        "preemption_count": aggregate_gate(
            lambda metrics: metrics["preemption_count"],
            baseline_policy="pressure",
            candidate_policy="bounded",
            reduction_min=ACCEPTANCE["resource_reduction_min"],
        ),
        "ttft_p99_ms": aggregate_gate(
            lambda metrics: metrics["ttft_p99_ms"],
            baseline_policy="pressure",
            candidate_policy="bounded",
            reduction_min=0.0,
            regression_max=ACCEPTANCE["ttft_p99_regression_max"],
        ),
    }
    accepted = all(gate["passed"] for gate in gates.values())
    return {
        "schema_version": 1,
        "revision": manifest["revision"],
        "matrix": {"runs": 27, "matched_triples": 9},
        "completion_gates_validated": True,
        "raw_metrics_recomputed": True,
        "execution_code_fingerprint": dict(
            zip(CODE_FINGERPRINT_KEYS, next(iter(code_fingerprints)))
        ),
        "triples": triples,
        "aggregate_gates": gates,
        "accepted": accepted,
        "decision": (
            "pareto_success_enter_stage16b"
            if accepted
            else "stage16a_failed_stop_before_backpressure"
        ),
    }


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as file:
        temporary_path = Path(file.name)
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    temporary_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / f"experiments/configs/{REVISION}_matrix.json",
    )
    parser.add_argument(
        "--results-directory",
        type=Path,
        default=REPO_ROOT / "experiments/results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / f"experiments/results/{REVISION}.aggregate.json",
    )
    args = parser.parse_args()
    aggregate = aggregate_stage16(
        args.results_directory.resolve(),
        args.manifest.resolve(),
    )
    write_json_atomic(args.output.resolve(), aggregate)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
