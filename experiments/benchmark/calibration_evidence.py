from __future__ import annotations

from collections import Counter
import hashlib
import json
from math import isfinite
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from experiments.benchmark.lifecycle import percentile


RUNTIME_KEYS = (
    "python",
    "gpu",
    "nvidia_driver",
    "torch",
    "torch_cuda",
    "transformers",
    "triton",
    "flash_attn",
    "requirements_lock_sha256",
    "pylock_sha256",
)
CODE_KEYS = (
    "runner_sha256",
    "open_loop_sha256",
    "lifecycle_sha256",
    "llm_engine_sha256",
)
REQUEST_KEYS = {
    "request_id",
    "request_class",
    "arrival_at",
    "admitted_at",
    "first_scheduled_at",
    "token_timestamps",
    "terminal_state",
    "terminal_at",
    "terminal_reason",
    "ttft_ms",
    "e2e_ms",
    "itl_ms",
    "max_progress_gap_ms",
}
STEP_KEYS = {
    "step_index",
    "started_at",
    "finished_at",
    "duration_ms",
    "phase",
    "num_scheduled_tokens",
    "events",
}
STEP_EVENT_KEYS = {
    "request_id",
    "seq_id",
    "phase",
    "num_scheduled_tokens",
    "emitted_token",
    "finished",
}
STEP_V2_KEYS = STEP_KEYS | {"schema_version", "scheduler"}
STEP_V3_KEYS = STEP_V2_KEYS
PREFILL_STEP_EVENT_V2_KEYS = STEP_EVENT_KEYS | {
    "prefill_kind",
    "actual_recompute_tokens",
    "resumed",
}
STEP_EVENT_V3_PROGRESS_KEYS = {
    "scheduler_step_index",
    "previous_progress_step",
    "progress_gap_steps",
    "had_emitted_token_before",
}
STEP_EVENT_V3_KEYS = STEP_EVENT_KEYS | STEP_EVENT_V3_PROGRESS_KEYS
PREFILL_STEP_EVENT_V3_KEYS = (
    PREFILL_STEP_EVENT_V2_KEYS | STEP_EVENT_V3_PROGRESS_KEYS
)


def recorded_path_is_absolute(value: str) -> bool:
    return (
        PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
    )


def resolve_recorded_artifact_path(
    value: Any,
    *,
    repository_root: Path,
    allowed_directory: Path,
    label: str,
) -> Path:
    """Relocate a repository artifact recorded from another checkout."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path is invalid")
    if not recorded_path_is_absolute(value):
        raise ValueError(f"{label} path must be absolute")
    normalized = value.replace("\\", "/")
    parts = normalized.split("/")
    anchors = [index for index, part in enumerate(parts) if part == "experiments"]
    if len(anchors) != 1:
        raise ValueError(f"{label} path is not repository-relative")
    relative_parts = parts[anchors[0] :]
    if any(part in {"", ".", ".."} for part in relative_parts):
        raise ValueError(f"{label} path contains unsafe components")
    path = repository_root.joinpath(*relative_parts).resolve()
    root = repository_root.resolve()
    allowed = allowed_directory.resolve()
    if (
        not path.is_relative_to(root)
        or not path.is_relative_to(allowed)
        or not path.is_file()
    ):
        raise ValueError(f"{label} artifact is missing")
    return path


def recorded_path_matches(
    value: Any,
    expected: Path,
    *,
    repository_root: Path,
    allowed_directory: Path,
    label: str,
) -> bool:
    try:
        return resolve_recorded_artifact_path(
            value,
            repository_root=repository_root,
            allowed_directory=allowed_directory,
            label=label,
        ) == expected.resolve()
    except ValueError:
        return False
SCHEDULER_STEP_KEYS = {
    "policy",
    "state",
    "mode",
    "kv_total_blocks",
    "kv_used_blocks_before",
    "kv_free_blocks_before",
    "kv_used_blocks_after",
    "kv_free_blocks_after",
    "running_ids_before",
    "waiting_ids_before",
    "running_ids_after",
    "waiting_ids_after",
    "selected_decode_ids",
    "selected_prefill_ids",
    "oldest_waiting_age",
    "resident_costs",
    "preemptions",
    "selected_decode_request_ids",
    "selected_prefill_request_ids",
}
SCHEDULER_STEP_V3_KEYS = SCHEDULER_STEP_KEYS | {
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
}
FAIRNESS_STATE_KEYS = {
    "seq_id",
    "location",
    "pending_recompute",
    "had_emitted_token",
    "waiting_age_steps",
    "last_progress_step",
    "steps_since_last_progress",
    "slo_deadline_at",
    "slo_deadline_kind",
    "request_id",
}
DRAIN_SLO_WATCH_KEYS = {
    "seq_id",
    "entry_progress_step",
    "deadline_at",
    "deadline_kind",
    "request_id",
}
RESIDENT_COST_KEYS = {
    "seq_id",
    "resident_kv_blocks",
    "resident_kv_tokens",
    "logical_context_tokens",
    "remaining_decode_tokens",
    "releasable_blocks",
    "estimated_recompute_tokens",
    "estimated_steps_to_release",
    "can_advance_without_free_block",
    "request_id",
}
PREEMPTION_EVENT_KEYS = {
    "seq_id",
    "reason",
    "triggering_seq_id",
    "logical_context_tokens",
    "resident_kv_tokens",
    "resident_blocks",
    "releasable_blocks",
    "estimated_recompute_tokens",
    "remaining_decode_tokens",
    "scheduler_state",
    "request_id",
    "triggering_request_id",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def require_exact_keys(
    value: Any,
    expected: set[str],
    label: str,
) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} schema differs")


def require_finite_number(value: Any, label: str) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite number")


def validate_request_schema(request: Any) -> None:
    require_exact_keys(request, REQUEST_KEYS, "request")
    if (
        not isinstance(request["request_id"], str)
        or request["request_class"] not in {"interactive", "long"}
        or request["terminal_state"] != "Finished"
        or request["terminal_reason"] is not None
    ):
        raise ValueError("request identity or terminal fields differ")
    for field in (
        "arrival_at",
        "admitted_at",
        "first_scheduled_at",
        "terminal_at",
        "ttft_ms",
        "e2e_ms",
        "max_progress_gap_ms",
    ):
        require_finite_number(request[field], f"request.{field}")
    for field in ("token_timestamps", "itl_ms"):
        if not isinstance(request[field], list):
            raise ValueError(f"request.{field} must be a list")
        for value in request[field]:
            require_finite_number(value, f"request.{field}")


def validate_step_schema(step: Any) -> None:
    schema_version = step.get("schema_version", 1) if isinstance(step, dict) else 1
    if schema_version == 1:
        require_exact_keys(step, STEP_KEYS, "step")
    elif schema_version == 2:
        require_exact_keys(step, STEP_V2_KEYS, "step")
        validate_scheduler_step_schema(step["scheduler"])
    elif schema_version == 3:
        require_exact_keys(step, STEP_V3_KEYS, "step")
        validate_scheduler_step_schema(
            step["scheduler"],
            schema_version=3,
        )
    else:
        raise ValueError("Unsupported step schema version")
    if (
        not isinstance(step["step_index"], int)
        or isinstance(step["step_index"], bool)
        or step["phase"] not in {"prefill", "decode"}
        or not isinstance(step["num_scheduled_tokens"], int)
        or isinstance(step["num_scheduled_tokens"], bool)
        or step["num_scheduled_tokens"] < 0
        or not isinstance(step["events"], list)
    ):
        raise ValueError("step field types differ")
    for field in ("started_at", "finished_at", "duration_ms"):
        require_finite_number(step[field], f"step.{field}")
    for event in step["events"]:
        validate_step_event_schema(event, schema_version=schema_version)


def validate_step_event_schema(
    event: Any,
    *,
    schema_version: int = 1,
) -> None:
    expected = STEP_EVENT_KEYS
    if schema_version == 2 and isinstance(event, dict) and event.get("phase") == "prefill":
        expected = PREFILL_STEP_EVENT_V2_KEYS
    elif schema_version == 3:
        expected = (
            PREFILL_STEP_EVENT_V3_KEYS
            if isinstance(event, dict) and event.get("phase") == "prefill"
            else STEP_EVENT_V3_KEYS
        )
    require_exact_keys(event, expected, "step event")
    if (
        type(event["emitted_token"]) is not bool
        or type(event["finished"]) is not bool
    ):
        raise ValueError("Step event flags must be booleans")
    if (
        not isinstance(event["request_id"], str)
        or not isinstance(event["seq_id"], int)
        or isinstance(event["seq_id"], bool)
        or event["phase"] not in {"prefill", "decode"}
        or not isinstance(event["num_scheduled_tokens"], int)
        or isinstance(event["num_scheduled_tokens"], bool)
        or event["num_scheduled_tokens"] <= 0
    ):
        raise ValueError("step event field types differ")
    if schema_version in {2, 3} and event["phase"] == "prefill":
        if (
            event["prefill_kind"]
            not in {"initial_prefill", "recompute_prefill"}
            or not isinstance(event["actual_recompute_tokens"], int)
            or isinstance(event["actual_recompute_tokens"], bool)
            or event["actual_recompute_tokens"] < 0
            or type(event["resumed"]) is not bool
        ):
            raise ValueError("Stage 15 Prefill accounting fields differ")
    if schema_version == 3:
        optional_integer_fields = (
            "previous_progress_step",
            "progress_gap_steps",
        )
        if (
            not isinstance(event["scheduler_step_index"], int)
            or isinstance(event["scheduler_step_index"], bool)
            or event["scheduler_step_index"] < 0
            or type(event["had_emitted_token_before"]) is not bool
            or any(
                value is not None
                and (
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value < 0
                )
                for value in (
                    event[field] for field in optional_integer_fields
                )
            )
        ):
            raise ValueError("Stage 16 progress fields differ")


def validate_scheduler_step_schema(
    scheduler: Any,
    *,
    schema_version: int = 2,
) -> None:
    require_exact_keys(
        scheduler,
        (
            SCHEDULER_STEP_V3_KEYS
            if schema_version == 3
            else SCHEDULER_STEP_KEYS
        ),
        "scheduler step",
    )
    integer_fields = (
        "kv_total_blocks",
        "kv_used_blocks_before",
        "kv_free_blocks_before",
        "kv_used_blocks_after",
        "kv_free_blocks_after",
        "oldest_waiting_age",
    )
    if (
        not isinstance(scheduler["policy"], str)
        or not isinstance(scheduler["state"], str)
        or not isinstance(scheduler["mode"], str)
        or any(
            not isinstance(scheduler[field], int)
            or isinstance(scheduler[field], bool)
            or scheduler[field] < 0
            for field in integer_fields
        )
    ):
        raise ValueError("scheduler step scalar fields differ")
    for field in (
        "running_ids_before",
        "waiting_ids_before",
        "running_ids_after",
        "waiting_ids_after",
        "selected_decode_ids",
        "selected_prefill_ids",
        "resident_costs",
        "preemptions",
        "selected_decode_request_ids",
        "selected_prefill_request_ids",
    ):
        if not isinstance(scheduler[field], list):
            raise ValueError(f"scheduler step {field} must be a list")
    for cost in scheduler["resident_costs"]:
        require_exact_keys(cost, RESIDENT_COST_KEYS, "resident cost")
    for preemption in scheduler["preemptions"]:
        require_exact_keys(
            preemption,
            PREEMPTION_EVENT_KEYS,
            "preemption event",
        )
    if schema_version == 3:
        for field in (
            "scheduler_step_index",
            "drain_episode_step",
            "drain_tokens",
        ):
            if (
                not isinstance(scheduler[field], int)
                or isinstance(scheduler[field], bool)
                or scheduler[field] < 0
            ):
                raise ValueError(f"scheduler step {field} differs")
        for field in (
            "drain_episode_id",
            "slo_guard_seq_id",
            "slo_guard_entry_progress_step",
        ):
            value = scheduler[field]
            if value is not None and (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(f"scheduler step {field} differs")
        if scheduler["drain_exit_reason"] is not None and not isinstance(
            scheduler["drain_exit_reason"], str
        ):
            raise ValueError("scheduler drain exit reason differs")
        if (
            scheduler["fairness_trigger_reason"] is not None
            and not isinstance(scheduler["fairness_trigger_reason"], str)
        ):
            raise ValueError("scheduler fairness trigger reason differs")
        if scheduler["resource_pressure"] is not None and type(
            scheduler["resource_pressure"]
        ) is not bool:
            raise ValueError("scheduler resource pressure differs")
        for field in ("slo_guard_deadline_at", "slo_guard_triggered_at"):
            value = scheduler[field]
            if value is not None:
                require_finite_number(value, f"scheduler {field}")
        if scheduler["drain_episode_started_at"] is not None:
            require_finite_number(
                scheduler["drain_episode_started_at"],
                "scheduler drain_episode_started_at",
            )
        if not isinstance(scheduler["active_progress_states"], list):
            raise ValueError("scheduler active progress states must be a list")
        for state in scheduler["active_progress_states"]:
            require_exact_keys(state, FAIRNESS_STATE_KEYS, "fairness state")
            if (
                not isinstance(state["seq_id"], int)
                or isinstance(state["seq_id"], bool)
                or state["location"] not in {"running", "waiting"}
                or type(state["pending_recompute"]) is not bool
                or type(state["had_emitted_token"]) is not bool
                or not isinstance(state["waiting_age_steps"], int)
                or isinstance(state["waiting_age_steps"], bool)
                or state["waiting_age_steps"] < 0
                or not isinstance(state["request_id"], str)
            ):
                raise ValueError("fairness state fields differ")
            for field in ("last_progress_step", "steps_since_last_progress"):
                value = state[field]
                if value is not None and (
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value < 0
                ):
                    raise ValueError(f"fairness state {field} differs")
            deadline = state["slo_deadline_at"]
            kind = state["slo_deadline_kind"]
            if deadline is not None:
                require_finite_number(deadline, "fairness state deadline")
            if (deadline is None) != (kind is None) or kind not in {
                None,
                "ttft_slo_deadline",
                "itl_slo_deadline",
            }:
                raise ValueError("fairness state SLO deadline differs")
        if not isinstance(scheduler["drain_slo_watch"], list):
            raise ValueError("scheduler drain SLO watch must be a list")
        for watch in scheduler["drain_slo_watch"]:
            require_exact_keys(
                watch,
                DRAIN_SLO_WATCH_KEYS,
                "drain SLO watch",
            )
            if (
                not isinstance(watch["seq_id"], int)
                or isinstance(watch["seq_id"], bool)
                or not isinstance(watch["request_id"], str)
                or watch["deadline_kind"]
                not in {"ttft_slo_deadline", "itl_slo_deadline"}
            ):
                raise ValueError("drain SLO watch fields differ")
            require_finite_number(
                watch["deadline_at"],
                "drain SLO deadline",
            )
            entry = watch["entry_progress_step"]
            if entry is not None and (
                not isinstance(entry, int)
                or isinstance(entry, bool)
                or entry < 0
            ):
                raise ValueError("drain SLO watch progress differs")


def canonical_structure_digest(value: Any) -> str:
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def structural_raw_fingerprint(
    requests: list[dict[str, Any]],
    steps: list[dict[str, Any]],
) -> tuple[str, str]:
    request_view = [
        {
            "request_id": request["request_id"],
            "request_class": request["request_class"],
            "output_token_count": len(request["token_timestamps"]),
            "terminal_state": request["terminal_state"],
        }
        for request in requests
    ]
    step_view = [
        {
            "step_index": step["step_index"],
            "events": [
                {
                    key: event[key]
                    for key in ("request_id", "emitted_token", "finished")
                }
                for event in step["events"]
            ],
        }
        for step in steps
    ]
    return (
        canonical_structure_digest(request_view),
        canonical_structure_digest(step_view),
    )


def distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def require_close(actual: Any, expected: Any, label: str) -> None:
    if actual is None or expected is None:
        if actual is not expected:
            raise ValueError(f"{label} differs: {actual!r} != {expected!r}")
        return
    if isinstance(expected, bool) or isinstance(expected, int):
        if actual != expected:
            raise ValueError(f"{label} differs: {actual!r} != {expected!r}")
        return
    if (
        not isinstance(actual, (int, float))
        or isinstance(actual, bool)
        or not isfinite(float(actual))
        or abs(float(actual) - float(expected)) > 1e-7
    ):
        raise ValueError(f"{label} differs: {actual!r} != {expected!r}")


def require_distribution(
    actual: dict[str, Any],
    expected: dict[str, Any],
    label: str,
) -> None:
    if set(actual) != set(expected):
        raise ValueError(f"{label} distribution schema differs")
    for key, value in expected.items():
        require_close(actual[key], value, f"{label}.{key}")


def require_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )


def expected_paths(
    *,
    stem: str,
    trace_stem: str | None = None,
    contract: dict[str, Any],
    repository_root: Path,
    results_directory: Path,
) -> dict[str, Path]:
    resolved_trace_stem = trace_stem or stem
    return {
        "config": (
            repository_root
            / contract["config_output_directory"]
            / f"{stem}.json"
        ).resolve(),
        "trace": (
            repository_root
            / contract["trace_output_directory"]
            / f"{resolved_trace_stem}.json"
        ).resolve(),
        "requests": (results_directory / f"{stem}.requests.jsonl").resolve(),
        "steps": (results_directory / f"{stem}.steps.jsonl").resolve(),
    }


def load_bound_artifacts(
    *,
    summary_path: Path,
    stem: str,
    trace_stem: str | None = None,
    contract: dict[str, Any],
    repository_root: Path,
    results_directory: Path,
) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["status"] != "passed":
        raise ValueError(f"Run did not pass: {summary_path}")
    paths = {
        "config": resolve_recorded_artifact_path(
            summary["config_path"],
            repository_root=repository_root,
            allowed_directory=repository_root / "experiments/configs",
            label="config",
        ),
        "trace": resolve_recorded_artifact_path(
            summary["provenance"]["trace_path"],
            repository_root=repository_root,
            allowed_directory=repository_root / "experiments/data",
            label="trace",
        ),
        "requests": resolve_recorded_artifact_path(
            summary["artifacts"]["requests_path"],
            repository_root=repository_root,
            allowed_directory=results_directory,
            label="requests",
        ),
        "steps": resolve_recorded_artifact_path(
            summary["artifacts"]["steps_path"],
            repository_root=repository_root,
            allowed_directory=results_directory,
            label="steps",
        ),
    }
    expected = expected_paths(
        stem=stem,
        trace_stem=trace_stem,
        contract=contract,
        repository_root=repository_root,
        results_directory=results_directory,
    )
    for label, path in paths.items():
        if path != expected[label]:
            raise ValueError(f"{label} path is not bound to {stem}: {path}")

    expected_hashes = {
        "config": summary["provenance"]["config_sha256"],
        "trace": summary["provenance"]["trace_sha256"],
        "requests": summary["artifacts"]["requests_sha256"],
        "steps": summary["artifacts"]["steps_sha256"],
    }
    for label, expected_hash in expected_hashes.items():
        require_hash(paths[label], expected_hash, label)
    return {
        "summary": summary,
        "config": json.loads(paths["config"].read_text(encoding="utf-8")),
        "trace": json.loads(paths["trace"].read_text(encoding="utf-8")),
        "requests": read_jsonl(paths["requests"]),
        "steps": read_jsonl(paths["steps"]),
        "paths": paths,
    }


def validate_config_binding(
    *,
    bound: dict[str, Any],
    stem: str,
    seed: int,
    contract: dict[str, Any],
    repository_root: Path,
    results_directory: Path,
    summary_path: Path,
) -> None:
    summary = bound["summary"]
    config = bound["config"]
    paths = bound["paths"]
    if config["output"]["run_id"] != stem:
        raise ValueError(f"Config run_id mismatch: {paths['config']}")
    if config["sampling"]["seed"] != contract["inference_seed_base"] + seed:
        raise ValueError(f"Config sampling seed mismatch: {paths['config']}")
    output_directory = (
        repository_root / config["output"]["directory"]
    ).resolve()
    if output_directory != results_directory:
        raise ValueError(f"Config output directory mismatch: {paths['config']}")
    trace_path = (
        repository_root / config["workload"]["trace_path"]
    ).resolve()
    if trace_path != paths["trace"]:
        raise ValueError(f"Config trace path mismatch: {paths['config']}")
    for field in ("engine", "sampling", "slo", "measurement"):
        if config[field] != summary[field]:
            raise ValueError(f"Config and summary {field} differ: {summary_path}")
    if config["upstream_commit"] != summary["upstream_commit"]:
        raise ValueError(f"Config upstream mismatch: {summary_path}")
    for field in ("repo_id", "revision", "sha256"):
        if config["model"][field] != summary["model"][field]:
            raise ValueError(f"Config and summary model {field} differ")
    if summary["model"]["verified_sha256"] != summary["model"]["sha256"]:
        raise ValueError(f"Model hashes were not verified: {summary_path}")


def validate_binding(
    *,
    summary_path: Path,
    stem: str,
    trace_stem: str | None = None,
    seed: int,
    contract: dict[str, Any],
    repository_root: Path,
    results_directory: Path,
) -> dict[str, Any]:
    bound = load_bound_artifacts(
        summary_path=summary_path,
        stem=stem,
        trace_stem=trace_stem,
        contract=contract,
        repository_root=repository_root,
        results_directory=results_directory,
    )
    validate_config_binding(
        bound=bound,
        stem=stem,
        seed=seed,
        contract=contract,
        repository_root=repository_root,
        results_directory=results_directory,
        summary_path=summary_path,
    )
    return bound


def validate_terminal_counts(
    summary: dict[str, Any],
    requests: list[dict[str, Any]],
) -> None:
    terminal = summary["summary"]["terminal_counts"]
    states = ("Finished", "Rejected", "Failed", "Cancelled", "Unfinished")
    if (
        not terminal["reconciled"]
        or sum(terminal[state] for state in states) != terminal["submitted"]
        or terminal["Finished"] != terminal["submitted"]
    ):
        raise ValueError("Calibration terminal reconciliation failed")
    raw_counts = Counter(request["terminal_state"] for request in requests)
    for state in states:
        if raw_counts[state] != terminal[state]:
            raise ValueError("Raw and summary terminal counts differ")


def index_validated_requests(
    trace: dict[str, Any],
    requests: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    for request in requests:
        validate_request_schema(request)
    trace_by_id = {
        request["request_id"]: request for request in trace["requests"]
    }
    raw_by_id = {request["request_id"]: request for request in requests}
    if (
        len(trace_by_id) != len(trace["requests"])
        or len(raw_by_id) != len(requests)
        or set(trace_by_id) != set(raw_by_id)
    ):
        raise ValueError("Trace and raw request IDs differ")
    return trace_by_id, raw_by_id


def validate_requests(
    trace: dict[str, Any],
    requests: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    trace_by_id, raw_by_id = index_validated_requests(trace, requests)
    for request_id, raw in raw_by_id.items():
        planned = trace_by_id[request_id]
        planned_arrival = float(planned["arrival_time"])
        arrival = float(raw["arrival_at"])
        terminal = float(raw["terminal_at"])
        timestamps = [float(value) for value in raw["token_timestamps"]]
        output_tokens = planned["max_output_tokens"]
        if raw["request_class"] != planned["request_class"]:
            raise ValueError(f"Request class mismatch for {request_id}")
        if (
            not isfinite(planned_arrival)
            or not isfinite(arrival)
            or abs(arrival - planned_arrival) > 1e-9
        ):
            raise ValueError(f"Arrival mismatch for {request_id}")
        if raw["terminal_state"] != "Finished":
            raise ValueError(f"Raw request did not finish: {request_id}")
        if (
            not isinstance(output_tokens, int)
            or isinstance(output_tokens, bool)
            or output_tokens <= 0
            or len(timestamps) != output_tokens
        ):
            raise ValueError(f"Output token count mismatch for {request_id}")
        if (
            not isfinite(terminal)
            or terminal < arrival
            or any(
                not isfinite(timestamp) or timestamp < arrival
                for timestamp in timestamps
            )
            or timestamps != sorted(timestamps)
            or terminal < timestamps[-1]
        ):
            raise ValueError(f"Invalid request timestamps for {request_id}")
    return raw_by_id


def validate_steps(
    steps: list[dict[str, Any]],
    requests: dict[str, dict[str, Any]],
    reported_step_count: int,
) -> None:
    for step in steps:
        validate_step_schema(step)
    if len(steps) != reported_step_count:
        raise ValueError("Step artifact count mismatch")
    emitted: Counter[str] = Counter()
    finished: Counter[str] = Counter()
    enforce_event_timestamps = any(
        step.get("schema_version", 1) >= 2 for step in steps
    )
    emitted_at: dict[str, list[float]] = {
        request_id: [] for request_id in requests
    }
    finished_at: dict[str, list[float]] = {
        request_id: [] for request_id in requests
    }
    for expected_index, step in enumerate(steps):
        if step["step_index"] != expected_index:
            raise ValueError("Step indices are not contiguous")
        for event in step["events"]:
            request_id = event["request_id"]
            if request_id not in requests:
                raise ValueError("Step references unknown request")
            if (
                type(event["emitted_token"]) is not bool
                or type(event["finished"]) is not bool
            ):
                raise ValueError("Step event flags must be booleans")
            emitted[request_id] += event["emitted_token"]
            finished[request_id] += event["finished"]
            if event["emitted_token"]:
                emitted_at[request_id].append(float(step["finished_at"]))
            if event["finished"]:
                finished_at[request_id].append(float(step["finished_at"]))
    for request_id, request in requests.items():
        if emitted[request_id] != len(request["token_timestamps"]):
            raise ValueError(f"Step emitted-token count mismatch for {request_id}")
        if finished[request_id] != 1:
            raise ValueError(f"Step finished-event count mismatch for {request_id}")
        if enforce_event_timestamps:
            timestamps = [
                float(value) for value in request["token_timestamps"]
            ]
            if len(timestamps) != len(emitted_at[request_id]) or any(
                abs(raw - event_time) > 1e-9
                for raw, event_time in zip(
                    timestamps,
                    emitted_at[request_id],
                    strict=True,
                )
            ):
                raise ValueError(
                    f"Step emitted-token timestamp mismatch for {request_id}"
                )
            terminal_at = request.get("terminal_at")
            if terminal_at is not None and (
                len(finished_at[request_id]) != 1
                or abs(float(terminal_at) - finished_at[request_id][0])
                > 1e-9
            ):
                raise ValueError(
                    f"Step finished timestamp mismatch for {request_id}"
                )


def validate_runtime(
    summary: dict[str, Any],
    requests: list[dict[str, Any]],
) -> float:
    runtime = summary["summary"]["runtime"]
    elapsed = runtime["elapsed_seconds"]
    if runtime["timed_out"] is not False:
        raise ValueError("Fully finished calibration run cannot be timed out")
    if (
        not isinstance(elapsed, (int, float))
        or isinstance(elapsed, bool)
        or not isfinite(elapsed)
    ):
        raise ValueError("Runtime elapsed_seconds must be finite")
    last_terminal = max(float(request["terminal_at"]) for request in requests)
    max_run = float(summary["measurement"]["max_run_seconds"])
    if elapsed < last_terminal or elapsed > max_run:
        raise ValueError("Runtime elapsed_seconds violates benchmark bounds")
    return float(elapsed)


def validate_measurement(
    summary: dict[str, Any],
    requests: list[dict[str, Any]],
) -> dict[str, Any]:
    start = float(summary["measurement"]["start_seconds"])
    end = float(summary["measurement"]["end_seconds"])
    duration = end - start
    if (
        not isfinite(start)
        or not isfinite(end)
        or start < 0
        or duration <= 0
    ):
        raise ValueError("Invalid measurement window")
    eligible = [
        request
        for request in requests
        if start <= float(request["arrival_at"]) < end
    ]
    reported_measurement = summary["summary"]["measurement"]
    for field, value in (("start", start), ("end", end), ("duration", duration)):
        require_close(
            reported_measurement[field],
            value,
            f"measurement.{field}",
        )
    require_close(
        reported_measurement["eligible_requests"],
        len(eligible),
        "measurement.eligible_requests",
    )
    return {"start": start, "end": end, "duration": duration, "eligible": eligible}


def validate_throughput(
    *,
    summary: dict[str, Any],
    trace: dict[str, Any],
    eligible: list[dict[str, Any]],
    start: float,
    end: float,
    duration: float,
    declared_offered_tps: float,
) -> dict[str, float]:
    offered_tokens = sum(
        int(request["max_output_tokens"])
        for request in trace["requests"]
        if start <= float(request["arrival_time"]) < end
    )
    offered_tps = offered_tokens / duration
    require_close(offered_tps, declared_offered_tps, "offered_output_tps")
    achieved_tps = (
        sum(len(request["token_timestamps"]) for request in eligible) / duration
    )
    require_close(
        summary["summary"]["output_throughput_tps"],
        achieved_tps,
        "output_throughput_tps",
    )
    return {
        "offered_output_tps": offered_tps,
        "achieved_output_throughput_tps": achieved_tps,
    }


def validate_measurement_and_throughput(
    summary: dict[str, Any],
    trace: dict[str, Any],
    requests: list[dict[str, Any]],
    declared_offered_tps: float,
) -> dict[str, Any]:
    measurement = validate_measurement(summary, requests)
    throughput = validate_throughput(
        summary=summary,
        trace=trace,
        eligible=measurement["eligible"],
        start=measurement["start"],
        end=measurement["end"],
        duration=measurement["duration"],
        declared_offered_tps=declared_offered_tps,
    )
    return {**measurement, **throughput}


def recompute_interactive_metrics(
    summary: dict[str, Any],
    requests: list[dict[str, Any]],
    duration: float,
) -> dict[str, Any]:
    interactive = [
        request for request in requests
        if request["request_class"] == "interactive"
    ]
    ttft_ms = [
        (
            float(request["token_timestamps"][0])
            - float(request["arrival_at"])
        ) * 1000
        for request in interactive
    ]
    itl_ms = [
        (float(tokens[index]) - float(tokens[index - 1])) * 1000
        for request in interactive
        for tokens in [request["token_timestamps"]]
        for index in range(1, len(tokens))
    ]
    reported_interactive = summary["summary"]["interactive"]
    require_close(
        reported_interactive["submitted"], len(interactive), "interactive.submitted"
    )
    require_distribution(
        reported_interactive["ttft_ms"],
        distribution(ttft_ms),
        "interactive.ttft_ms",
    )
    require_distribution(
        reported_interactive["itl_ms"],
        distribution(itl_ms),
        "interactive.itl_ms",
    )
    successes = recompute_slo_successes(interactive, summary["slo"])
    require_close(
        reported_interactive["slo_successes"], successes, "interactive.slo_successes"
    )
    require_close(
        reported_interactive["slo_goodput_rps"],
        successes / duration,
        "interactive.slo_goodput_rps",
    )
    return {
        "ttft": distribution(ttft_ms),
        "itl": distribution(itl_ms),
    }


def recompute_metrics(
    summary: dict[str, Any],
    trace: dict[str, Any],
    requests: list[dict[str, Any]],
    declared_offered_tps: float,
) -> dict[str, Any]:
    measurement = validate_measurement_and_throughput(
        summary,
        trace,
        requests,
        declared_offered_tps,
    )
    eligible = measurement["eligible"]
    interactive = recompute_interactive_metrics(
        summary,
        eligible,
        measurement["duration"],
    )
    long_requests = [
        request for request in eligible if request["request_class"] == "long"
    ]
    validate_long_metrics(summary, long_requests, measurement["duration"])
    return {
        "offered_output_tps": measurement["offered_output_tps"],
        "achieved_output_throughput_tps": (
            measurement["achieved_output_throughput_tps"]
        ),
        "interactive_ttft": interactive["ttft"],
        "interactive_itl": interactive["itl"],
        "window_end": measurement["end"],
    }


def recompute_slo_successes(
    requests: list[dict[str, Any]],
    slo: dict[str, Any],
) -> int:
    successes = 0
    for request in requests:
        timestamps = [float(value) for value in request["token_timestamps"]]
        ttft_ms = (timestamps[0] - float(request["arrival_at"])) * 1000
        itl_ms = [
            (timestamps[index] - timestamps[index - 1]) * 1000
            for index in range(1, len(timestamps))
        ]
        p99_itl = percentile(itl_ms, 0.99)
        if (
            ttft_ms <= float(slo["ttft_slo_ms"])
            and (
                (
                    p99_itl is not None
                    and p99_itl <= float(slo["itl_slo_ms"])
                )
                or (p99_itl is None and not slo["require_itl"])
            )
        ):
            successes += 1
    return successes


def validate_long_metrics(
    summary: dict[str, Any],
    requests: list[dict[str, Any]],
    duration: float,
) -> None:
    reported = summary["summary"]["long"]
    require_close(reported["submitted"], len(requests), "long.submitted")
    require_close(reported["finished"], len(requests), "long.finished")
    require_close(
        reported["request_goodput_rps"],
        len(requests) / duration,
        "long.request_goodput_rps",
    )
    require_close(
        reported["token_goodput_tps"],
        sum(len(request["token_timestamps"]) for request in requests) / duration,
        "long.token_goodput_tps",
    )


def build_run_record(
    *,
    summary_path: Path,
    load: str,
    seed: int,
    summary: dict[str, Any],
    requests: list[dict[str, Any]],
    elapsed: float,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    end = metrics["window_end"]
    last_terminal = max(float(request["terminal_at"]) for request in requests)
    backlog = sum(float(request["terminal_at"]) > end for request in requests)
    return {
        "load": load,
        "seed": seed,
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "repository_head": summary["repository_head"],
        "offered_output_tps": metrics["offered_output_tps"],
        "achieved_output_throughput_tps": (
            metrics["achieved_output_throughput_tps"]
        ),
        "submitted": len(requests),
        "finished": len(requests),
        "backlog_at_window_end": backlog,
        "reported_elapsed_seconds": elapsed,
        "drain_after_window_seconds": max(0.0, last_terminal - end),
        "reported_runtime_tail_seconds": max(0.0, elapsed - end),
        "interactive_ttft_p99_ms": metrics["interactive_ttft"]["p99"],
        "interactive_itl_p99_ms": metrics["interactive_itl"]["p99"],
        "interactive_max_itl_ms": metrics["interactive_itl"]["max"],
        "terminal_reconciled": True,
        "timed_out": False,
    }


def build_fingerprints(summary: dict[str, Any]) -> dict[str, Any]:
    model = {
        key: summary["model"][key]
        for key in ("repo_id", "revision", "sha256", "verified_sha256")
    }
    sampling = {
        key: value for key, value in summary["sampling"].items()
        if key != "seed"
    }
    return {
        "upstream": summary["upstream_commit"],
        "head": summary["repository_head"],
        "model": json.dumps(model, sort_keys=True),
        "engine": json.dumps(summary["engine"], sort_keys=True),
        "sampling": json.dumps(sampling, sort_keys=True),
        "slo": json.dumps(summary["slo"], sort_keys=True),
        "measurement": json.dumps(summary["measurement"], sort_keys=True),
        "runtime": json.dumps(
            {key: summary["runtime"][key] for key in RUNTIME_KEYS},
            sort_keys=True,
        ),
        "code": json.dumps(
            {key: summary["provenance"][key] for key in CODE_KEYS},
            sort_keys=True,
        ),
        "raw_pair": (
            summary["artifacts"]["requests_sha256"],
            summary["artifacts"]["steps_sha256"],
        ),
    }


def validate_run(
    *,
    summary_path: Path,
    load: str,
    seed: int,
    trace_stem: str | None = None,
    contract: dict[str, Any],
    repository_root: Path,
    results_directory: Path,
) -> dict[str, Any]:
    stem = f"{contract['run_id_prefix']}_{load}_seed{seed}"
    bound = validate_binding(
        summary_path=summary_path,
        stem=stem,
        trace_stem=trace_stem,
        seed=seed,
        contract=contract,
        repository_root=repository_root,
        results_directory=results_directory,
    )
    summary = bound["summary"]
    requests = bound["requests"]
    validate_terminal_counts(summary, requests)
    raw_by_id = validate_requests(bound["trace"], requests)
    validate_steps(
        bound["steps"],
        raw_by_id,
        summary["summary"]["runtime"]["steps"],
    )
    elapsed = validate_runtime(summary, requests)
    metrics = recompute_metrics(
        summary,
        bound["trace"],
        requests,
        float(contract["loads"][load]["offered_output_tps"]),
    )
    return package_validated_run(
        summary_path=summary_path,
        load=load,
        seed=seed,
        summary=summary,
        requests=requests,
        steps=bound["steps"],
        paths=bound["paths"],
        elapsed=elapsed,
        metrics=metrics,
    )


def package_validated_run(
    *,
    summary_path: Path,
    load: str,
    seed: int,
    summary: dict[str, Any],
    requests: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    paths: dict[str, Path],
    elapsed: float,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    fingerprints = build_fingerprints(summary)
    fingerprints["structural_raw_pair"] = structural_raw_fingerprint(
        requests,
        steps,
    )
    return {
        "run": build_run_record(
            summary_path=summary_path,
            load=load,
            seed=seed,
            summary=summary,
            requests=requests,
            elapsed=elapsed,
            metrics=metrics,
        ),
        "fingerprints": fingerprints,
        "paths": paths,
        "validated_requests": requests,
    }
