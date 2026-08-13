from __future__ import annotations

"""Strict replay helpers for Stage 12 slack-aware admission evidence."""

from collections import Counter
import hashlib
import json
import argparse
import subprocess
from pathlib import Path
from typing import Any

from experiments.aggregate_stage11 import (
    ADMISSION_V2_KEYS,
    CACHE_BLOCK_KEYS,
    CACHE_STATE_KEYS,
    finite_number,
    required_blocks,
    validate_cache_event,
)
from experiments.benchmark.open_loop import (
    AdmissionConfig,
    ReservationRuntime,
    SlackEstimator,
    RequestSpec,
    prepare_requests,
)
from experiments.benchmark.lifecycle import (
    RequestRecord,
    TerminalState,
    summarize_requests,
)
from experiments.benchmark.calibration_evidence import (
    recorded_path_is_absolute,
    recorded_path_matches,
    resolve_recorded_artifact_path,
)
from experiments.derive_stage12_eta import derive


REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_AGGREGATE_SOURCES = {
    "0fcb855e20e180d5b27a60b82261d6824fb07fa5a7b117a25d13edeee42e68df": (
        "562837582518579d1db32a7254cbb5b9e414a1f3",
        "experiments/aggregate_stage12.py",
    ),
}
CODE_PATHS = {
    "runner_sha256": Path("experiments/run_open_loop.py"),
    "open_loop_sha256": Path("experiments/benchmark/open_loop.py"),
    "lifecycle_sha256": Path("experiments/benchmark/lifecycle.py"),
    "llm_engine_sha256": Path("nanovllm/engine/llm_engine.py"),
    "scheduler_sha256": Path("nanovllm/engine/scheduler.py"),
    "block_manager_sha256": Path("nanovllm/engine/block_manager.py"),
    "model_runner_sha256": Path("nanovllm/engine/model_runner.py"),
    "config_module_sha256": Path("nanovllm/config.py"),
}
MANIFEST_KEYS = {
    "schema_version",
    "stage",
    "kind",
    "run_id",
    "config_path",
    "config_sha256",
    "trace_path",
    "trace_sha256",
    "execution_code",
    "aggregate_sha256",
    "upstream_commit",
    "calibration",
}
CALIBRATION_KEYS = {
    "config_path",
    "config_sha256",
    "trace_path",
    "trace_sha256",
    "requests_path",
    "requests_sha256",
    "steps_path",
    "steps_sha256",
    "summary_path",
    "summary_sha256",
    "phase_timings_path",
    "phase_timings_sha256",
    "derivation_script_path",
    "derivation_script_sha256",
    "derivation_output_path",
    "derivation_output_sha256",
    "eta_prefill_seconds",
    "eta_decode_seconds_per_token",
    "eta_safety_margin_seconds",
}


SLACK_KEYS = {
    "schema_version",
    "decision_index",
    "request_id",
    "observed_at",
    "cache_state_index",
    "admission_event_index",
    "cache_event_index",
    "full_required_blocks",
    "incremental_reservation_blocks",
    "reserved_blocks_before",
    "capacity_shortfall_blocks",
    "deadline_at",
    "predicted_free_at",
    "predicted_releasable_blocks",
    "slack_ms",
    "active_reservations",
    "reason",
}
ACTIVE_KEYS = {
    "request_id",
    "reservation_blocks",
    "admitted_at",
    "max_output_tokens",
    "generated_tokens",
    "last_token_at",
    "predicted_release_at",
}
ADMISSION_V3_KEYS = ADMISSION_V2_KEYS


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execution_code_fingerprint() -> dict[str, str]:
    return {
        field: sha256(REPO_ROOT / relative_path)
        for field, relative_path in CODE_PATHS.items()
    }


def aggregate_source_is_verified(expected_sha256: Any) -> bool:
    if not isinstance(expected_sha256, str):
        return False
    if expected_sha256 == sha256(Path(__file__)):
        return True
    source = HISTORICAL_AGGREGATE_SOURCES.get(expected_sha256)
    if source is None:
        return False
    commit, repository_path = source
    try:
        completed = subprocess.run(
            ["git", "show", f"{commit}:{repository_path}"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return hashlib.sha256(completed.stdout).hexdigest() == expected_sha256


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def resolve_recorded_manifest_path(
    value: Any,
    *,
    field: str,
    allowed_directory: Path,
) -> Path:
    """Resolve a frozen repository path independently of checkout location."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"manifest {field} differs")
    if not recorded_path_is_absolute(value):
        raise ValueError(f"manifest {field} must be an absolute path")
    try:
        return resolve_recorded_artifact_path(
            value,
            repository_root=REPO_ROOT,
            allowed_directory=allowed_directory,
            label=f"manifest {field}",
        )
    except ValueError as exception:
        raise ValueError(f"manifest {field} differs") from exception


def _manifest_path(
    value: Any,
    *,
    field: str,
    allowed_directory: Path,
) -> Path:
    return resolve_recorded_manifest_path(
        value,
        field=field,
        allowed_directory=allowed_directory,
    )


def load_frozen_manifest(
    *, manifest_path: Path,
    config_path: Path,
    replay: bool = False,
) -> dict[str, Any]:
    """Load a pre-run manifest; it is the Stage 12 verification trust root."""
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        set(document) != MANIFEST_KEYS
        or document.get("schema_version") != 1
        or document.get("stage") != "stage12"
        or document.get("kind") not in {"smoke", "formal"}
    ):
        raise ValueError("Stage 12 manifest schema differs")
    if set(document["execution_code"]) != set(CODE_PATHS):
        raise ValueError("manifest execution code differs")
    if not replay and document["execution_code"] != execution_code_fingerprint():
        raise ValueError("manifest execution source differs")
    if not aggregate_source_is_verified(document.get("aggregate_sha256")):
        raise ValueError("manifest aggregate source differs")
    bound_config = _manifest_path(
        document["config_path"],
        field="config_path",
        allowed_directory=REPO_ROOT / "experiments/configs",
    )
    if bound_config.resolve() != config_path.resolve() or sha256(bound_config) != document["config_sha256"]:
        raise ValueError("manifest config binding differs")
    config = json.loads(bound_config.read_text(encoding="utf-8"))
    if config.get("output", {}).get("run_id") != document.get("run_id"):
        raise ValueError("manifest run identifier differs")
    if config.get("upstream_commit") != document.get("upstream_commit"):
        raise ValueError("manifest upstream commit differs")
    trace_path = _manifest_path(
        document["trace_path"],
        field="trace_path",
        allowed_directory=REPO_ROOT / "experiments/data",
    )
    smoke_trace_reference = Path(config["workload"]["trace_path"])
    if not smoke_trace_reference.is_absolute():
        smoke_trace_reference = REPO_ROOT / smoke_trace_reference
    if (
        smoke_trace_reference.resolve() != trace_path.resolve()
        or sha256(trace_path) != document["trace_sha256"]
    ):
        raise ValueError("manifest trace binding differs")
    calibration = document.get("calibration")
    if not isinstance(calibration, dict) or set(calibration) != CALIBRATION_KEYS:
        raise ValueError("manifest calibration schema differs")
    calibration_domains = {
        "config": REPO_ROOT / "experiments/configs",
        "trace": REPO_ROOT / "experiments/data",
        "requests": REPO_ROOT / "experiments/results",
        "steps": REPO_ROOT / "experiments/results",
        "summary": REPO_ROOT / "experiments/results",
        "phase_timings": REPO_ROOT / "experiments/results",
        "derivation_script": REPO_ROOT / "experiments",
        "derivation_output": REPO_ROOT / "experiments/results",
    }
    calibration_paths: dict[str, Path] = {}
    for name, allowed_directory in calibration_domains.items():
        path = _manifest_path(
            calibration[f"{name}_path"],
            field=f"calibration.{name}_path",
            allowed_directory=allowed_directory,
        )
        if sha256(path) != calibration[f"{name}_sha256"]:
            raise ValueError(f"manifest calibration {name} binding differs")
        calibration_paths[name] = path
    calibration_config = json.loads(
        calibration_paths["config"].read_text(encoding="utf-8")
    )
    calibration_trace = json.loads(
        calibration_paths["trace"].read_text(encoding="utf-8")
    )
    calibration_summary = json.loads(
        calibration_paths["summary"].read_text(encoding="utf-8")
    )
    if (
        calibration_config.get("output", {}).get("run_id") == document["run_id"]
        or calibration_paths["config"].resolve() == bound_config.resolve()
        or calibration_paths["trace"].resolve() == trace_path.resolve()
        or calibration_config.get("admission", {}).get("policy") != "disabled"
        or calibration_config.get("profiling", {}).get("cuda_events") is not True
        or calibration_summary.get("status") != "passed"
        or calibration_summary.get("error") is not None
        or calibration_summary.get("provenance", {}).get("config_sha256")
        != calibration["config_sha256"]
        or calibration_summary.get("provenance", {}).get("trace_sha256")
        != calibration["trace_sha256"]
        or calibration_paths["derivation_script"].resolve()
        != (REPO_ROOT / "experiments/derive_stage12_eta.py").resolve()
    ):
        raise ValueError("manifest calibration provenance differs")
    calibration_trace_reference = Path(
        calibration_config.get("workload", {}).get("trace_path", "")
    )
    if not calibration_trace_reference.is_absolute():
        calibration_trace_reference = REPO_ROOT / calibration_trace_reference
    if calibration_trace_reference.resolve() != calibration_paths["trace"].resolve():
        raise ValueError("manifest calibration trace path differs")
    if (
        {field: calibration_summary["provenance"].get(field) for field in CODE_PATHS}
        != document["execution_code"]
        or calibration_summary.get("runtime", {}).get("cuda_available") is not True
    ):
        raise ValueError("manifest calibration execution provenance differs")
    for field in ("engine", "slo", "measurement"):
        if calibration_config.get(field) != config.get(field):
            raise ValueError(f"manifest calibration {field} differs from smoke")
        if calibration_summary.get(field) != calibration_config.get(field):
            raise ValueError(f"calibration summary {field} differs")
    calibration_sampling = calibration_config.get("sampling")
    target_sampling = config.get("sampling")
    if (
        not isinstance(calibration_sampling, dict)
        or not isinstance(target_sampling, dict)
        or set(calibration_sampling) != set(target_sampling)
        or any(
            calibration_sampling.get(field) != target_sampling.get(field)
            for field in calibration_sampling
            if field != "seed"
        )
        or calibration_summary.get("sampling") != calibration_sampling
    ):
        raise ValueError("manifest calibration sampling profile differs")
    calibration_model = calibration_config.get("model")
    summary_model = calibration_summary.get("model")
    if (
        calibration_model != config.get("model")
        or not isinstance(summary_model, dict)
        or any(summary_model.get(field) != calibration_model.get(field) for field in calibration_model)
        or summary_model.get("verified_sha256") != calibration_model.get("sha256")
    ):
        raise ValueError("manifest calibration model differs from smoke")
    for name, artifact_key in (
        ("requests", "requests"),
        ("steps", "steps"),
        ("phase_timings", "phase_timings"),
    ):
        if (
            not recorded_path_matches(
                calibration_summary.get("artifacts", {}).get(
                    f"{artifact_key}_path"
                ),
                calibration_paths[name],
                repository_root=REPO_ROOT,
                allowed_directory=REPO_ROOT / "experiments/results",
                label=f"calibration {name}",
            )
            or calibration_summary.get("artifacts", {}).get(f"{artifact_key}_sha256")
            != calibration[f"{name}_sha256"]
        ):
            raise ValueError(f"manifest calibration artifact differs: {name}")
    calibration_requests = read_jsonl(calibration_paths["requests"])
    calibration_steps = read_jsonl(calibration_paths["steps"])
    validate_raw(
        trace=calibration_trace,
        requests=calibration_requests,
        steps=calibration_steps,
        summary=calibration_summary,
    )
    derivation = json.loads(
        calibration_paths["derivation_output"].read_text(encoding="utf-8")
    )
    recomputed_eta = derive(
        steps=calibration_steps,
        timings=read_jsonl(calibration_paths["phase_timings"]),
    )
    if derivation != recomputed_eta:
        raise ValueError("manifest calibration derivation differs")
    eta_fields = (
        "eta_prefill_seconds",
        "eta_decode_seconds_per_token",
        "eta_safety_margin_seconds",
    )
    if any(
        not finite_number(calibration[field])
        or calibration[field] <= 0
        or calibration[field] != derivation.get(field)
        for field in eta_fields
    ):
        raise ValueError("manifest calibration ETA differs")
    if any(config["admission"].get(field) != calibration[field] for field in eta_fields):
        raise ValueError("manifest/config ETA differs")
    return {
        "config_sha256": document["config_sha256"],
        "trace_sha256": document["trace_sha256"],
        "execution_code": document["execution_code"],
    }


def _runtime_from_event(value: dict[str, Any]) -> ReservationRuntime:
    if set(value) != ACTIVE_KEYS:
        raise ValueError("slack active reservation schema differs")
    if (
        not isinstance(value["request_id"], str)
        or not value["request_id"]
        or not _positive_int(value["reservation_blocks"])
        or not finite_number(value["admitted_at"])
        or not _positive_int(value["max_output_tokens"])
        or not isinstance(value["generated_tokens"], int)
        or isinstance(value["generated_tokens"], bool)
        or value["generated_tokens"] < 0
        or value["generated_tokens"] > value["max_output_tokens"]
        or (
            value["last_token_at"] is not None
            and not finite_number(value["last_token_at"])
        )
    ):
        raise ValueError("slack active reservation value differs")
    return ReservationRuntime(
        request_id=value["request_id"],
        reservation_blocks=value["reservation_blocks"],
        admitted_at=value["admitted_at"],
        max_output_tokens=value["max_output_tokens"],
        generated_tokens=value["generated_tokens"],
        last_token_at=value["last_token_at"],
    )


def _validated_cache_states(
    cache_states: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Validate snapshots before using them as the prefix-cache replay truth."""
    states: dict[int, dict[str, Any]] = {}
    previous_time = -float("inf")
    for index, state in enumerate(cache_states):
        if set(state) != CACHE_STATE_KEYS or state.get("schema_version") != 1:
            raise ValueError("cache state schema differs")
        if (
            state.get("state_index") != index
            or not finite_number(state.get("observed_at"))
            or state["observed_at"] < previous_time
            or not isinstance(state.get("blocks"), list)
        ):
            raise ValueError("cache state ordering differs")
        block_ids: set[int] = set()
        for block in state["blocks"]:
            if (
                set(block) != CACHE_BLOCK_KEYS
                or not isinstance(block.get("block_id"), int)
                or isinstance(block.get("block_id"), bool)
                or block["block_id"] < 0
                or block["block_id"] in block_ids
                or not isinstance(block.get("hash"), int)
                or isinstance(block.get("hash"), bool)
                or type(block.get("used")) is not bool
                or not isinstance(block.get("ref_count"), int)
                or isinstance(block.get("ref_count"), bool)
                or block["ref_count"] < 0
                or not isinstance(block.get("token_ids"), list)
                or not block["token_ids"]
                or any(
                    not isinstance(token, int) or isinstance(token, bool)
                    for token in block["token_ids"]
                )
                or (block["used"] and block["ref_count"] <= 0)
                or (not block["used"] and block["ref_count"] != 0)
            ):
                raise ValueError("cache state block differs")
            block_ids.add(block["block_id"])
        states[index] = state
        previous_time = state["observed_at"]
    return states


def validate_admission_events(
    *,
    trace: dict[str, Any],
    requests: list[dict[str, Any]],
    admission_events: list[dict[str, Any]],
    cache_events: list[dict[str, Any]],
    cache_states: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    admission: dict[str, Any],
    block_size: int,
) -> None:
    """Replay Stage 12's complete FIFO lifecycle, including early rejection."""
    if len(admission_events) != len(cache_events):
        raise ValueError("admission/cache event counts differ")
    specs = {item["request_id"]: item for item in trace["requests"]}
    raw = {item["request_id"]: item for item in requests}
    if len(specs) != len(trace["requests"]) or set(specs) != set(raw):
        raise ValueError("trace/raw request identifiers differ")
    try:
        prepared = {
            item.spec.request_id: item.prompt_token_ids
            for item in prepare_requests(
                [RequestSpec(**item) for item in trace["requests"]],
                token_id_upper_bound=10000,
            )
        }
    except (TypeError, ValueError) as exception:
        raise ValueError("admission trace construction differs") from exception
    state_by_index = _validated_cache_states(cache_states)
    arrivals = sorted(
        trace["requests"], key=lambda item: (item["arrival_time"], item["request_id"])
    )
    queue: list[str] = []
    active: set[str] = set()
    rejected: set[str] = set()
    previews: dict[str, dict[str, Any]] = {}
    cursor = 0
    reserved = 0
    previous_time = -float("inf")
    for index, (event, cache_event) in enumerate(
        zip(admission_events, cache_events, strict=True)
    ):
        if set(event) != ADMISSION_V3_KEYS or event.get("schema_version") != 3:
            raise ValueError("Stage 12 admission schema differs")
        request_id = event.get("request_id")
        if request_id not in specs or event.get("event_index") != index:
            raise ValueError("admission event identity differs")
        action = event.get("action")
        if action not in {"admitted", "released", "rejected"}:
            raise ValueError("unknown admission action")
        if (
            not finite_number(event.get("arrival_at"))
            or not finite_number(event.get("observed_at"))
            or event["observed_at"] < previous_time
            or event["observed_at"] < event["arrival_at"]
            or abs(event["arrival_at"] - specs[request_id]["arrival_time"]) > 1e-9
            or not _positive_int(event.get("required_blocks"))
            or not _positive_int(event.get("reservation_blocks"))
            or not isinstance(event.get("reserved_blocks_after"), int)
            or isinstance(event.get("reserved_blocks_after"), bool)
        ):
            raise ValueError("admission event value differs")
        full = required_blocks(specs[request_id], block_size)
        if event["required_blocks"] != full:
            raise ValueError("admission footprint differs")
        while cursor < len(arrivals) and arrivals[cursor]["arrival_time"] <= event["observed_at"]:
            queue.append(arrivals[cursor]["request_id"])
            cursor += 1
        if action == "released":
            if event.get("queue_wait_ms") is not None or request_id not in active:
                raise ValueError("invalid release event")
            reservation = previews[request_id]["incremental_reservation_blocks"]
            if event["reservation_blocks"] != reservation or event.get("reason") is not None:
                raise ValueError("release reservation differs")
            active.remove(request_id)
            reserved -= reservation
        else:
            if not finite_number(event.get("queue_wait_ms")) or not queue or queue[0] != request_id:
                raise ValueError("admission FIFO ordering differs")
            wait_seconds = event["observed_at"] - event["arrival_at"]
            if abs(event["queue_wait_ms"] - wait_seconds * 1000) > 1e-6:
                raise ValueError("admission wait differs")
            if action == "admitted":
                if event.get("reason") is not None or request_id in active or request_id in rejected:
                    raise ValueError("invalid admitted event")
                if wait_seconds >= admission["max_queue_wait_seconds"]:
                    raise ValueError("admission occurred after deadline")
                if reserved + event["reservation_blocks"] > admission["total_kv_blocks"]:
                    raise ValueError("admission exceeds reservation capacity")
                queue.pop(0)
                active.add(request_id)
                reserved += event["reservation_blocks"]
            else:
                reason = event.get("reason")
                if request_id in active or reason not in {
                    "kv_reservation_timeout",
                    "predicted_deadline_miss",
                }:
                    raise ValueError("invalid rejected event")
                if reason == "kv_reservation_timeout" and wait_seconds < admission["max_queue_wait_seconds"]:
                    raise ValueError("timeout rejection occurred early")
                if reason == "predicted_deadline_miss" and wait_seconds >= admission["max_queue_wait_seconds"]:
                    raise ValueError("predicted rejection occurred after deadline")
                queue.pop(0)
                rejected.add(request_id)
        if event["reserved_blocks_after"] != reserved or reserved < 0:
            raise ValueError("admission reservation replay differs")
        state_index = cache_event.get("cache_state_index")
        if state_index not in state_by_index:
            raise ValueError("cache event references an unknown state")
        try:
            validate_cache_event(
                cache_event,
                index=index,
                action=action,
                request_id=request_id,
                observed_at=event["observed_at"],
                reserved_blocks_after=reserved,
                reason=event.get("reason"),
                expected_full_blocks=full,
                block_size=block_size,
                prompt_token_ids=prepared[request_id],
                cache_state=state_by_index[state_index],
                permit_prior_state=action == "released",
            )
        except ValueError as exception:
            raise ValueError(f"cache replay differs at admission event {index}") from exception
        if cache_event["incremental_reservation_blocks"] != event["reservation_blocks"]:
            raise ValueError("admission/cache reservation differs")
        if action == "admitted":
            previews[request_id] = cache_event
        elif action == "released":
            prior = previews.pop(request_id)
            for field in (
                "full_required_blocks", "matched_prefix_blocks", "active_shared_blocks",
                "inactive_cached_blocks", "matched_prefix_tokens",
                "incremental_reservation_blocks", "matched_block_ids", "active_block_ids",
                "inactive_block_ids", "cache_state_index",
            ):
                if cache_event[field] != prior[field]:
                    raise ValueError("release cache preview differs from admission")
        previous_time = event["observed_at"]
    if active or queue or cursor != len(arrivals) or reserved or previews:
        raise ValueError("admission lifecycle did not return to zero")
    for request_id, record in raw.items():
        actions = [
            event["action"] for event in admission_events if event["request_id"] == request_id
        ]
        if record.get("terminal_state") == "Finished":
            if actions != ["admitted", "released"]:
                raise ValueError("finished request lifecycle differs")
            admitted, released = (
                event
                for event in admission_events
                if event["request_id"] == request_id
            )
            if (
                abs(admitted["observed_at"] - record.get("admitted_at", float("nan"))) > 1e-6
                or abs(released["observed_at"] - record.get("terminal_at", float("nan"))) > 1e-6
            ):
                raise ValueError("finished request timestamps differ")
        elif record.get("terminal_state") == "Rejected":
            if actions != ["rejected"] or record.get("terminal_reason") not in {
                "kv_reservation_timeout", "predicted_deadline_miss",
            }:
                raise ValueError("rejected request lifecycle differs")
            rejected_event = next(
                event for event in admission_events if event["request_id"] == request_id
            )
            if abs(rejected_event["observed_at"] - record.get("terminal_at", float("nan"))) > 1e-6:
                raise ValueError("rejected request timestamp differs")
        else:
            raise ValueError("formal run has non-terminal request")
    intervals: dict[str, tuple[float, float] | None] = {}
    for request_id, record in raw.items():
        events = [event for event in admission_events if event["request_id"] == request_id]
        if record["terminal_state"] == "Rejected":
            intervals[request_id] = None
        else:
            intervals[request_id] = (events[0]["observed_at"], events[1]["observed_at"])
    for step in steps:
        for event in step["events"]:
            interval = intervals[event["request_id"]]
            if interval is None:
                raise ValueError("rejected request has scheduled step event")
            admitted_at, released_at = interval
            if (
                step["started_at"] < admitted_at
                or step["finished_at"] > released_at
            ):
                raise ValueError("step event falls outside admission interval")


def validate_slack_events(
    *,
    slack_events: list[dict[str, Any]],
    admission_events: list[dict[str, Any]],
    cache_events: list[dict[str, Any]],
    cache_states: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    trace: dict[str, Any],
    admission: dict[str, Any],
    block_size: int,
) -> None:
    """Replay every early rejection from raw lifecycle and decision evidence."""
    if admission.get("policy") != "slack_aware_prefix_fifo":
        raise ValueError("Stage 12 replay requires slack-aware policy")
    capacity = admission.get("total_kv_blocks")
    if not _positive_int(capacity):
        raise ValueError("slack capacity differs")
    config = AdmissionConfig(
        policy="slack_aware_prefix_fifo",
        total_kv_blocks=capacity,
        kvcache_block_size=block_size,
        max_queue_wait_seconds=admission.get("max_queue_wait_seconds"),
        observe_prefix_cache=admission.get("observe_prefix_cache"),
        eta_prefill_seconds=admission.get("eta_prefill_seconds"),
        eta_decode_seconds_per_token=admission.get("eta_decode_seconds_per_token"),
        eta_safety_margin_seconds=admission.get("eta_safety_margin_seconds"),
    )
    estimator = SlackEstimator(config)
    specs = {item["request_id"]: item for item in trace["requests"]}
    raw = {item["request_id"]: item for item in requests}
    if set(specs) != set(raw):
        raise ValueError("slack trace/request binding differs")
    try:
        prepared = {
            item.spec.request_id: item.prompt_token_ids
            for item in prepare_requests(
                [RequestSpec(**item) for item in trace["requests"]],
                token_id_upper_bound=10000,
            )
        }
    except (TypeError, ValueError) as exception:
        raise ValueError("slack trace construction differs") from exception
    state_by_index = {item.get("state_index"): item for item in cache_states}
    if len(state_by_index) != len(cache_states):
        raise ValueError("cache state identifiers differ")
    previous_admission_index = -1
    for event_index, event in enumerate(slack_events):
        if set(event) != SLACK_KEYS or event.get("schema_version") != 1:
            raise ValueError("slack event schema differs")
        if event.get("decision_index") != event_index or event.get("reason") != "predicted_deadline_miss":
            raise ValueError("slack event identity differs")
        if not all(finite_number(event.get(field)) for field in ("observed_at", "deadline_at", "predicted_free_at", "slack_ms")):
            raise ValueError("slack timing differs")
        admission_index = event.get("admission_event_index")
        cache_index = event.get("cache_event_index")
        if (
            not isinstance(admission_index, int)
            or admission_index <= previous_admission_index
            or admission_index >= len(admission_events)
            or not isinstance(cache_index, int)
            or cache_index < 0
            or cache_index >= len(cache_events)
        ):
            raise ValueError("slack event index differs")
        previous_admission_index = admission_index
        admission_event = admission_events[admission_index]
        cache_event = cache_events[cache_index]
        request_id = event.get("request_id")
        if request_id not in specs or (
            admission_event.get("action"), admission_event.get("request_id"), admission_event.get("reason")
        ) != ("rejected", request_id, "predicted_deadline_miss") or (
            cache_event.get("action"), cache_event.get("request_id"), cache_event.get("reason")
        ) != ("rejected", request_id, "predicted_deadline_miss"):
            raise ValueError("slack rejection binding differs")
        if cache_event.get("cache_state_index") != event.get("cache_state_index") or event["cache_state_index"] not in state_by_index:
            raise ValueError("slack cache-state binding differs")
        if abs(admission_event.get("observed_at", float("nan")) - event["observed_at"]) > 1e-6 or abs(cache_event.get("observed_at", float("nan")) - event["observed_at"]) > 1e-6:
            raise ValueError("slack observed timestamp differs")
        expected_active: dict[str, ReservationRuntime] = {}
        for prior in admission_events[:admission_index]:
            action = prior.get("action")
            prior_id = prior.get("request_id")
            if action == "admitted":
                reservation = prior.get("reservation_blocks")
                if prior_id not in specs or not _positive_int(reservation):
                    raise ValueError("slack admission source differs")
                expected_active[prior_id] = ReservationRuntime(
                    request_id=prior_id,
                    reservation_blocks=reservation,
                    admitted_at=prior.get("observed_at"),
                    max_output_tokens=specs[prior_id]["max_output_tokens"],
                )
            elif action == "released":
                expected_active.pop(prior_id, None)
        for step in steps:
            if step.get("finished_at", float("inf")) > event["observed_at"]:
                continue
            for step_event in step.get("events", []):
                runtime = expected_active.get(step_event.get("request_id"))
                if runtime is not None and step_event.get("emitted_token") is True:
                    runtime.generated_tokens += 1
                    runtime.last_token_at = step["finished_at"]
        active = [_runtime_from_event(item) for item in event["active_reservations"]]
        if len({item.request_id for item in active}) != len(active):
            raise ValueError("slack active reservations are duplicated")
        if any(item.request_id == request_id for item in active):
            raise ValueError("rejected request cannot already be active")
        if [
            (
                item.request_id,
                item.reservation_blocks,
                item.admitted_at,
                item.max_output_tokens,
                item.generated_tokens,
                item.last_token_at,
            )
            for item in active
        ] != [
            (
                item.request_id,
                item.reservation_blocks,
                item.admitted_at,
                item.max_output_tokens,
                item.generated_tokens,
                item.last_token_at,
            )
            for item in expected_active.values()
        ]:
            raise ValueError("slack active reservation replay differs")
        for raw_active, expected in zip(event["active_reservations"], active, strict=True):
            if raw_active["predicted_release_at"] != estimator.predict_release_at(expected):
                raise ValueError("slack active ETA differs")
        reserved = sum(item.reservation_blocks for item in active)
        if reserved != event.get("reserved_blocks_before"):
            raise ValueError("slack reserved footprint differs")
        spec = specs[request_id]
        full = required_blocks(spec, block_size)
        incremental = cache_event.get("incremental_reservation_blocks")
        validate_cache_event(
            cache_event,
            index=cache_index,
            action="rejected",
            request_id=request_id,
            observed_at=event["observed_at"],
            reserved_blocks_after=cache_event.get("reserved_blocks_after"),
            reason="predicted_deadline_miss",
            expected_full_blocks=full,
            block_size=block_size,
            prompt_token_ids=prepared[request_id],
            cache_state=state_by_index[event["cache_state_index"]],
            permit_prior_state=False,
        )
        if (
            admission_event.get("required_blocks") != full
            or admission_event.get("reservation_blocks") != incremental
            or admission_event.get("reserved_blocks_after") != reserved
            or cache_event.get("reserved_blocks_after") != reserved
            or event.get("full_required_blocks") != full
            or event.get("incremental_reservation_blocks") != incremental
            or not _positive_int(incremental)
        ):
            raise ValueError("slack request footprint differs")
        shortfall = reserved + incremental - capacity
        if shortfall <= 0 or event.get("capacity_shortfall_blocks") != shortfall:
            raise ValueError("slack shortfall differs")
        predicted = estimator.earliest_capacity_time(active, shortfall_blocks=shortfall)
        if predicted is None or (
            abs(predicted[0] - event["predicted_free_at"]) > 1e-6
            or predicted[1] != event.get("predicted_releasable_blocks")
        ):
            raise ValueError("slack ETA replay differs")
        deadline = spec["arrival_time"] + config.max_queue_wait_seconds
        if (
            abs(deadline - event["deadline_at"]) > 1e-6
            or event["observed_at"] >= deadline
            or event["predicted_free_at"] <= deadline
            or abs(event["slack_ms"] - (deadline - event["predicted_free_at"]) * 1000) > 1e-6
            or raw[request_id].get("terminal_reason") != "predicted_deadline_miss"
            or raw[request_id].get("terminal_state") != "Rejected"
            or abs(raw[request_id].get("terminal_at", float("nan")) - event["observed_at"]) > 1e-6
        ):
            raise ValueError("slack deadline contract differs")


def validate_raw(
    *,
    trace: dict[str, Any],
    requests: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    """Recompute terminal records and SLO summary from Stage 12 raw output."""
    planned = {item["request_id"]: item for item in trace["requests"]}
    raw = {item["request_id"]: item for item in requests}
    if (
        len(planned) != len(trace["requests"])
        or len(raw) != len(requests)
        or set(planned) != set(raw)
    ):
        raise ValueError("trace/raw request identifiers differ")
    emitted: Counter[str] = Counter()
    finished: Counter[str] = Counter()
    emitted_at: dict[str, list[float]] = {request_id: [] for request_id in raw}
    finished_at: dict[str, list[float]] = {request_id: [] for request_id in raw}
    previous_finished_at = -float("inf")
    step_keys = {
        "step_index", "started_at", "finished_at", "duration_ms", "phase",
        "num_scheduled_tokens", "events",
    }
    event_keys = {
        "request_id", "seq_id", "phase", "num_scheduled_tokens",
        "emitted_token", "finished",
    }
    for index, step in enumerate(steps):
        if (
            set(step) != step_keys
            or step.get("step_index") != index
            or not isinstance(step.get("step_index"), int)
            or isinstance(step.get("step_index"), bool)
            or step.get("phase") not in {"prefill", "decode"}
            or not isinstance(step.get("num_scheduled_tokens"), int)
            or isinstance(step.get("num_scheduled_tokens"), bool)
            or step["num_scheduled_tokens"] < 0
            or not isinstance(step.get("events"), list)
            or any(
                not finite_number(step.get(field))
                for field in ("started_at", "finished_at", "duration_ms")
            )
            or step["finished_at"] < step["started_at"]
            or step["finished_at"] < previous_finished_at
        ):
            raise ValueError("step schema differs")
        if abs(step["duration_ms"] - (step["finished_at"] - step["started_at"]) * 1000) > 1e-4:
            raise ValueError("step duration differs")
        token_sum = 0
        for event in step["events"]:
            if (
                set(event) != event_keys
                or event.get("request_id") not in raw
                or event.get("phase") != step["phase"]
                or not isinstance(event.get("seq_id"), int)
                or isinstance(event.get("seq_id"), bool)
                or not isinstance(event.get("num_scheduled_tokens"), int)
                or isinstance(event.get("num_scheduled_tokens"), bool)
                or event["num_scheduled_tokens"] < 0
                or type(event.get("emitted_token")) is not bool
                or type(event.get("finished")) is not bool
            ):
                raise ValueError("step event schema differs")
            token_sum += event["num_scheduled_tokens"]
            emitted[event["request_id"]] += event["emitted_token"]
            finished[event["request_id"]] += event["finished"]
            if event["emitted_token"]:
                emitted_at[event["request_id"]].append(step["finished_at"])
            if event["finished"]:
                finished_at[event["request_id"]].append(step["finished_at"])
        if token_sum != step["num_scheduled_tokens"]:
            raise ValueError("step/event token totals differ")
        previous_finished_at = step["finished_at"]
    rebuilt: list[RequestRecord] = []
    for request_id, item in raw.items():
        spec = planned[request_id]
        if (
            item.get("request_class") != spec["request_class"]
            or not finite_number(item.get("arrival_at"))
            or abs(item["arrival_at"] - spec["arrival_time"]) > 1e-9
        ):
            raise ValueError("raw request trace binding differs")
        record = RequestRecord(
            request_id=request_id,
            request_class=item["request_class"],
            arrival_at=item["arrival_at"],
        )
        state = item.get("terminal_state")
        if state == "Finished":
            timestamps = item.get("token_timestamps")
            if (
                item.get("terminal_reason") is not None
                or not finite_number(item.get("admitted_at"))
                or not isinstance(timestamps, list)
                or len(timestamps) != spec["max_output_tokens"]
                or timestamps != sorted(timestamps)
                or timestamps != emitted_at[request_id]
                or emitted[request_id] != len(timestamps)
                or finished[request_id] != 1
                or not finite_number(item.get("terminal_at"))
                or item["terminal_at"] != finished_at[request_id][0]
                or any(
                    not finite_number(value) or value < item["arrival_at"]
                    for value in [item["admitted_at"], *timestamps, item["terminal_at"]]
                )
            ):
                raise ValueError("finished raw lifecycle differs")
            record.mark_admitted(item["admitted_at"])
            for timestamp in timestamps:
                record.record_token(timestamp)
            record.mark_terminal(TerminalState.FINISHED, item["terminal_at"])
        elif state == "Rejected":
            if (
                item.get("admitted_at") is not None
                or item.get("token_timestamps")
                or emitted[request_id]
                or finished[request_id]
                or item.get("terminal_reason") not in {
                    "kv_reservation_timeout", "predicted_deadline_miss",
                }
                or not finite_number(item.get("terminal_at"))
                or item["terminal_at"] < item["arrival_at"]
            ):
                raise ValueError("rejected raw lifecycle differs")
            record.mark_terminal(
                TerminalState.REJECTED,
                item["terminal_at"],
                reason=item["terminal_reason"],
            )
        else:
            raise ValueError("formal run has non-terminal request")
        rebuilt.append(record)
    recomputed = summarize_requests(
        rebuilt,
        measurement_start=summary["measurement"]["start_seconds"],
        measurement_end=summary["measurement"]["end_seconds"],
        ttft_slo_ms=summary["slo"]["ttft_slo_ms"],
        itl_slo_ms=summary["slo"]["itl_slo_ms"],
        require_itl=summary["slo"]["require_itl"],
    )
    for key, value in recomputed.items():
        if summary["summary"].get(key) != value:
            raise ValueError(f"raw metric recomputation differs: {key}")


def validate_run_artifacts(
    *,
    config_path: Path,
    summary_path: Path,
    manifest_path: Path,
) -> None:
    """Validate one completed slack-aware run without trusting summary paths."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    frozen = load_frozen_manifest(
        manifest_path=manifest_path,
        config_path=config_path,
        replay=True,
    )
    manifest_reference = summary.get("stage12_manifest")
    if (
        not isinstance(manifest_reference, dict)
        or set(manifest_reference) != {"path", "sha256"}
        or not recorded_path_matches(
            manifest_reference.get("path"),
            manifest_path,
            repository_root=REPO_ROOT,
            allowed_directory=REPO_ROOT / "experiments",
            label="Stage 12 manifest",
        )
        or manifest_reference.get("sha256") != sha256(manifest_path)
    ):
        raise ValueError("summary Stage 12 manifest binding differs")
    if config.get("admission", {}).get("policy") != "slack_aware_prefix_fifo":
        raise ValueError("Stage 12 config policy differs")
    run_id = config.get("output", {}).get("run_id")
    if not isinstance(run_id, str) or summary_path.name != f"{run_id}.summary.json":
        raise ValueError("Stage 12 run identifier differs")
    output_directory = Path(config["output"]["directory"])
    if not output_directory.is_absolute():
        output_directory = REPO_ROOT / output_directory
    if summary_path.parent != output_directory:
        raise ValueError("summary output directory differs")
    if summary.get("status") != "passed" or summary.get("error") is not None:
        raise ValueError("formal run did not pass")
    if (
        sha256(config_path) != frozen["config_sha256"]
        or summary.get("provenance", {}).get("config_sha256")
        != frozen["config_sha256"]
    ):
        raise ValueError("config hash differs")
    trace_path = Path(config["workload"]["trace_path"])
    if not trace_path.is_absolute():
        trace_path = REPO_ROOT / trace_path
    if (
        sha256(trace_path) != frozen["trace_sha256"]
        or summary["provenance"].get("trace_sha256") != frozen["trace_sha256"]
    ):
        raise ValueError("trace hash differs")
    if {
        field: summary["provenance"].get(field) for field in CODE_PATHS
    } != frozen["execution_code"]:
        raise ValueError("execution code fingerprint differs")
    for field in ("engine", "sampling", "slo", "measurement"):
        if summary.get(field) != config.get(field):
            raise ValueError(f"summary {field} differs from frozen config")
    if (
        summary.get("upstream_commit") != config.get("upstream_commit")
        or summary.get("model") is None
        or config.get("model") is None
        or any(
            summary["model"].get(field) != config["model"].get(field)
            for field in ("repo_id", "revision", "sha256")
        )
        or summary["model"].get("verified_sha256") != config["model"].get("sha256")
    ):
        raise ValueError("summary model provenance differs")
    expected_admission = config["admission"]
    if summary.get("admission") is None or any(
        summary["admission"].get(key) != value
        for key, value in expected_admission.items()
    ):
        raise ValueError("admission summary/config differs")
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("artifact manifest differs")
    documents: dict[str, list[dict[str, Any]]] = {}
    for name in (
        "requests", "steps", "admission_events", "cache_events", "cache_states", "slack_events"
    ):
        path_key = f"{name}_path"
        hash_key = f"{name}_sha256"
        artifact_path = summary_path.parent / f"{run_id}.{name.replace('_events', '').replace('_states', '_states')}.jsonl"
        # Never let an artifact path in the untrusted summary choose the file;
        # only use it as an exact, auditable spelling check.
        if name == "admission_events":
            artifact_path = summary_path.parent / f"{run_id}.admission.jsonl"
        elif name == "cache_events":
            artifact_path = summary_path.parent / f"{run_id}.cache.jsonl"
        elif name == "cache_states":
            artifact_path = summary_path.parent / f"{run_id}.cache_states.jsonl"
        elif name == "slack_events":
            artifact_path = summary_path.parent / f"{run_id}.slack.jsonl"
        if (
            not recorded_path_matches(
                artifacts.get(path_key),
                artifact_path,
                repository_root=REPO_ROOT,
                allowed_directory=summary_path.parent,
                label=f"Stage 12 {name}",
            )
            or artifacts.get(hash_key) != sha256(artifact_path)
        ):
            raise ValueError(f"artifact hash/path differs: {name}")
        documents[name] = read_jsonl(artifact_path)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    validate_raw(
        trace=trace,
        requests=documents["requests"],
        steps=documents["steps"],
        summary=summary,
    )
    block_size = config["engine"]["kvcache_block_size"]
    validate_admission_events(
        trace=trace,
        requests=documents["requests"],
        admission_events=documents["admission_events"],
        cache_events=documents["cache_events"],
        cache_states=documents["cache_states"],
        steps=documents["steps"],
        admission=summary["admission"],
        block_size=block_size,
    )
    validate_slack_events(
        slack_events=documents["slack_events"],
        admission_events=documents["admission_events"],
        cache_events=documents["cache_events"],
        cache_states=documents["cache_states"],
        requests=documents["requests"],
        steps=documents["steps"],
        trace=trace,
        admission=summary["admission"],
        block_size=block_size,
    )
    expected_early = {
        index
        for index, event in enumerate(documents["admission_events"])
        if event.get("action") == "rejected"
        and event.get("reason") == "predicted_deadline_miss"
    }
    recorded_early = {
        event.get("admission_event_index") for event in documents["slack_events"]
    }
    if expected_early != recorded_early or len(recorded_early) != len(documents["slack_events"]):
        raise ValueError("early rejection/slack evidence differs")
    terminal = summary["summary"].get("terminal_counts")
    counts = Counter(item.get("terminal_state") for item in documents["requests"])
    states = ("Finished", "Rejected", "Failed", "Cancelled", "Unfinished")
    if (
        not isinstance(terminal, dict)
        or terminal.get("submitted") != len(documents["requests"])
        or not terminal.get("reconciled")
        or any(terminal.get(state) != counts[state] for state in states)
        or counts["Failed"]
        or counts["Cancelled"]
        or counts["Unfinished"]
        or summary.get("summary", {}).get("runtime", {}).get("timed_out")
        or summary.get("pressure", {}).get("oom_detected")
        or summary.get("runtime", {}).get("cuda_available") is not True
        or summary.get("pressure", {}).get("kv_cache", {}).get("final_used_blocks") != 0
        or summary["admission"].get("final_reserved_blocks") != 0
    ):
        raise ValueError("completion gate differs")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate one frozen Stage 12 smoke artifact set."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    validate_run_artifacts(
        config_path=args.config.resolve(),
        summary_path=args.summary.resolve(),
        manifest_path=args.manifest.resolve(),
    )
    print("Stage 12 artifact validation passed")


if __name__ == "__main__":
    main()
