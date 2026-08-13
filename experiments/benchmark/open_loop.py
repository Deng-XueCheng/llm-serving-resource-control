from __future__ import annotations

import hashlib
import json
import random
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any, Callable, Protocol

import torch

from nanovllm import SamplingParams
from nanovllm.engine.block_manager import PrefixCachePreview
from nanovllm.engine.llm_engine import EngineStepResult

from experiments.benchmark.lifecycle import (
    RequestRecord,
    TerminalState,
    summarize_requests,
)


TRACE_V1_KEYS = {
    "request_id",
    "request_class",
    "arrival_time",
    "prompt_length",
    "max_output_tokens",
    "seed",
}
TRACE_V2_KEYS = TRACE_V1_KEYS | {
    "prefix_group",
    "shared_prefix_length",
}


@dataclass(frozen=True, slots=True)
class RequestSpec:
    request_id: str
    request_class: str
    arrival_time: float
    prompt_length: int
    max_output_tokens: int
    seed: int
    prefix_group: str | None = None
    shared_prefix_length: int = 0


@dataclass(frozen=True, slots=True)
class PreparedRequest:
    spec: RequestSpec
    prompt_token_ids: list[int]


@dataclass(frozen=True, slots=True)
class AdmissionConfig:
    policy: str = "disabled"
    total_kv_blocks: int | None = None
    kvcache_block_size: int = 256
    max_queue_wait_seconds: float = 0.0
    observe_prefix_cache: bool = False
    eta_prefill_seconds: float = 0.0
    eta_decode_seconds_per_token: float = 0.0
    eta_safety_margin_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.policy not in {
            "disabled",
            "kv_aware_fifo",
            "prefix_aware_fifo",
            "slack_aware_prefix_fifo",
        }:
            raise ValueError("Unknown admission policy")
        if not isinstance(self.observe_prefix_cache, bool):
            raise ValueError("observe_prefix_cache must be a boolean")
        if self.policy == "disabled":
            if any(
                getattr(self, field_name) != 0
                for field_name in (
                    "eta_prefill_seconds",
                    "eta_decode_seconds_per_token",
                    "eta_safety_margin_seconds",
                )
            ):
                raise ValueError("ETA fields are only valid for slack-aware admission")
            return
        if (
            self.policy in {"prefix_aware_fifo", "slack_aware_prefix_fifo"}
            and not self.observe_prefix_cache
        ):
            raise ValueError(
                "prefix-aware admission requires prefix cache observation"
            )
        if (
            not isinstance(self.total_kv_blocks, int)
            or isinstance(self.total_kv_blocks, bool)
            or self.total_kv_blocks <= 0
        ):
            raise ValueError("KV-aware admission requires total KV blocks")
        if (
            not isinstance(self.kvcache_block_size, int)
            or isinstance(self.kvcache_block_size, bool)
            or self.kvcache_block_size <= 0
        ):
            raise ValueError("kvcache block size must be positive")
        if (
            not isinstance(self.max_queue_wait_seconds, (int, float))
            or isinstance(self.max_queue_wait_seconds, bool)
            or not isfinite(float(self.max_queue_wait_seconds))
            or self.max_queue_wait_seconds <= 0
        ):
            raise ValueError("max queue wait must be positive")
        eta_fields = (
            "eta_prefill_seconds",
            "eta_decode_seconds_per_token",
            "eta_safety_margin_seconds",
        )
        if self.policy == "slack_aware_prefix_fifo":
            for field_name in eta_fields:
                value = getattr(self, field_name)
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not isfinite(float(value))
                    or value <= 0
                ):
                    raise ValueError(f"{field_name} must be positive")
        elif any(getattr(self, field_name) != 0 for field_name in eta_fields):
            raise ValueError("ETA fields are only valid for slack-aware admission")


@dataclass(slots=True)
class ReservationRuntime:
    request_id: str
    reservation_blocks: int
    admitted_at: float | None
    max_output_tokens: int
    generated_tokens: int = 0
    last_token_at: float | None = None


class SlackEstimator:
    def __init__(self, config: AdmissionConfig) -> None:
        if config.policy != "slack_aware_prefix_fifo":
            raise ValueError("SlackEstimator requires slack-aware admission")
        self.config = config

    def predict_release_at(self, runtime: ReservationRuntime) -> float | None:
        if runtime.admitted_at is None:
            return None
        if runtime.generated_tokens < 0 or runtime.generated_tokens > runtime.max_output_tokens:
            return None
        if runtime.generated_tokens == 0:
            return (
                runtime.admitted_at
                + self.config.eta_prefill_seconds
                + runtime.max_output_tokens * self.config.eta_decode_seconds_per_token
                + self.config.eta_safety_margin_seconds
            )
        if runtime.last_token_at is None:
            return None
        remaining_tokens = runtime.max_output_tokens - runtime.generated_tokens
        return (
            max(
                runtime.last_token_at,
                runtime.admitted_at + self.config.eta_prefill_seconds,
            )
            + remaining_tokens * self.config.eta_decode_seconds_per_token
            + self.config.eta_safety_margin_seconds
        )

    def earliest_capacity_time(
        self,
        active: list[ReservationRuntime],
        *,
        shortfall_blocks: int,
    ) -> tuple[float, int] | None:
        if shortfall_blocks <= 0:
            raise ValueError("shortfall_blocks must be positive")
        candidates: list[tuple[float, ReservationRuntime]] = []
        for runtime in active:
            predicted = self.predict_release_at(runtime)
            if predicted is None:
                return None
            candidates.append((predicted, runtime))
        released = 0
        for predicted, runtime in sorted(candidates, key=lambda item: item[0]):
            released += runtime.reservation_blocks
            if released >= shortfall_blocks:
                return predicted, released
        return None


class KVAdmissionController:
    def __init__(self, config: AdmissionConfig) -> None:
        self.config = config
        self.reserved_by_request: dict[str, int] = {}
        self.peak_reserved_blocks = 0
        self.max_queue_wait_seconds = 0.0

    def required_blocks(self, request: PreparedRequest) -> int:
        tokens = (
            request.spec.prompt_length + request.spec.max_output_tokens - 1
        )
        return (tokens + self.config.kvcache_block_size - 1) // (
            self.config.kvcache_block_size
        )

    def reservation_blocks(
        self,
        request: PreparedRequest,
        preview: PrefixCachePreview | None,
    ) -> int:
        full_required_blocks = self.required_blocks(request)
        if self.config.policy not in {
            "prefix_aware_fifo",
            "slack_aware_prefix_fifo",
        }:
            return full_required_blocks
        if preview is None:
            raise RuntimeError("Prefix-aware admission requires cache preview")
        if preview.full_required_blocks != full_required_blocks:
            raise RuntimeError("Cache preview footprint differs from request")
        if (
            preview.matched_prefix_blocks
            != preview.active_shared_blocks + preview.inactive_cached_blocks
            or preview.active_shared_blocks < 0
            or preview.inactive_cached_blocks < 0
            or preview.matched_prefix_blocks >= full_required_blocks
            or preview.incremental_reservation_blocks
            != full_required_blocks - preview.active_shared_blocks
            or preview.incremental_reservation_blocks <= 0
        ):
            raise RuntimeError("Cache preview violates reservation contract")
        return preview.incremental_reservation_blocks

    @property
    def reserved_blocks(self) -> int:
        return sum(self.reserved_by_request.values())

    def can_admit(
        self,
        request: PreparedRequest,
        preview: PrefixCachePreview | None,
    ) -> bool:
        if self.config.policy == "disabled":
            return True
        required = self.reservation_blocks(request, preview)
        assert self.config.total_kv_blocks is not None
        if required > self.config.total_kv_blocks:
            raise ValueError("Request KV footprint exceeds admission capacity")
        return self.reserved_blocks + required <= self.config.total_kv_blocks

    def admit(
        self,
        request: PreparedRequest,
        queue_wait_seconds: float,
        preview: PrefixCachePreview | None,
    ) -> None:
        if request.spec.request_id in self.reserved_by_request:
            raise RuntimeError("Request admission reservation already exists")
        self.max_queue_wait_seconds = max(
            self.max_queue_wait_seconds,
            queue_wait_seconds,
        )
        if self.config.policy == "disabled":
            return
        self.reserved_by_request[request.spec.request_id] = (
            self.reservation_blocks(request, preview)
        )
        self.peak_reserved_blocks = max(
            self.peak_reserved_blocks,
            self.reserved_blocks,
        )

    def reject(self, queue_wait_seconds: float) -> None:
        self.max_queue_wait_seconds = max(
            self.max_queue_wait_seconds,
            queue_wait_seconds,
        )

    def release(self, request_id: str) -> None:
        if self.config.policy == "disabled":
            return
        if self.reserved_by_request.pop(request_id, None) is None:
            raise RuntimeError("Finished request lacks admission reservation")

    def snapshot(self, records: list[RequestRecord]) -> dict[str, Any]:
        snapshot = {
            "schema_version": 1,
            "policy": self.config.policy,
            "total_kv_blocks": self.config.total_kv_blocks,
            "max_queue_wait_seconds": self.config.max_queue_wait_seconds,
            "admitted_requests": sum(
                record.admitted_at is not None for record in records
            ),
            "rejected_requests": sum(
                record.terminal_state is TerminalState.REJECTED
                for record in records
            ),
            "max_observed_queue_wait_ms": self.max_queue_wait_seconds * 1000,
            "peak_reserved_blocks": self.peak_reserved_blocks,
            "final_reserved_blocks": self.reserved_blocks,
        }
        if self.config.policy == "prefix_aware_fifo":
            snapshot["schema_version"] = 2
            snapshot["observe_prefix_cache"] = self.config.observe_prefix_cache
        elif self.config.policy == "slack_aware_prefix_fifo":
            snapshot.update(
                {
                    "schema_version": 3,
                    "observe_prefix_cache": self.config.observe_prefix_cache,
                    "eta_prefill_seconds": self.config.eta_prefill_seconds,
                    "eta_decode_seconds_per_token": (
                        self.config.eta_decode_seconds_per_token
                    ),
                    "eta_safety_margin_seconds": (
                        self.config.eta_safety_margin_seconds
                    ),
                }
            )
        return snapshot


@dataclass(frozen=True, slots=True)
class OpenLoopResult:
    status: str
    records: list[RequestRecord]
    step_events: list[dict[str, Any]]
    summary: dict[str, Any]
    error: dict[str, Any] | None
    phase_timings: list[dict[str, Any]] = field(default_factory=list)
    admission: dict[str, Any] | None = None
    admission_events: list[dict[str, Any]] = field(default_factory=list)
    cache_events: list[dict[str, Any]] = field(default_factory=list)
    cache_states: list[dict[str, Any]] = field(default_factory=list)
    slack_events: list[dict[str, Any]] = field(default_factory=list)


class OpenLoopEngine(Protocol):
    def add_request(
        self,
        prompt: list[int],
        sampling_params: SamplingParams,
    ) -> int: ...

    def is_finished(self) -> bool: ...

    def step_with_events(self) -> EngineStepResult: ...


def load_trace(path: Path) -> list[RequestSpec]:
    with path.open(encoding="utf-8") as file:
        document = json.load(file)
    trace_keys = {"schema_version", "description", "time_unit", "requests"}
    if not isinstance(document, dict) or set(document) != trace_keys:
        raise ValueError(
            f"Trace must contain exactly {sorted(trace_keys)}"
        )
    schema_version = document["schema_version"]
    if schema_version not in {1, 2}:
        raise ValueError("Trace schema_version must be 1 or 2")
    if document["time_unit"] != "seconds":
        raise ValueError("Trace time_unit must be seconds")
    if (
        not isinstance(document["description"], str)
        or not document["description"]
    ):
        raise ValueError("Trace description must be a non-empty string")
    if not isinstance(document["requests"], list):
        raise ValueError("Trace requests must be a list")

    specs = []
    request_ids = set()
    for index, value in enumerate(document["requests"]):
        expected_keys = (
            TRACE_V1_KEYS if schema_version == 1 else TRACE_V2_KEYS
        )
        if not isinstance(value, dict) or set(value) != expected_keys:
            raise ValueError(
                f"Trace request {index} must contain exactly "
                f"{sorted(expected_keys)}"
            )
        spec = RequestSpec(**value)
        if not spec.request_id or spec.request_id in request_ids:
            raise ValueError(
                f"Trace request_id must be unique and non-empty: "
                f"{spec.request_id!r}"
            )
        if spec.request_class not in {"interactive", "long"}:
            raise ValueError(
                f"Unsupported request_class: {spec.request_class}"
            )
        if (
            not isinstance(spec.arrival_time, (int, float))
            or isinstance(spec.arrival_time, bool)
            or not isfinite(spec.arrival_time)
            or spec.arrival_time < 0
        ):
            raise ValueError("arrival_time must be a finite non-negative number")
        for field_name in ("prompt_length", "max_output_tokens"):
            field_value = getattr(spec, field_name)
            if (
                not isinstance(field_value, int)
                or isinstance(field_value, bool)
                or field_value <= 0
            ):
                raise ValueError(f"{field_name} must be a positive integer")
        if not isinstance(spec.seed, int) or isinstance(spec.seed, bool):
            raise ValueError("seed must be an integer")
        if schema_version == 2:
            if spec.prefix_group is None:
                if spec.shared_prefix_length != 0:
                    raise ValueError(
                        "shared_prefix_length must be 0 without prefix_group"
                    )
            else:
                if (
                    not isinstance(spec.prefix_group, str)
                    or not spec.prefix_group
                ):
                    raise ValueError("prefix_group must be a non-empty string")
                if (
                    not isinstance(spec.shared_prefix_length, int)
                    or isinstance(spec.shared_prefix_length, bool)
                    or spec.shared_prefix_length <= 0
                    or spec.shared_prefix_length >= spec.prompt_length
                ):
                    raise ValueError(
                        "shared_prefix_length must be positive and shorter "
                        "than prompt_length"
                    )
        request_ids.add(spec.request_id)
        specs.append(spec)

    return sorted(specs, key=lambda spec: (spec.arrival_time, spec.request_id))


def prepare_requests(
    specs: list[RequestSpec],
    *,
    token_id_upper_bound: int,
) -> list[PreparedRequest]:
    if token_id_upper_bound <= 1:
        raise ValueError("token_id_upper_bound must be greater than 1")
    prepared = []
    for spec in specs:
        if spec.prefix_group is None:
            generator = random.Random(spec.seed)
            prompt_token_ids = [
                generator.randrange(1, token_id_upper_bound)
                for _ in range(spec.prompt_length)
            ]
        else:
            digest = hashlib.sha256(
                spec.prefix_group.encode("utf-8")
            ).digest()
            prefix_generator = random.Random(
                int.from_bytes(digest[:8], "big")
            )
            suffix_generator = random.Random(spec.seed)
            prompt_token_ids = [
                prefix_generator.randrange(1, token_id_upper_bound)
                for _ in range(spec.shared_prefix_length)
            ] + [
                suffix_generator.randrange(1, token_id_upper_bound)
                for _ in range(
                    spec.prompt_length - spec.shared_prefix_length
                )
            ]
        prepared.append(
            PreparedRequest(
                spec=spec,
                prompt_token_ids=prompt_token_ids,
            )
        )
    return prepared


def run_open_loop(
    engine: OpenLoopEngine,
    requests: list[PreparedRequest],
    *,
    temperature: float,
    ignore_eos: bool,
    measurement_start: float,
    measurement_end: float,
    ttft_slo_ms: float,
    itl_slo_ms: float,
    require_itl: bool = True,
    max_run_seconds: float,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
    synchronize: Callable[[], None] = lambda: None,
    profiler: Any | None = None,
    admission: AdmissionConfig | None = None,
) -> OpenLoopResult:
    if not requests:
        raise ValueError("Open-loop benchmark requires at least one request")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if max_run_seconds <= 0:
        raise ValueError("max_run_seconds must be positive")
    if max(request.spec.arrival_time for request in requests) > max_run_seconds:
        raise ValueError("All trace arrivals must occur before max_run_seconds")

    ordered_requests = sorted(
        requests,
        key=lambda request: (
            request.spec.arrival_time,
            request.spec.request_id,
        ),
    )
    pending = deque(ordered_requests)
    admission_queue: deque[PreparedRequest] = deque()
    controller = KVAdmissionController(admission or AdmissionConfig())
    records = [
        RequestRecord(
            request_id=request.spec.request_id,
            request_class=request.spec.request_class,
            arrival_at=request.spec.arrival_time,
        )
        for request in ordered_requests
    ]
    records_by_id = {record.request_id: record for record in records}
    request_id_by_seq_id: dict[int, str] = {}
    step_events: list[dict[str, Any]] = []
    phase_timings: list[dict[str, Any]] = []
    admission_events: list[dict[str, Any]] = []
    cache_events: list[dict[str, Any]] = []
    cache_states: list[dict[str, Any]] = []
    slack_events: list[dict[str, Any]] = []
    preview_by_request: dict[str, PrefixCachePreview] = {}
    cache_state_by_request: dict[str, int] = {}
    reservation_runtime: dict[str, ReservationRuntime] = {}
    slack_estimator = (
        SlackEstimator(controller.config)
        if controller.config.policy == "slack_aware_prefix_fifo"
        else None
    )

    def preview_prefix_cache(
        request: PreparedRequest,
    ) -> PrefixCachePreview | None:
        if not controller.config.observe_prefix_cache:
            return None
        preview_fn = getattr(engine, "preview_prefix_cache", None)
        if preview_fn is None:
            raise RuntimeError("Engine does not expose prefix cache preview")
        preview = preview_fn(
            request.prompt_token_ids,
            request.spec.max_output_tokens,
        )
        if not isinstance(preview, PrefixCachePreview):
            raise RuntimeError("Engine returned invalid prefix cache preview")
        if (
            len(preview.matched_block_ids)
            != preview.matched_prefix_blocks
            or len(preview.active_block_ids)
            != preview.active_shared_blocks
            or len(preview.inactive_block_ids)
            != preview.inactive_cached_blocks
            or set(preview.active_block_ids)
            | set(preview.inactive_block_ids)
            != set(preview.matched_block_ids)
            or set(preview.active_block_ids)
            & set(preview.inactive_block_ids)
        ):
            raise RuntimeError("Engine returned inconsistent prefix cache IDs")
        return preview

    def capture_cache_state(observed_at: float) -> int:
        snapshot_fn = getattr(engine, "cache_state_snapshot", None)
        if snapshot_fn is None:
            raise RuntimeError("Engine does not expose cache state snapshot")
        blocks = snapshot_fn()
        if not isinstance(blocks, list):
            raise RuntimeError("Engine returned invalid cache state snapshot")
        state_index = len(cache_states)
        cache_states.append(
            {
                "schema_version": 1,
                "state_index": state_index,
                "observed_at": observed_at,
                "blocks": blocks,
            }
        )
        return state_index

    def record_cache_event(
        *,
        action: str,
        request: PreparedRequest,
        observed_at: float,
        preview: PrefixCachePreview,
        cache_state_index: int,
        reason: str | None,
    ) -> None:
        cache_events.append(
            {
                "schema_version": 2,
                "event_index": len(cache_events),
                "action": action,
                "request_id": request.spec.request_id,
                "observed_at": observed_at,
                "full_required_blocks": preview.full_required_blocks,
                "matched_prefix_blocks": preview.matched_prefix_blocks,
                "active_shared_blocks": preview.active_shared_blocks,
                "inactive_cached_blocks": preview.inactive_cached_blocks,
                "matched_prefix_tokens": preview.matched_prefix_tokens,
                "incremental_reservation_blocks": (
                    preview.incremental_reservation_blocks
                ),
                "matched_block_ids": list(preview.matched_block_ids),
                "active_block_ids": list(preview.active_block_ids),
                "inactive_block_ids": list(preview.inactive_block_ids),
                "cache_state_index": cache_state_index,
                "reserved_blocks_after": controller.reserved_blocks,
                "reason": reason,
            }
        )

    def record_admission_event(
        *,
        action: str,
        request: PreparedRequest,
        observed_at: float,
        queue_wait_seconds: float | None,
        preview: PrefixCachePreview | None,
        reason: str | None,
    ) -> None:
        event = {
            "schema_version": 1,
            "event_index": len(admission_events),
            "action": action,
            "request_id": request.spec.request_id,
            "arrival_at": request.spec.arrival_time,
            "observed_at": observed_at,
            "queue_wait_ms": (
                None
                if queue_wait_seconds is None
                else queue_wait_seconds * 1000
            ),
            "required_blocks": controller.required_blocks(request),
            "reserved_blocks_after": controller.reserved_blocks,
            "reason": reason,
        }
        if controller.config.policy in {
            "prefix_aware_fifo",
            "slack_aware_prefix_fifo",
        }:
            event["schema_version"] = (
                3 if controller.config.policy == "slack_aware_prefix_fifo" else 2
            )
            event["reservation_blocks"] = controller.reservation_blocks(
                request,
                preview,
            )
        admission_events.append(event)

    def record_slack_event(
        *,
        request: PreparedRequest,
        observed_at: float,
        preview: PrefixCachePreview,
        cache_state_index: int,
        admission_event_index: int,
        cache_event_index: int,
        predicted_free_at: float,
        predicted_releasable_blocks: int,
    ) -> None:
        assert slack_estimator is not None
        required = controller.reservation_blocks(request, preview)
        shortfall = controller.reserved_blocks + required - controller.config.total_kv_blocks
        active = []
        for runtime in reservation_runtime.values():
            active.append(
                {
                    "request_id": runtime.request_id,
                    "reservation_blocks": runtime.reservation_blocks,
                    "admitted_at": runtime.admitted_at,
                    "max_output_tokens": runtime.max_output_tokens,
                    "generated_tokens": runtime.generated_tokens,
                    "last_token_at": runtime.last_token_at,
                    "predicted_release_at": slack_estimator.predict_release_at(runtime),
                }
            )
        deadline = request.spec.arrival_time + controller.config.max_queue_wait_seconds
        slack_events.append(
            {
                "schema_version": 1,
                "decision_index": len(slack_events),
                "request_id": request.spec.request_id,
                "observed_at": observed_at,
                "cache_state_index": cache_state_index,
                "admission_event_index": admission_event_index,
                "cache_event_index": cache_event_index,
                "full_required_blocks": controller.required_blocks(request),
                "incremental_reservation_blocks": required,
                "reserved_blocks_before": controller.reserved_blocks,
                "capacity_shortfall_blocks": shortfall,
                "deadline_at": deadline,
                "predicted_free_at": predicted_free_at,
                "predicted_releasable_blocks": predicted_releasable_blocks,
                "slack_ms": (deadline - predicted_free_at) * 1000,
                "active_reservations": active,
                "reason": "predicted_deadline_miss",
            }
        )

    request_by_id = {
        request.spec.request_id: request for request in ordered_requests
    }
    benchmark_started = clock()
    step_index = 0

    error = None
    try:
        while pending or admission_queue or not engine.is_finished():
            elapsed = clock() - benchmark_started
            if elapsed >= max_run_seconds:
                break

            while pending and pending[0].spec.arrival_time <= elapsed:
                admission_queue.append(pending.popleft())

            while admission_queue:
                request = admission_queue[0]
                queue_wait = elapsed - request.spec.arrival_time
                preview = preview_prefix_cache(request)
                if (
                    controller.config.policy
                    in {
                        "kv_aware_fifo",
                        "prefix_aware_fifo",
                        "slack_aware_prefix_fifo",
                    }
                    and queue_wait >= controller.config.max_queue_wait_seconds
                ):
                    admission_queue.popleft()
                    controller.reject(queue_wait)
                    record_admission_event(
                        action="rejected",
                        request=request,
                        observed_at=elapsed,
                        queue_wait_seconds=queue_wait,
                        preview=preview,
                        reason="kv_reservation_timeout",
                    )
                    if preview is not None:
                        record_cache_event(
                            action="rejected",
                            request=request,
                            observed_at=elapsed,
                            preview=preview,
                            cache_state_index=capture_cache_state(elapsed),
                            reason="kv_reservation_timeout",
                        )
                    records_by_id[request.spec.request_id].mark_terminal(
                        TerminalState.REJECTED,
                        elapsed,
                        reason="kv_reservation_timeout",
                    )
                    continue
                can_admit = controller.can_admit(request, preview)
                if (
                    not can_admit
                    and slack_estimator is not None
                    and preview is not None
                ):
                    required = controller.reservation_blocks(request, preview)
                    assert controller.config.total_kv_blocks is not None
                    shortfall = (
                        controller.reserved_blocks + required
                        - controller.config.total_kv_blocks
                    )
                    prediction = slack_estimator.earliest_capacity_time(
                        list(reservation_runtime.values()),
                        shortfall_blocks=shortfall,
                    )
                    deadline = (
                        request.spec.arrival_time
                        + controller.config.max_queue_wait_seconds
                    )
                    if prediction is not None and prediction[0] > deadline:
                        admission_queue.popleft()
                        controller.reject(queue_wait)
                        admission_event_index = len(admission_events)
                        record_admission_event(
                            action="rejected",
                            request=request,
                            observed_at=elapsed,
                            queue_wait_seconds=queue_wait,
                            preview=preview,
                            reason="predicted_deadline_miss",
                        )
                        cache_event_index = len(cache_events)
                        cache_state_index = capture_cache_state(elapsed)
                        record_cache_event(
                            action="rejected",
                            request=request,
                            observed_at=elapsed,
                            preview=preview,
                            cache_state_index=cache_state_index,
                            reason="predicted_deadline_miss",
                        )
                        record_slack_event(
                            request=request,
                            observed_at=elapsed,
                            preview=preview,
                            cache_state_index=cache_state_index,
                            admission_event_index=admission_event_index,
                            cache_event_index=cache_event_index,
                            predicted_free_at=prediction[0],
                            predicted_releasable_blocks=prediction[1],
                        )
                        records_by_id[request.spec.request_id].mark_terminal(
                            TerminalState.REJECTED,
                            elapsed,
                            reason="predicted_deadline_miss",
                        )
                        continue
                if not can_admit:
                    break
                admission_queue.popleft()
                record = records_by_id[request.spec.request_id]
                controller.admit(request, queue_wait, preview)
                record.mark_admitted(elapsed)
                if slack_estimator is not None:
                    reservation_runtime[request.spec.request_id] = ReservationRuntime(
                        request_id=request.spec.request_id,
                        reservation_blocks=controller.reservation_blocks(request, preview),
                        admitted_at=elapsed,
                        max_output_tokens=request.spec.max_output_tokens,
                    )
                if preview is not None:
                    preview_by_request[request.spec.request_id] = preview
                    cache_state_by_request[request.spec.request_id] = (
                        capture_cache_state(elapsed)
                    )
                record_admission_event(
                    action="admitted",
                    request=request,
                    observed_at=elapsed,
                    queue_wait_seconds=queue_wait,
                    preview=preview,
                    reason=None,
                )
                if preview is not None:
                    record_cache_event(
                        action="admitted",
                            request=request,
                            observed_at=elapsed,
                            preview=preview,
                            cache_state_index=cache_state_by_request[
                                request.spec.request_id
                            ],
                            reason=None,
                    )
                try:
                    seq_id = engine.add_request(
                        request.prompt_token_ids,
                        SamplingParams(
                            temperature=temperature,
                            max_tokens=request.spec.max_output_tokens,
                            ignore_eos=ignore_eos,
                        ),
                    )
                except Exception:
                    controller.release(request.spec.request_id)
                    reservation_runtime.pop(request.spec.request_id, None)
                    record_admission_event(
                        action="released",
                        request=request,
                        observed_at=elapsed,
                        queue_wait_seconds=None,
                        preview=preview,
                        reason="engine_add_error",
                    )
                    if preview is not None:
                        record_cache_event(
                            action="released",
                            request=request,
                            observed_at=elapsed,
                            preview=preview,
                            cache_state_index=cache_state_by_request[
                                request.spec.request_id
                            ],
                            reason="engine_add_error",
                        )
                    raise
                if seq_id in request_id_by_seq_id:
                    controller.release(request.spec.request_id)
                    reservation_runtime.pop(request.spec.request_id, None)
                    record_admission_event(
                        action="released",
                        request=request,
                        observed_at=elapsed,
                        queue_wait_seconds=None,
                        preview=preview,
                        reason="duplicate_sequence_id",
                    )
                    if preview is not None:
                        record_cache_event(
                            action="released",
                            request=request,
                            observed_at=elapsed,
                            preview=preview,
                            cache_state_index=cache_state_by_request[
                                request.spec.request_id
                            ],
                            reason="duplicate_sequence_id",
                        )
                    raise RuntimeError(f"Duplicate internal sequence ID: {seq_id}")
                configure_request_slo = getattr(
                    engine,
                    "configure_request_slo",
                    None,
                )
                if configure_request_slo is not None:
                    configure_request_slo(
                        seq_id,
                        ttft_deadline_at=(
                            benchmark_started
                            + request.spec.arrival_time
                            + ttft_slo_ms / 1000.0
                        ),
                        itl_slo_s=itl_slo_ms / 1000.0,
                    )
                request_id_by_seq_id[seq_id] = request.spec.request_id

            if engine.is_finished():
                if pending:
                    wait_seconds = pending[0].spec.arrival_time - elapsed
                    if wait_seconds > 0:
                        sleep(wait_seconds)
                continue

            synchronize()
            step_started = clock() - benchmark_started
            result = engine.step_with_events()
            synchronize()
            step_finished = clock() - benchmark_started

            serialized_events = []
            for event in result.events:
                request_id = request_id_by_seq_id.get(event.seq_id)
                if request_id is None:
                    raise RuntimeError(
                        f"Step referenced unknown sequence ID: {event.seq_id}"
                    )
                record = records_by_id[request_id]
                record.record_schedule(step_started)
                if event.emitted_token_id is not None:
                    record.record_token(step_finished)
                    runtime = reservation_runtime.get(request_id)
                    if runtime is not None:
                        runtime.generated_tokens += 1
                        runtime.last_token_at = step_finished
                if event.finished:
                    record.mark_terminal(
                        TerminalState.FINISHED,
                        step_finished,
                    )
                    controller.release(request_id)
                    reservation_runtime.pop(request_id, None)
                    preview = preview_by_request.get(request_id)
                    record_admission_event(
                        action="released",
                        request=request_by_id[request_id],
                        observed_at=step_finished,
                        queue_wait_seconds=None,
                        preview=preview,
                        reason=None,
                    )
                    if preview is not None:
                        record_cache_event(
                            action="released",
                            request=request_by_id[request_id],
                            observed_at=step_finished,
                            preview=preview,
                            cache_state_index=cache_state_by_request[request_id],
                            reason=None,
                        )
                serialized_event = {
                        "request_id": request_id,
                        "seq_id": event.seq_id,
                        "phase": event.phase,
                        "num_scheduled_tokens": event.num_scheduled_tokens,
                        "emitted_token": event.emitted_token_id is not None,
                        "finished": event.finished,
                    }
                if event.prefill_kind is not None:
                    serialized_event.update(
                        {
                            "prefill_kind": event.prefill_kind,
                            "actual_recompute_tokens": (
                                event.actual_recompute_tokens
                            ),
                            "resumed": event.resumed,
                        }
                    )
                if event.scheduler_step_index is not None:
                    serialized_event.update(
                        {
                            "scheduler_step_index": (
                                event.scheduler_step_index
                            ),
                            "previous_progress_step": (
                                event.previous_progress_step
                            ),
                            "progress_gap_steps": (
                                event.progress_gap_steps
                            ),
                            "had_emitted_token_before": (
                                event.had_emitted_token_before
                            ),
                        }
                    )
                serialized_events.append(serialized_event)

            serialized_step = {
                    "step_index": step_index,
                    "started_at": step_started,
                    "finished_at": step_finished,
                    "duration_ms": (step_finished - step_started) * 1000,
                    "phase": result.phase,
                    "num_scheduled_tokens": result.num_scheduled_tokens,
                    "events": serialized_events,
                }
            if result.scheduler_event is not None:
                scheduler_event = asdict(result.scheduler_event)
                is_bounded_schema = (
                    scheduler_event["policy"]
                    == "recompute_aware_bounded"
                )
                serialized_step["schema_version"] = (
                    3 if is_bounded_schema else 2
                )
                if not is_bounded_schema:
                    for field in (
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
                    ):
                        scheduler_event.pop(field)
                    for event in serialized_events:
                        for field in (
                            "scheduler_step_index",
                            "previous_progress_step",
                            "progress_gap_steps",
                            "had_emitted_token_before",
                        ):
                            event.pop(field, None)
                for preemption in scheduler_event["preemptions"]:
                    preemption["request_id"] = request_id_by_seq_id.get(
                        preemption["seq_id"]
                    )
                    preemption["triggering_request_id"] = (
                        request_id_by_seq_id.get(
                            preemption["triggering_seq_id"]
                        )
                        if preemption["triggering_seq_id"] is not None
                        else None
                    )
                for cost in scheduler_event["resident_costs"]:
                    cost["request_id"] = request_id_by_seq_id.get(
                        cost["seq_id"]
                    )
                if is_bounded_schema:
                    for fairness in scheduler_event[
                        "active_progress_states"
                    ]:
                        fairness["request_id"] = (
                            request_id_by_seq_id.get(fairness["seq_id"])
                        )
                    for watch in scheduler_event["drain_slo_watch"]:
                        watch["request_id"] = request_id_by_seq_id.get(
                            watch["seq_id"]
                        )
                scheduler_event["selected_decode_request_ids"] = [
                    request_id_by_seq_id[seq_id]
                    for seq_id in scheduler_event["selected_decode_ids"]
                ]
                scheduler_event["selected_prefill_request_ids"] = [
                    request_id_by_seq_id[seq_id]
                    for seq_id in scheduler_event["selected_prefill_ids"]
                ]
                serialized_step["scheduler"] = scheduler_event
            step_events.append(serialized_step)
            if result.phase_timings_ms is not None:
                phase_timings.append(
                    {
                        "step_index": step_index,
                        **result.phase_timings_ms,
                    }
                )
            if profiler is not None:
                profiler.step()
            step_index += 1
    except Exception as exception:
        error = {
            "type": type(exception).__name__,
            "message": str(exception),
            "is_cuda_oom": isinstance(
                exception,
                torch.cuda.OutOfMemoryError,
            ),
        }

    terminal_time = min(
        clock() - benchmark_started,
        max_run_seconds,
    )
    cleanup_reason = (
        "error_cleanup"
        if error is not None
        else "timeout_cleanup"
    )
    if error is not None or any(
        record.terminal_state is None for record in records
    ):
        for request_id in tuple(controller.reserved_by_request):
            controller.release(request_id)
            reservation_runtime.pop(request_id, None)
            preview = preview_by_request.get(request_id)
            record_admission_event(
                action="released",
                request=request_by_id[request_id],
                observed_at=terminal_time,
                queue_wait_seconds=None,
                preview=preview,
                reason=cleanup_reason,
            )
            if preview is not None:
                record_cache_event(
                    action="released",
                    request=request_by_id[request_id],
                    observed_at=terminal_time,
                    preview=preview,
                    cache_state_index=cache_state_by_request[request_id],
                    reason=cleanup_reason,
                )
    for record in records:
        if record.terminal_state is None:
            if error is not None:
                state = (
                    TerminalState.FAILED
                    if record.admitted_at is not None
                    else TerminalState.CANCELLED
                )
                reason = f"{error['type']}: {error['message']}"
            else:
                state = TerminalState.UNFINISHED
                reason = "max_run_seconds reached"
            record.mark_terminal(
                state,
                max(terminal_time, record.arrival_at),
                reason=reason,
            )

    summary = summarize_requests(
        records,
        measurement_start=measurement_start,
        measurement_end=measurement_end,
        ttft_slo_ms=ttft_slo_ms,
        itl_slo_ms=itl_slo_ms,
        require_itl=require_itl,
    )
    summary["runtime"] = {
        "elapsed_seconds": clock() - benchmark_started,
        "steps": len(step_events),
        "timed_out": any(
            record.terminal_state is TerminalState.UNFINISHED
            for record in records
        ),
    }
    status = (
        "failed"
        if error is not None
        or any(
            record.terminal_state is TerminalState.UNFINISHED
            for record in records
        )
        else "passed"
    )
    return OpenLoopResult(
        status=status,
        records=records,
        step_events=step_events,
        summary=summary,
        error=error,
        phase_timings=phase_timings,
        admission=controller.snapshot(records),
        admission_events=admission_events,
        cache_events=cache_events,
        cache_states=cache_states,
        slack_events=slack_events,
    )
