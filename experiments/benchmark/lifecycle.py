from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import ceil, floor
from typing import Any


class TerminalState(str, Enum):
    FINISHED = "Finished"
    REJECTED = "Rejected"
    FAILED = "Failed"
    CANCELLED = "Cancelled"
    UNFINISHED = "Unfinished"


@dataclass(slots=True)
class RequestRecord:
    request_id: str
    request_class: str
    arrival_at: float
    admitted_at: float | None = None
    first_scheduled_at: float | None = None
    last_scheduled_at: float | None = None
    max_progress_gap_s: float | None = None
    token_timestamps: list[float] = field(default_factory=list)
    terminal_state: TerminalState | None = None
    terminal_at: float | None = None
    terminal_reason: str | None = None

    def mark_admitted(self, timestamp: float) -> None:
        if self.admitted_at is not None:
            raise RuntimeError(f"Request {self.request_id} was already admitted")
        self._validate_timestamp(timestamp)
        self.admitted_at = timestamp

    def record_schedule(self, timestamp: float) -> None:
        self._validate_timestamp(timestamp)
        if self.terminal_state is not None:
            raise RuntimeError(
                f"Cannot schedule terminal request {self.request_id}"
            )
        if self.first_scheduled_at is None:
            self.first_scheduled_at = timestamp
        if self.last_scheduled_at is not None:
            progress_gap = timestamp - self.last_scheduled_at
            self.max_progress_gap_s = max(
                self.max_progress_gap_s or 0.0,
                progress_gap,
            )
        self.last_scheduled_at = timestamp

    def record_token(self, timestamp: float) -> None:
        self._validate_timestamp(timestamp)
        if self.terminal_state is not None:
            raise RuntimeError(
                f"Cannot emit a token for terminal request {self.request_id}"
            )
        if self.token_timestamps and timestamp < self.token_timestamps[-1]:
            raise ValueError("Token timestamps must be monotonic")
        self.token_timestamps.append(timestamp)

    def mark_terminal(
        self,
        state: TerminalState,
        timestamp: float,
        *,
        reason: str | None = None,
    ) -> None:
        if self.terminal_state is not None:
            raise RuntimeError(
                f"Request {self.request_id} already ended as "
                f"{self.terminal_state.value}"
            )
        self._validate_timestamp(timestamp)
        if state is TerminalState.FINISHED and not self.token_timestamps:
            raise ValueError("Finished requests must have emitted at least one token")
        self.terminal_state = state
        self.terminal_at = timestamp
        self.terminal_reason = reason

    @property
    def ttft_s(self) -> float | None:
        if not self.token_timestamps:
            return None
        return self.token_timestamps[0] - self.arrival_at

    @property
    def e2e_s(self) -> float | None:
        if self.terminal_state is not TerminalState.FINISHED:
            return None
        assert self.terminal_at is not None
        return self.terminal_at - self.arrival_at

    @property
    def itl_samples_s(self) -> list[float]:
        return [
            current - previous
            for previous, current in zip(
                self.token_timestamps,
                self.token_timestamps[1:],
                strict=False,
            )
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "request_class": self.request_class,
            "arrival_at": self.arrival_at,
            "admitted_at": self.admitted_at,
            "first_scheduled_at": self.first_scheduled_at,
            "token_timestamps": self.token_timestamps,
            "terminal_state": (
                self.terminal_state.value if self.terminal_state else None
            ),
            "terminal_at": self.terminal_at,
            "terminal_reason": self.terminal_reason,
            "ttft_ms": self.ttft_s * 1000 if self.ttft_s is not None else None,
            "e2e_ms": self.e2e_s * 1000 if self.e2e_s is not None else None,
            "itl_ms": [value * 1000 for value in self.itl_samples_s],
            "max_progress_gap_ms": (
                self.max_progress_gap_s * 1000
                if self.max_progress_gap_s is not None
                else None
            ),
        }

    def _validate_timestamp(self, timestamp: float) -> None:
        if timestamp < self.arrival_at:
            raise ValueError(
                f"Timestamp {timestamp} precedes arrival {self.arrival_at}"
            )


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * quantile
    lower = floor(rank)
    upper = ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def distribution_ms(values_s: list[float]) -> dict[str, float | int | None]:
    values_ms = [value * 1000 for value in values_s]
    return {
        "count": len(values_ms),
        "p50": percentile(values_ms, 0.50),
        "p95": percentile(values_ms, 0.95),
        "p99": percentile(values_ms, 0.99),
        "max": max(values_ms) if values_ms else None,
    }


def summarize_requests(
    records: list[RequestRecord],
    *,
    measurement_start: float,
    measurement_end: float,
    ttft_slo_ms: float,
    itl_slo_ms: float,
    require_itl: bool = True,
) -> dict[str, Any]:
    duration = measurement_end - measurement_start
    if duration <= 0:
        raise ValueError("measurement_end must be greater than measurement_start")
    if any(record.terminal_state is None for record in records):
        raise ValueError("Every submitted request must have a terminal state")

    terminal_counts: dict[str, int | bool] = {
        state.value: sum(
            record.terminal_state is state for record in records
        )
        for state in TerminalState
    }
    terminal_counts["submitted"] = len(records)
    terminal_total = sum(
        int(terminal_counts[state.value]) for state in TerminalState
    )
    terminal_counts["reconciled"] = terminal_total == len(records)

    eligible = [
        record
        for record in records
        if measurement_start <= record.arrival_at < measurement_end
    ]
    interactive = [
        record for record in eligible if record.request_class == "interactive"
    ]
    long_requests = [
        record for record in eligible if record.request_class == "long"
    ]
    interactive_ttft = [
        record.ttft_s
        for record in interactive
        if record.ttft_s is not None
    ]
    interactive_itl = [
        sample
        for record in interactive
        for sample in record.itl_samples_s
    ]

    slo_successes = 0
    for record in interactive:
        if record.terminal_state is not TerminalState.FINISHED:
            continue
        ttft_ms = record.ttft_s * 1000 if record.ttft_s is not None else None
        request_p99_itl_s = percentile(record.itl_samples_s, 0.99)
        request_p99_itl_ms = (
            request_p99_itl_s * 1000
            if request_p99_itl_s is not None
            else None
        )
        if (
            ttft_ms is not None
            and ttft_ms <= ttft_slo_ms
            and (
                (request_p99_itl_ms is not None and request_p99_itl_ms <= itl_slo_ms)
                or (request_p99_itl_ms is None and not require_itl)
            )
        ):
            slo_successes += 1

    finished_long = sum(
        record.terminal_state is TerminalState.FINISHED
        for record in long_requests
    )
    total_output_tokens = sum(len(record.token_timestamps) for record in eligible)
    long_output_tokens = sum(
        len(record.token_timestamps) for record in long_requests
    )

    return {
        "measurement": {
            "start": measurement_start,
            "end": measurement_end,
            "duration": duration,
            "eligible_requests": len(eligible),
        },
        "terminal_counts": terminal_counts,
        "interactive": {
            "submitted": len(interactive),
            "ttft_ms": distribution_ms(interactive_ttft),
            "itl_ms": distribution_ms(interactive_itl),
            "slo_successes": slo_successes,
            "slo_goodput_rps": slo_successes / duration,
        },
        "long": {
            "submitted": len(long_requests),
            "finished": finished_long,
            "request_goodput_rps": finished_long / duration,
            "token_goodput_tps": long_output_tokens / duration,
        },
        "output_throughput_tps": total_output_tokens / duration,
    }
