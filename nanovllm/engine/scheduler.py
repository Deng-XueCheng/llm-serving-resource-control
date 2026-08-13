from collections import deque
from dataclasses import dataclass
from time import perf_counter
from typing import Callable

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


@dataclass(frozen=True, slots=True)
class ResidentSequenceCost:
    seq_id: int
    resident_kv_blocks: int
    resident_kv_tokens: int
    logical_context_tokens: int
    remaining_decode_tokens: int
    releasable_blocks: int
    estimated_recompute_tokens: int
    estimated_steps_to_release: int
    can_advance_without_free_block: bool

    @property
    def release_efficiency(self) -> float:
        return self.releasable_blocks / max(
            self.estimated_steps_to_release,
            1,
        )

    @property
    def preemption_harm(self) -> float:
        if self.releasable_blocks == 0:
            return float("inf")
        return self.estimated_recompute_tokens / self.releasable_blocks


@dataclass(frozen=True, slots=True)
class PreemptionEvent:
    seq_id: int
    reason: str
    triggering_seq_id: int | None
    logical_context_tokens: int
    resident_kv_tokens: int
    resident_blocks: int
    releasable_blocks: int
    estimated_recompute_tokens: int
    remaining_decode_tokens: int
    scheduler_state: str


@dataclass(frozen=True, slots=True)
class RequestFairnessState:
    seq_id: int
    location: str
    pending_recompute: bool
    had_emitted_token: bool
    waiting_age_steps: int
    last_progress_step: int | None
    steps_since_last_progress: int | None
    slo_deadline_at: float | None
    slo_deadline_kind: str | None


@dataclass(frozen=True, slots=True)
class DrainSloWatchState:
    seq_id: int
    entry_progress_step: int | None
    deadline_at: float
    deadline_kind: str


@dataclass(frozen=True, slots=True)
class DrainExitDecision:
    reason: str
    slo_guard_seq_id: int | None = None
    slo_guard_entry_progress_step: int | None = None
    slo_guard_deadline_at: float | None = None
    slo_guard_triggered_at: float | None = None


@dataclass(frozen=True, slots=True)
class ScheduledSequence:
    sequence: Sequence
    phase: str
    num_scheduled_tokens: int
    requires_sampling: bool


@dataclass(frozen=True, slots=True)
class MixedScheduleBatch:
    items: tuple[ScheduledSequence, ...]

    @property
    def num_scheduled_tokens(self) -> int:
        return sum(item.num_scheduled_tokens for item in self.items)


@dataclass(frozen=True, slots=True)
class SchedulerStepEvent:
    policy: str
    state: str
    mode: str
    kv_total_blocks: int
    kv_used_blocks_before: int
    kv_free_blocks_before: int
    kv_used_blocks_after: int
    kv_free_blocks_after: int
    running_ids_before: tuple[int, ...]
    waiting_ids_before: tuple[int, ...]
    running_ids_after: tuple[int, ...]
    waiting_ids_after: tuple[int, ...]
    selected_decode_ids: tuple[int, ...]
    selected_prefill_ids: tuple[int, ...]
    oldest_waiting_age: int
    resident_costs: tuple[ResidentSequenceCost, ...]
    preemptions: tuple[PreemptionEvent, ...]
    scheduler_step_index: int
    active_progress_states: tuple[RequestFairnessState, ...]
    drain_slo_watch: tuple[DrainSloWatchState, ...]
    drain_episode_id: int | None
    drain_episode_step: int
    drain_tokens: int
    drain_episode_started_at: float | None
    drain_exit_reason: str | None
    fairness_trigger_reason: str | None
    resource_pressure: bool | None
    slo_guard_seq_id: int | None
    slo_guard_entry_progress_step: int | None
    slo_guard_deadline_at: float | None
    slo_guard_triggered_at: float | None
    planned_decode_budget: int | None = None
    planned_prefill_budget: int | None = None
    actual_decode_tokens: int = 0
    actual_prefill_tokens: int = 0
    unused_budget: int = 0
    min_ttft_slack: float | None = None
    min_itl_slack: float | None = None
    mixed_step_ratio: float = 0.0


class Scheduler:

    def __init__(
        self,
        config: Config,
        *,
        clock: Callable[[], float] = perf_counter,
    ):
        self._clock = clock
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.policy = getattr(config, "scheduler_policy", "prefill_first")
        self.decode_token_budget = getattr(config, "decode_token_budget", 0)
        self.decode_step_guard = getattr(config, "decode_step_guard", 0)
        self.incremental_kv_allocation = getattr(
            config,
            "incremental_kv_allocation",
            False,
        )
        self.decode_tokens_since_prefill = 0
        self.decode_steps_since_waiting = 0
        self.preemption_count = 0
        self.actual_recompute_tokens = 0
        self.resume_count = 0
        self._mixed_step_count = 0
        self._mixed_total_step_count = 0
        self.last_step_event: SchedulerStepEvent | None = None
        self._step_preemptions: list[PreemptionEvent] = []
        self._step_mode = "uninitialized"
        self._step_state = "static"
        self._step_running_ids_before: tuple[int, ...] = ()
        self._step_waiting_ids_before: tuple[int, ...] = ()
        self._step_kv_used_before = 0
        self._step_kv_free_before = 0
        if (
            not isinstance(self.policy, str)
            or self.policy
            not in {
                "prefill_first",
                "decode_first_budgeted",
                "pressure_aware_decode",
                "chunked_prefill_budgeted",
                "recompute_aware",
                "recompute_aware_bounded",
                "mixed_token_budget",
                "mixed_slo_budget",
            }
        ):
            raise ValueError(f"Unknown scheduler policy: {self.policy}")
        if self.policy in {
            "decode_first_budgeted",
            "pressure_aware_decode",
            "chunked_prefill_budgeted",
            "recompute_aware",
            "recompute_aware_bounded",
            "mixed_token_budget",
            "mixed_slo_budget",
        } and (
            not isinstance(self.decode_token_budget, int)
            or isinstance(self.decode_token_budget, bool)
            or self.decode_token_budget <= 0
        ):
            raise ValueError("decode_token_budget must be a positive integer")
        if self.policy == "prefill_first" and (
            not isinstance(self.decode_token_budget, int)
            or isinstance(self.decode_token_budget, bool)
            or self.decode_token_budget != 0
        ):
            raise ValueError("decode_token_budget must be 0 for prefill_first")
        if (
            not isinstance(self.decode_step_guard, int)
            or isinstance(self.decode_step_guard, bool)
            or self.decode_step_guard < 0
        ):
            raise ValueError("decode_step_guard must be a non-negative integer")
        if self.policy == "prefill_first" and self.decode_step_guard != 0:
            raise ValueError(
                "decode_step_guard must be 0 for prefill_first"
            )
        if self.policy in {
            "pressure_aware_decode",
            "chunked_prefill_budgeted",
            "recompute_aware",
            "recompute_aware_bounded",
        }:
            self._initialize_pressure_aware_config(config)
        if self.policy == "mixed_slo_budget":
            self.mixed_min_prefill_tokens = config.mixed_min_prefill_tokens
            self.mixed_waiting_age_threshold = config.mixed_waiting_age_threshold
            self.mixed_slack_threshold_s = config.mixed_slack_threshold_ms / 1000
            self.ttft_slo_s = config.ttft_slo_ms / 1000
            self.itl_slo_s = config.itl_slo_ms / 1000
            self._waiting_age_steps = {}
        self._scheduler_step_index = -1
        self._step_active_progress_states: tuple[
            RequestFairnessState, ...
        ] = ()
        self._step_drain_slo_watch: tuple[DrainSloWatchState, ...] = ()
        self._step_drain_episode_id: int | None = None
        self._step_drain_episode_step = 0
        self._step_drain_tokens = 0
        self._step_drain_episode_started_at: float | None = None
        self._step_drain_exit_reason: str | None = None
        self._step_fairness_trigger_reason: str | None = None
        self._step_resource_pressure: bool | None = None
        self._step_slo_guard_seq_id: int | None = None
        self._step_slo_guard_entry_progress_step: int | None = None
        self._step_slo_guard_deadline_at: float | None = None
        self._step_slo_guard_triggered_at: float | None = None
        if self.policy == "recompute_aware_bounded":
            self._initialize_bounded_drain_config(config)
        if self.policy == "chunked_prefill_budgeted":
            if not hasattr(config, "prefill_chunk_token_budget"):
                raise ValueError(
                    "chunked_prefill_budgeted requires config field: "
                    "prefill_chunk_token_budget"
                )
            self.prefill_chunk_token_budget = (
                config.prefill_chunk_token_budget
            )
            self._last_effective_prefill_token_budget = (
                self.max_num_batched_tokens
            )
            if (
                not isinstance(self.prefill_chunk_token_budget, int)
                or isinstance(self.prefill_chunk_token_budget, bool)
                or self.prefill_chunk_token_budget <= 0
                or self.prefill_chunk_token_budget
                > self.max_num_batched_tokens
            ):
                raise ValueError(
                    "prefill_chunk_token_budget must be a positive integer "
                    "no larger than max_num_batched_tokens"
                )

    def _initialize_pressure_aware_config(self, config: Config) -> None:
        required_fields = (
            "pressure_decode_token_budget",
            "pressure_decode_step_guard",
            "pressure_high_utilization",
            "pressure_critical_utilization",
            "pressure_preemption_window",
            "pressure_preemption_threshold",
            "pressure_hysteresis_steps",
            "pressure_waiting_age_threshold",
        )
        missing = [
            field for field in required_fields if not hasattr(config, field)
        ]
        if missing:
            raise ValueError(
                "pressure_aware_decode requires config fields: "
                + ", ".join(missing)
            )
        self.pressure_decode_token_budget = config.pressure_decode_token_budget
        self.pressure_decode_step_guard = config.pressure_decode_step_guard
        self.pressure_high_utilization = config.pressure_high_utilization
        self.pressure_critical_utilization = config.pressure_critical_utilization
        self.pressure_preemption_window = config.pressure_preemption_window
        self.pressure_preemption_threshold = config.pressure_preemption_threshold
        self.pressure_hysteresis_steps = config.pressure_hysteresis_steps
        self.pressure_waiting_age_threshold = config.pressure_waiting_age_threshold

        if (
            not isinstance(self.pressure_decode_token_budget, int)
            or isinstance(self.pressure_decode_token_budget, bool)
            or self.pressure_decode_token_budget <= 0
            or self.pressure_decode_token_budget > self.decode_token_budget
        ):
            raise ValueError(
                "pressure_decode_token_budget must be a positive integer "
                "no larger than decode_token_budget"
            )
        if (
            not isinstance(self.pressure_decode_step_guard, int)
            or isinstance(self.pressure_decode_step_guard, bool)
            or self.pressure_decode_step_guard < 0
        ):
            raise ValueError(
                "pressure_decode_step_guard must be non-negative"
            )
        if (
            not isinstance(self.pressure_high_utilization, (int, float))
            or isinstance(self.pressure_high_utilization, bool)
            or not 0 < self.pressure_high_utilization < 1
        ):
            raise ValueError("pressure_high_utilization must be in (0, 1)")
        if (
            not isinstance(self.pressure_critical_utilization, (int, float))
            or isinstance(self.pressure_critical_utilization, bool)
            or not 0 < self.pressure_critical_utilization <= 1
        ):
            raise ValueError(
                "pressure_critical_utilization must be in (0, 1]"
            )
        if self.pressure_high_utilization > self.pressure_critical_utilization:
            raise ValueError(
                "pressure_critical_utilization must be greater than or equal "
                "to pressure_high_utilization"
            )
        for name, value in (
            ("pressure_preemption_window", self.pressure_preemption_window),
            ("pressure_preemption_threshold", self.pressure_preemption_threshold),
            ("pressure_hysteresis_steps", self.pressure_hysteresis_steps),
            (
                "pressure_waiting_age_threshold",
                self.pressure_waiting_age_threshold,
            ),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")

        self._pressure_state = "normal"
        self._pressure_clean_steps = 0
        self._recent_preemption_deltas: deque[int] = deque(
            maxlen=self.pressure_preemption_window
        )
        self._waiting_age_steps: dict[int, int] = {}
        self._pressure_decision_count = 0
        self._pressure_last_decision: dict | None = None

    def _initialize_bounded_drain_config(self, config: Config) -> None:
        required = (
            "max_drain_steps",
            "waiting_age_limit",
            "ttft_slo_ms",
            "itl_slo_ms",
        )
        missing = [name for name in required if not hasattr(config, name)]
        if missing:
            raise ValueError(
                "recompute_aware_bounded requires config fields: "
                + ", ".join(missing)
            )
        self.max_drain_steps = config.max_drain_steps
        self.waiting_age_limit = config.waiting_age_limit
        self.ttft_slo_s = config.ttft_slo_ms / 1000.0
        self.itl_slo_s = config.itl_slo_ms / 1000.0
        for name, value in (
            ("max_drain_steps", self.max_drain_steps),
            ("waiting_age_limit", self.waiting_age_limit),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (
            ("ttft_slo_ms", config.ttft_slo_ms),
            ("itl_slo_ms", config.itl_slo_ms),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive")
        self._drain_episode_counter = 0
        self._active_drain_episode_id: int | None = None
        self._active_drain_steps = 0
        self._active_drain_tokens = 0
        self._active_drain_started_at: float | None = None
        self._drain_slo_watch: dict[int, tuple[int | None, float, str]] = {}

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        if self.policy == "decode_first_budgeted" and not self.waiting:
            self.decode_tokens_since_prefill = 0
            self.decode_steps_since_waiting = 0
        if self.policy in {
            "pressure_aware_decode",
            "chunked_prefill_budgeted",
            "recompute_aware",
            "recompute_aware_bounded",
            "mixed_slo_budget",
        }:
            self._waiting_age_steps[seq.seq_id] = 0
        self.initialize_request_progress(seq)
        self.waiting.append(seq)

    def initialize_request_progress(self, seq: Sequence) -> None:
        if self.policy not in {"recompute_aware_bounded", "mixed_slo_budget"}:
            return
        now = self._clock()
        if seq.service_arrival_at is None:
            seq.service_arrival_at = now
        seq.itl_slo_s = self.itl_slo_s
        if seq.num_completion_tokens > 0:
            if seq.next_token_deadline_at is None:
                seq.next_token_deadline_at = now + self.itl_slo_s
        elif seq.ttft_deadline_at is None:
            seq.ttft_deadline_at = now + self.ttft_slo_s

    def schedule(self) -> tuple[list[Sequence], bool] | MixedScheduleBatch:
        self._begin_step_observability()
        if self.policy in {"mixed_token_budget", "mixed_slo_budget"}:
            batch = (
                self._schedule_mixed_slo_budget()
                if self.policy == "mixed_slo_budget"
                else self._schedule_mixed_token_budget()
            )
            self._finish_mixed_step_observability(batch)
            return batch
        scheduled, is_prefill = self._schedule_impl()
        self._finish_step_observability(scheduled, is_prefill)
        return scheduled, is_prefill

    def _schedule_mixed_token_budget(self) -> MixedScheduleBatch:
        self._step_planned_decode_budget = self.max_num_batched_tokens
        self._step_planned_prefill_budget = 0
        self._step_min_ttft_slack = None
        self._step_min_itl_slack = None
        items: list[ScheduledSequence] = []
        decode, _ = self._schedule_decode(
            self.max_num_batched_tokens,
            rotate=True,
        )
        for seq in decode:
            items.append(ScheduledSequence(seq, "decode", 1, True))

        remaining = self.max_num_batched_tokens - len(decode)
        self._step_planned_prefill_budget = remaining if self.waiting else 0
        if remaining > 0 and self.waiting:
            prefill = self._schedule_prefill(remaining)
            for seq in prefill:
                completes_prefill = (
                    seq.num_cached_tokens + seq.num_scheduled_tokens
                    == seq.num_tokens
                )
                items.append(
                    ScheduledSequence(
                        seq,
                        "prefill",
                        seq.num_scheduled_tokens,
                        completes_prefill,
                    )
                )
        if not items:
            raise RuntimeError("Scheduler cannot make progress")
        self._step_mode = (
            "mixed"
            if {item.phase for item in items} == {"decode", "prefill"}
            else items[0].phase
        )
        return MixedScheduleBatch(tuple(items))

    def _schedule_mixed_slo_budget(self) -> MixedScheduleBatch:
        now = self._clock()
        ttft_slacks = [
            seq.ttft_deadline_at - now
            for seq in self.waiting
            if seq.ttft_deadline_at is not None
        ]
        itl_slacks = [
            seq.next_token_deadline_at - now
            for seq in self.running
            if seq.next_token_deadline_at is not None
        ]
        min_ttft_slack = min(ttft_slacks, default=None)
        min_itl_slack = min(itl_slacks, default=None)
        oldest_age = max(self._waiting_age_steps.values(), default=0)
        self._step_min_ttft_slack = min_ttft_slack
        self._step_min_itl_slack = min_itl_slack

        if not self.waiting:
            planned_prefill = 0
        else:
            age_levels = oldest_age // self.mixed_waiting_age_threshold
            planned_prefill = min(
                self.max_num_batched_tokens - 1,
                self.mixed_min_prefill_tokens * max(1, age_levels),
            )
            if (
                min_ttft_slack is not None
                and min_ttft_slack <= self.mixed_slack_threshold_s
            ):
                planned_prefill = max(
                    planned_prefill,
                    self.mixed_min_prefill_tokens,
                )
        planned_decode = self.max_num_batched_tokens - planned_prefill
        if (
            self.running
            and min_itl_slack is not None
            and min_itl_slack <= self.mixed_slack_threshold_s
        ):
            planned_decode = max(1, planned_decode)
            planned_prefill = self.max_num_batched_tokens - planned_decode
        self._step_planned_decode_budget = planned_decode
        self._step_planned_prefill_budget = planned_prefill

        items: list[ScheduledSequence] = []
        urgent_prefill = bool(self.waiting) and (
            oldest_age >= self.mixed_waiting_age_threshold
            or (
                min_ttft_slack is not None
                and min_ttft_slack <= self.mixed_slack_threshold_s
            )
        )
        if urgent_prefill and planned_prefill > 0:
            self._free_kv_for_waiting_head(reason="mixed_slo_prefill")
            prefill = self._schedule_prefill(planned_prefill)
            items.extend(self._mixed_prefill_items(prefill))

        prefilled_running = [
            item.sequence
            for item in items
            if item.sequence in self.running
        ]
        for seq in prefilled_running:
            self.running.remove(seq)
        decode, _ = self._schedule_decode(planned_decode, rotate=True)
        self.running.extend(prefilled_running)
        items[0:0] = [ScheduledSequence(seq, "decode", 1, True) for seq in decode]

        remaining = self.max_num_batched_tokens - sum(
            item.num_scheduled_tokens for item in items
        )
        has_prefill_items = any(item.phase == "prefill" for item in items)
        if remaining > 0 and self.waiting and not has_prefill_items:
            prefill = self._schedule_prefill(remaining)
            items.extend(self._mixed_prefill_items(prefill))
        if not items:
            raise RuntimeError("Scheduler cannot make progress")
        scheduled_prefill_ids = {
            item.sequence.seq_id for item in items if item.phase == "prefill"
        }
        self._waiting_age_steps = {
            seq.seq_id: (
                0 if seq.seq_id in scheduled_prefill_ids
                else self._waiting_age_steps.get(seq.seq_id, 0) + 1
            )
            for seq in self.waiting
        }
        self._step_mode = (
            "mixed"
            if {item.phase for item in items} == {"decode", "prefill"}
            else items[0].phase
        )
        return MixedScheduleBatch(tuple(items))

    @staticmethod
    def _mixed_prefill_items(
        seqs: list[Sequence],
    ) -> list[ScheduledSequence]:
        return [
            ScheduledSequence(
                seq,
                "prefill",
                seq.num_scheduled_tokens,
                seq.num_cached_tokens + seq.num_scheduled_tokens
                == seq.num_tokens,
            )
            for seq in seqs
        ]

    def _finish_mixed_step_observability(
        self,
        batch: MixedScheduleBatch,
    ) -> None:
        decode_ids = tuple(
            item.sequence.seq_id
            for item in batch.items
            if item.phase == "decode"
        )
        prefill_ids = tuple(
            item.sequence.seq_id
            for item in batch.items
            if item.phase == "prefill"
        )
        self._mixed_total_step_count += 1
        if decode_ids and prefill_ids:
            self._mixed_step_count += 1
        self.last_step_event = SchedulerStepEvent(
            policy=self.policy,
            state=self._step_state,
            mode=self._step_mode,
            kv_total_blocks=len(self.block_manager.blocks),
            kv_used_blocks_before=self._step_kv_used_before,
            kv_free_blocks_before=self._step_kv_free_before,
            kv_used_blocks_after=len(self.block_manager.used_block_ids),
            kv_free_blocks_after=len(self.block_manager.free_block_ids),
            running_ids_before=self._step_running_ids_before,
            waiting_ids_before=self._step_waiting_ids_before,
            running_ids_after=tuple(seq.seq_id for seq in self.running),
            waiting_ids_after=tuple(seq.seq_id for seq in self.waiting),
            selected_decode_ids=decode_ids,
            selected_prefill_ids=prefill_ids,
            oldest_waiting_age=self._step_oldest_waiting_age,
            resident_costs=self._step_resident_costs,
            preemptions=tuple(self._step_preemptions),
            scheduler_step_index=self._scheduler_step_index,
            active_progress_states=self._step_active_progress_states,
            drain_slo_watch=self._step_drain_slo_watch,
            drain_episode_id=None,
            drain_episode_step=0,
            drain_tokens=0,
            drain_episode_started_at=None,
            drain_exit_reason=None,
            fairness_trigger_reason=None,
            resource_pressure=None,
            slo_guard_seq_id=None,
            slo_guard_entry_progress_step=None,
            slo_guard_deadline_at=None,
            slo_guard_triggered_at=None,
            planned_decode_budget=self._step_planned_decode_budget,
            planned_prefill_budget=self._step_planned_prefill_budget,
            actual_decode_tokens=sum(
                item.num_scheduled_tokens
                for item in batch.items if item.phase == "decode"
            ),
            actual_prefill_tokens=sum(
                item.num_scheduled_tokens
                for item in batch.items if item.phase == "prefill"
            ),
            unused_budget=self.max_num_batched_tokens - batch.num_scheduled_tokens,
            min_ttft_slack=self._step_min_ttft_slack,
            min_itl_slack=self._step_min_itl_slack,
            mixed_step_ratio=(
                self._mixed_step_count / self._mixed_total_step_count
            ),
        )

    def _schedule_impl(self) -> tuple[list[Sequence], bool]:
        if self.policy == "pressure_aware_decode":
            return self._schedule_pressure_aware()
        if self.policy == "chunked_prefill_budgeted":
            return self._schedule_pressure_aware(
                prefill_chunk_token_budget=(
                    self.prefill_chunk_token_budget
                )
            )
        if self.policy == "recompute_aware":
            return self._schedule_recompute_aware()
        if self.policy == "recompute_aware_bounded":
            return self._schedule_recompute_aware_bounded()

        rotate_decode = self.policy == "decode_first_budgeted"
        if self.policy == "decode_first_budgeted" and self.running:
            if not self.waiting:
                scheduled, _ = self._schedule_decode(
                    self.decode_token_budget,
                    rotate=True,
                )
                if scheduled:
                    return scheduled, False
            else:
                remaining_budget = (
                    self.decode_token_budget
                    - self.decode_tokens_since_prefill
                )
                guard_allows_decode = (
                    self.decode_step_guard == 0
                    or self.decode_steps_since_waiting
                    < self.decode_step_guard
                )
                if remaining_budget > 0 and guard_allows_decode:
                    scheduled, _ = self._schedule_decode(
                        min(self.max_num_seqs, remaining_budget),
                        rotate=True,
                    )
                    if scheduled:
                        self.decode_tokens_since_prefill += len(scheduled)
                        self.decode_steps_since_waiting += 1
                        return scheduled, False
                else:
                    self._free_kv_for_waiting_head()

        scheduled_seqs = self._schedule_prefill()
        if scheduled_seqs:
            self.decode_tokens_since_prefill = 0
            self.decode_steps_since_waiting = 0
            return scheduled_seqs, True

        scheduled_seqs, _ = self._schedule_decode(
            (
                max(
                    0,
                    self.decode_token_budget
                    - self.decode_tokens_since_prefill,
                )
                if rotate_decode and self.waiting
                else (
                    self.decode_token_budget
                    if rotate_decode
                    else self.max_num_seqs
                )
            ),
            rotate=rotate_decode,
        )
        if scheduled_seqs:
            if rotate_decode and self.waiting:
                self.decode_tokens_since_prefill += len(scheduled_seqs)
                self.decode_steps_since_waiting += 1
            return scheduled_seqs, False

        scheduled_seqs = self._schedule_prefill()
        if scheduled_seqs:
            self.decode_tokens_since_prefill = 0
            self.decode_steps_since_waiting = 0
            return scheduled_seqs, True
        raise RuntimeError("Scheduler cannot make progress")

    def _begin_step_observability(self) -> None:
        self._scheduler_step_index += 1
        self._step_preemptions = []
        self._step_mode = "uninitialized"
        self._step_state = (
            self._pressure_state
            if self.policy in {
                "pressure_aware_decode",
                "chunked_prefill_budgeted",
                "recompute_aware",
                "recompute_aware_bounded",
            }
            else "static"
        )
        self._step_running_ids_before = tuple(
            seq.seq_id for seq in self.running
        )
        self._step_waiting_ids_before = tuple(
            seq.seq_id for seq in self.waiting
        )
        self._step_kv_used_before = len(
            self.block_manager.used_block_ids
        )
        self._step_kv_free_before = len(
            self.block_manager.free_block_ids
        )
        self._step_oldest_waiting_age = max(
            getattr(self, "_waiting_age_steps", {}).values(),
            default=0,
        )
        self._step_resident_costs = tuple(
            self.resident_sequence_cost(seq) for seq in self.running
        )
        self._step_active_progress_states = tuple(
            self._request_fairness_state(seq, "running")
            for seq in self.running
        ) + tuple(
            self._request_fairness_state(seq, "waiting")
            for seq in self.waiting
        )
        self._step_drain_slo_watch = ()
        self._step_drain_episode_id = None
        self._step_drain_episode_step = 0
        self._step_drain_tokens = 0
        self._step_drain_episode_started_at = None
        self._step_drain_exit_reason = None
        self._step_fairness_trigger_reason = None
        self._step_resource_pressure = None
        self._step_slo_guard_seq_id = None
        self._step_slo_guard_entry_progress_step = None
        self._step_slo_guard_deadline_at = None
        self._step_slo_guard_triggered_at = None

    def _request_fairness_state(
        self,
        seq: Sequence,
        location: str,
    ) -> RequestFairnessState:
        last_progress_step = getattr(seq, "last_progress_step", None)
        slo_deadline_at, slo_deadline_kind = self._sequence_slo_deadline(seq)
        return RequestFairnessState(
            seq_id=seq.seq_id,
            location=location,
            pending_recompute=seq.pending_recompute,
            had_emitted_token=seq.num_completion_tokens > 0,
            waiting_age_steps=(
                getattr(self, "_waiting_age_steps", {}).get(seq.seq_id, 0)
                if location == "waiting"
                else 0
            ),
            last_progress_step=last_progress_step,
            steps_since_last_progress=(
                self._scheduler_step_index - last_progress_step
                if last_progress_step is not None
                else None
            ),
            slo_deadline_at=slo_deadline_at,
            slo_deadline_kind=slo_deadline_kind,
        )

    def _finish_step_observability(
        self,
        scheduled: list[Sequence],
        is_prefill: bool,
    ) -> None:
        if self._step_mode == "uninitialized":
            self._step_mode = "prefill" if is_prefill else "decode"
        selected_ids = tuple(seq.seq_id for seq in scheduled)
        self.last_step_event = SchedulerStepEvent(
            policy=self.policy,
            state=self._step_state,
            mode=self._step_mode,
            kv_total_blocks=len(self.block_manager.blocks),
            kv_used_blocks_before=self._step_kv_used_before,
            kv_free_blocks_before=self._step_kv_free_before,
            kv_used_blocks_after=len(self.block_manager.used_block_ids),
            kv_free_blocks_after=len(self.block_manager.free_block_ids),
            running_ids_before=self._step_running_ids_before,
            waiting_ids_before=self._step_waiting_ids_before,
            running_ids_after=tuple(seq.seq_id for seq in self.running),
            waiting_ids_after=tuple(seq.seq_id for seq in self.waiting),
            selected_decode_ids=() if is_prefill else selected_ids,
            selected_prefill_ids=selected_ids if is_prefill else (),
            oldest_waiting_age=self._step_oldest_waiting_age,
            resident_costs=self._step_resident_costs,
            preemptions=tuple(self._step_preemptions),
            scheduler_step_index=self._scheduler_step_index,
            active_progress_states=self._step_active_progress_states,
            drain_slo_watch=self._step_drain_slo_watch,
            drain_episode_id=self._step_drain_episode_id,
            drain_episode_step=self._step_drain_episode_step,
            drain_tokens=self._step_drain_tokens,
            drain_episode_started_at=self._step_drain_episode_started_at,
            drain_exit_reason=self._step_drain_exit_reason,
            fairness_trigger_reason=self._step_fairness_trigger_reason,
            resource_pressure=self._step_resource_pressure,
            slo_guard_seq_id=self._step_slo_guard_seq_id,
            slo_guard_entry_progress_step=(
                self._step_slo_guard_entry_progress_step
            ),
            slo_guard_deadline_at=self._step_slo_guard_deadline_at,
            slo_guard_triggered_at=self._step_slo_guard_triggered_at,
        )

    def resident_sequence_cost(
        self,
        seq: Sequence,
    ) -> ResidentSequenceCost:
        releasable_blocks = sum(
            self.block_manager.blocks[block_id].ref_count == 1
            for block_id in seq.block_table
        )
        guaranteed_shared_prefix_tokens = 0
        for block_id in seq.block_table[:-1]:
            block = self.block_manager.blocks[block_id]
            if block.ref_count <= 1:
                break
            guaranteed_shared_prefix_tokens += self.block_size
        estimated_recompute_tokens = max(
            seq.num_tokens - guaranteed_shared_prefix_tokens,
            0,
        )
        remaining_decode_tokens = max(
            seq.max_tokens - seq.num_completion_tokens,
            0,
        )
        return ResidentSequenceCost(
            seq_id=seq.seq_id,
            resident_kv_blocks=len(seq.block_table),
            resident_kv_tokens=seq.num_cached_tokens,
            logical_context_tokens=seq.num_tokens,
            remaining_decode_tokens=remaining_decode_tokens,
            releasable_blocks=releasable_blocks,
            estimated_recompute_tokens=estimated_recompute_tokens,
            estimated_steps_to_release=remaining_decode_tokens,
            can_advance_without_free_block=(
                len(seq) % self.block_size != 1
            ),
        )

    def select_recompute_victim(
        self,
        candidates,
    ) -> Sequence:
        candidates = list(candidates)
        if not candidates:
            raise RuntimeError("No resident sequence is available to preempt")
        return min(
            candidates,
            key=lambda seq: (
                self.resident_sequence_cost(seq).releasable_blocks == 0,
                self.resident_sequence_cost(seq).preemption_harm,
                self.resident_sequence_cost(seq).estimated_recompute_tokens,
                -self.resident_sequence_cost(seq).releasable_blocks,
                seq.seq_id,
            ),
        )

    def _select_preemption_victim(self, candidates) -> Sequence:
        if self.policy in {
            "recompute_aware",
            "recompute_aware_bounded",
        }:
            return self.select_recompute_victim(candidates)
        return candidates[-1]

    def _rank_running_for_drain(self) -> None:
        ranked = sorted(
            self.running,
            key=lambda seq: (
                not self.resident_sequence_cost(
                    seq
                ).can_advance_without_free_block,
                -self.resident_sequence_cost(seq).release_efficiency,
                -self.resident_sequence_cost(
                    seq
                ).estimated_recompute_tokens,
                self.resident_sequence_cost(seq).remaining_decode_tokens,
                seq.seq_id,
            ),
        )
        self.running = deque(ranked)

    def _schedule_recompute_aware(self) -> tuple[list[Sequence], bool]:
        state, signals = self._classify_pressure_state()
        self._step_state = state
        preemption_count_before = self.preemption_count
        forced_prefill = (
            bool(self.waiting)
            and signals["max_initial_waiting_age_steps"]
            >= self.pressure_waiting_age_threshold
        )
        if state in {"pressure", "critical"}:
            decode_budget = self.pressure_decode_token_budget
            decode_guard = self.pressure_decode_step_guard
        else:
            decode_budget = self.decode_token_budget
            decode_guard = self.decode_step_guard
        remaining_budget = max(
            0,
            decode_budget - self.decode_tokens_since_prefill,
        )

        if forced_prefill:
            self._step_mode = "starvation_prefill"
            self._free_kv_for_waiting_head(
                reason="waiting_prefill_allocation"
            )
            scheduled = self._schedule_prefill()
            if scheduled:
                self.decode_tokens_since_prefill = 0
                self.decode_steps_since_waiting = 0
                return self._finish_pressure_decision(
                    scheduled,
                    True,
                    state,
                    signals,
                    decode_budget,
                    decode_guard,
                    forced_prefill,
                    preemption_count_before,
                )

        if self.running:
            if not self.waiting:
                self._step_mode = "normal_decode"
                scheduled, _ = self._schedule_decode(
                    decode_budget,
                    rotate=True,
                )
                if scheduled:
                    return self._finish_pressure_decision(
                        scheduled,
                        False,
                        state,
                        signals,
                        decode_budget,
                        decode_guard,
                        forced_prefill,
                        preemption_count_before,
                    )
            else:
                guard_allows_decode = (
                    decode_guard == 0
                    or self.decode_steps_since_waiting < decode_guard
                )
                high_pressure = state in {"pressure", "critical"}
                if high_pressure:
                    self._rank_running_for_drain()
                    max_decode_seqs = min(1, remaining_budget)
                    self._step_mode = "drain_decode"
                elif remaining_budget > 0 and guard_allows_decode:
                    max_decode_seqs = min(
                        self.max_num_seqs,
                        remaining_budget,
                    )
                    self._step_mode = "baseline_decode"
                else:
                    max_decode_seqs = 0
                if max_decode_seqs > 0:
                    scheduled, _ = self._schedule_decode(
                        max_decode_seqs,
                        rotate=True,
                    )
                    if scheduled:
                        self.decode_tokens_since_prefill += len(scheduled)
                        self.decode_steps_since_waiting += 1
                        return self._finish_pressure_decision(
                            scheduled,
                            False,
                            state,
                            signals,
                            decode_budget,
                            decode_guard,
                            forced_prefill,
                            preemption_count_before,
                        )
                self._step_mode = "budget_prefill"
                self._free_kv_for_waiting_head(
                    reason="waiting_prefill_allocation"
                )

        scheduled = self._schedule_prefill()
        if scheduled:
            self.decode_tokens_since_prefill = 0
            self.decode_steps_since_waiting = 0
            if self._step_mode == "uninitialized":
                self._step_mode = "prefill"
            return self._finish_pressure_decision(
                scheduled,
                True,
                state,
                signals,
                decode_budget,
                decode_guard,
                forced_prefill,
                preemption_count_before,
            )

        self._step_mode = "fallback_decode"
        scheduled, _ = self._schedule_decode(
            remaining_budget if self.waiting else decode_budget,
            rotate=True,
        )
        if scheduled:
            if self.waiting:
                self.decode_tokens_since_prefill += len(scheduled)
                self.decode_steps_since_waiting += 1
            return self._finish_pressure_decision(
                scheduled,
                False,
                state,
                signals,
                decode_budget,
                decode_guard,
                forced_prefill,
                preemption_count_before,
            )
        raise RuntimeError("Scheduler cannot make progress")

    def _active_sequences(self) -> list[Sequence]:
        return [*self.running, *self.waiting]

    def _sequence_slo_deadline(
        self,
        seq: Sequence,
    ) -> tuple[float | None, str | None]:
        if seq.num_completion_tokens > 0:
            return seq.next_token_deadline_at, "itl_slo_deadline"
        return seq.ttft_deadline_at, "ttft_slo_deadline"

    def _start_drain_episode(self) -> None:
        self._drain_episode_counter += 1
        self._active_drain_episode_id = self._drain_episode_counter
        self._active_drain_steps = 0
        self._active_drain_tokens = 0
        now = self._clock()
        self._active_drain_started_at = now
        self._drain_slo_watch = {}
        for seq in self._active_sequences():
            deadline, reason = self._sequence_slo_deadline(seq)
            if deadline is not None and reason is not None and deadline > now:
                self._drain_slo_watch[seq.seq_id] = (
                    seq.last_progress_step,
                    deadline,
                    reason,
                )
        self._capture_drain_slo_watch()

    def _capture_drain_slo_watch(self) -> None:
        self._step_drain_slo_watch = tuple(
            DrainSloWatchState(
                seq_id=seq_id,
                entry_progress_step=entry_progress_step,
                deadline_at=deadline,
                deadline_kind=reason,
            )
            for seq_id, (
                entry_progress_step,
                deadline,
                reason,
            ) in sorted(self._drain_slo_watch.items())
        )

    def _close_drain_episode(
        self,
        decision: DrainExitDecision,
    ) -> None:
        self._capture_drain_slo_watch()
        self._step_drain_episode_id = self._active_drain_episode_id
        self._step_drain_episode_step = self._active_drain_steps
        self._step_drain_tokens = self._active_drain_tokens
        self._step_drain_episode_started_at = self._active_drain_started_at
        self._step_drain_exit_reason = decision.reason
        self._step_slo_guard_seq_id = decision.slo_guard_seq_id
        self._step_slo_guard_entry_progress_step = (
            decision.slo_guard_entry_progress_step
        )
        self._step_slo_guard_deadline_at = decision.slo_guard_deadline_at
        self._step_slo_guard_triggered_at = decision.slo_guard_triggered_at
        self._active_drain_episode_id = None
        self._active_drain_steps = 0
        self._active_drain_tokens = 0
        self._active_drain_started_at = None
        self._drain_slo_watch = {}

    def _expired_drain_slo_guard(
        self,
    ) -> DrainExitDecision | None:
        now = self._clock()
        active = {seq.seq_id: seq for seq in self._active_sequences()}
        for seq_id in sorted(self._drain_slo_watch):
            entry_progress_step, deadline, reason = self._drain_slo_watch[
                seq_id
            ]
            seq = active.get(seq_id)
            if seq is None or seq.last_progress_step != entry_progress_step:
                continue
            if now >= deadline:
                return DrainExitDecision(
                    reason=reason,
                    slo_guard_seq_id=seq_id,
                    slo_guard_entry_progress_step=entry_progress_step,
                    slo_guard_deadline_at=deadline,
                    slo_guard_triggered_at=now,
                )
        return None

    def _bounded_drain_exit(
        self,
        signals: dict[str, int | float],
        *,
        resource_pressure: bool,
    ) -> DrainExitDecision | None:
        if self._active_drain_episode_id is None:
            if signals["max_waiting_age_steps"] >= self.waiting_age_limit:
                return DrainExitDecision("waiting_age_limit")
            return None
        if not resource_pressure:
            return DrainExitDecision("resource_released")
        if self._active_drain_steps >= self.max_drain_steps:
            return DrainExitDecision("max_drain_steps")
        if signals["max_waiting_age_steps"] >= self.waiting_age_limit:
            return DrainExitDecision("waiting_age_limit")
        slo_guard = self._expired_drain_slo_guard()
        if slo_guard is not None:
            return slo_guard
        return None

    def _schedule_bounded_prefill(
        self,
        *,
        decision: DrainExitDecision,
        state: str,
        signals: dict[str, int | float],
        decode_budget: int,
        decode_guard: int,
        preemption_count_before: int,
    ) -> tuple[list[Sequence], bool] | None:
        self._step_mode = "bounded_prefill"
        if self.waiting:
            self._free_kv_for_waiting_head(
                reason="bounded_fairness_prefill"
            )
        scheduled = self._schedule_prefill()
        if not scheduled:
            return None
        if self._active_drain_episode_id is not None:
            self._close_drain_episode(decision)
        else:
            self._step_fairness_trigger_reason = decision.reason
            self._step_slo_guard_seq_id = decision.slo_guard_seq_id
            self._step_slo_guard_entry_progress_step = (
                decision.slo_guard_entry_progress_step
            )
            self._step_slo_guard_deadline_at = decision.slo_guard_deadline_at
            self._step_slo_guard_triggered_at = (
                decision.slo_guard_triggered_at
            )
        self.decode_tokens_since_prefill = 0
        self.decode_steps_since_waiting = 0
        return self._finish_pressure_decision(
            scheduled,
            True,
            state,
            signals,
            decode_budget,
            decode_guard,
            False,
            preemption_count_before,
        )

    def _schedule_recompute_aware_bounded(
        self,
    ) -> tuple[list[Sequence], bool]:
        state, signals = self._classify_pressure_state()
        self._step_state = state
        preemption_count_before = self.preemption_count
        forced_initial_prefill = (
            bool(self.waiting)
            and signals["max_initial_waiting_age_steps"]
            >= self.pressure_waiting_age_threshold
        )
        if state in {"pressure", "critical"}:
            decode_budget = self.pressure_decode_token_budget
            decode_guard = self.pressure_decode_step_guard
        else:
            decode_budget = self.decode_token_budget
            decode_guard = self.decode_step_guard
        remaining_budget = max(
            0,
            decode_budget - self.decode_tokens_since_prefill,
        )

        resource_pressure = (
            signals["kv_utilization"] >= self.pressure_high_utilization
            or signals["recent_preemptions"]
            >= self.pressure_preemption_threshold
        )
        self._step_resource_pressure = resource_pressure

        if (
            forced_initial_prefill
            and self._active_drain_episode_id is None
        ):
            scheduled = self._schedule_bounded_prefill(
                decision=DrainExitDecision("initial_waiting_age"),
                state=state,
                signals=signals,
                decode_budget=decode_budget,
                decode_guard=decode_guard,
                preemption_count_before=preemption_count_before,
            )
            if scheduled is not None:
                return scheduled

        if self.running and self.waiting:
            exit_decision = self._bounded_drain_exit(
                signals,
                resource_pressure=resource_pressure,
            )
            if exit_decision is not None:
                scheduled = self._schedule_bounded_prefill(
                    decision=exit_decision,
                    state=state,
                    signals=signals,
                    decode_budget=decode_budget,
                    decode_guard=decode_guard,
                    preemption_count_before=preemption_count_before,
                )
                if scheduled is not None:
                    return scheduled
            if resource_pressure and remaining_budget > 0:
                self._rank_running_for_drain()
                self._step_mode = "drain_decode"
                scheduled, _ = self._schedule_decode(
                    min(1, remaining_budget),
                    rotate=True,
                )
                if scheduled:
                    if self._active_drain_episode_id is None:
                        self._start_drain_episode()
                    else:
                        self._capture_drain_slo_watch()
                    self._active_drain_steps += 1
                    self._active_drain_tokens += len(scheduled)
                    self._step_drain_episode_id = (
                        self._active_drain_episode_id
                    )
                    self._step_drain_episode_step = self._active_drain_steps
                    self._step_drain_tokens = self._active_drain_tokens
                    self._step_drain_episode_started_at = (
                        self._active_drain_started_at
                    )
                    self.decode_tokens_since_prefill += len(scheduled)
                    self.decode_steps_since_waiting += 1
                    return self._finish_pressure_decision(
                        scheduled,
                        False,
                        state,
                        signals,
                        decode_budget,
                        decode_guard,
                        False,
                        preemption_count_before,
                    )

            guard_allows_decode = (
                decode_guard == 0
                or self.decode_steps_since_waiting < decode_guard
            )
            if remaining_budget > 0 and guard_allows_decode:
                self._step_mode = "baseline_decode"
                scheduled, _ = self._schedule_decode(
                    min(self.max_num_seqs, remaining_budget),
                    rotate=True,
                )
                if scheduled:
                    self.decode_tokens_since_prefill += len(scheduled)
                    self.decode_steps_since_waiting += 1
                    return self._finish_pressure_decision(
                        scheduled,
                        False,
                        state,
                        signals,
                        decode_budget,
                        decode_guard,
                        False,
                        preemption_count_before,
                    )

        if self.running and not self.waiting:
            if self._active_drain_episode_id is not None:
                self._close_drain_episode(
                    DrainExitDecision("waiting_empty")
                )
            self._step_mode = "normal_decode"
            scheduled, _ = self._schedule_decode(
                decode_budget,
                rotate=True,
            )
            if scheduled:
                return self._finish_pressure_decision(
                    scheduled,
                    False,
                    state,
                    signals,
                    decode_budget,
                    decode_guard,
                    False,
                    preemption_count_before,
                )

        if not self.running and self._active_drain_episode_id is not None:
            self._close_drain_episode(
                DrainExitDecision("resident_empty")
            )

        self._step_mode = "prefill"
        if self.waiting:
            self._free_kv_for_waiting_head(
                reason="waiting_prefill_allocation"
            )
        scheduled = self._schedule_prefill()
        if scheduled:
            self.decode_tokens_since_prefill = 0
            self.decode_steps_since_waiting = 0
            return self._finish_pressure_decision(
                scheduled,
                True,
                state,
                signals,
                decode_budget,
                decode_guard,
                False,
                preemption_count_before,
            )

        self._step_mode = "fallback_decode"
        scheduled, _ = self._schedule_decode(
            remaining_budget if self.waiting else decode_budget,
            rotate=True,
        )
        if scheduled:
            if self.waiting:
                self.decode_tokens_since_prefill += len(scheduled)
                self.decode_steps_since_waiting += 1
            return self._finish_pressure_decision(
                scheduled,
                False,
                state,
                signals,
                decode_budget,
                decode_guard,
                False,
                preemption_count_before,
            )
        raise RuntimeError("Scheduler cannot make progress")

    def _schedule_pressure_aware(
        self,
        prefill_chunk_token_budget: int | None = None,
    ) -> tuple[list[Sequence], bool]:
        state, signals = self._classify_pressure_state()
        preemption_count_before = self.preemption_count
        forced_prefill = (
            bool(self.waiting)
            and signals["max_waiting_age_steps"]
            >= self.pressure_waiting_age_threshold
        )
        if state in {"pressure", "critical"}:
            decode_budget = self.pressure_decode_token_budget
            decode_guard = self.pressure_decode_step_guard
        elif state == "normal":
            decode_budget = self.decode_token_budget
            decode_guard = self.decode_step_guard
        else:
            decode_budget = 0
            decode_guard = 0
        remaining_budget = max(
            0,
            decode_budget - self.decode_tokens_since_prefill,
        )

        if forced_prefill:
            self._free_kv_for_waiting_head()
            scheduled = self._schedule_adaptive_prefill(
                prefill_chunk_token_budget
            )
            if scheduled:
                self.decode_tokens_since_prefill = 0
                self.decode_steps_since_waiting = 0
                return self._finish_pressure_decision(
                    scheduled,
                    True,
                    state,
                    signals,
                    decode_budget,
                    decode_guard,
                    forced_prefill,
                    preemption_count_before,
                )

        if self.running:
            if not self.waiting:
                scheduled, _ = self._schedule_decode(
                    decode_budget,
                    rotate=True,
                )
                if scheduled:
                    return self._finish_pressure_decision(
                        scheduled,
                        False,
                        state,
                        signals,
                        decode_budget,
                        decode_guard,
                        forced_prefill,
                        preemption_count_before,
                    )
            else:
                guard_allows_decode = (
                    decode_guard == 0
                    or self.decode_steps_since_waiting < decode_guard
                )
                if remaining_budget > 0 and guard_allows_decode:
                    scheduled, _ = self._schedule_decode(
                        min(self.max_num_seqs, remaining_budget),
                        rotate=True,
                    )
                    if scheduled:
                        self.decode_tokens_since_prefill += len(scheduled)
                        self.decode_steps_since_waiting += 1
                        return self._finish_pressure_decision(
                            scheduled,
                            False,
                            state,
                            signals,
                            decode_budget,
                            decode_guard,
                            forced_prefill,
                            preemption_count_before,
                        )
                self._free_kv_for_waiting_head()

        scheduled = self._schedule_adaptive_prefill(
            prefill_chunk_token_budget
        )
        if scheduled:
            self.decode_tokens_since_prefill = 0
            self.decode_steps_since_waiting = 0
            return self._finish_pressure_decision(
                scheduled,
                True,
                state,
                signals,
                decode_budget,
                decode_guard,
                forced_prefill,
                preemption_count_before,
            )

        scheduled, _ = self._schedule_decode(
            remaining_budget if self.waiting else decode_budget,
            rotate=True,
        )
        if scheduled:
            if self.waiting:
                self.decode_tokens_since_prefill += len(scheduled)
                self.decode_steps_since_waiting += 1
            return self._finish_pressure_decision(
                scheduled,
                False,
                state,
                signals,
                decode_budget,
                decode_guard,
                forced_prefill,
                preemption_count_before,
            )
        raise RuntimeError("Scheduler cannot make progress")

    def _classify_pressure_state(self) -> tuple[str, dict[str, int | float]]:
        signals = {
            "kv_utilization": self.block_manager.current_utilization(),
            "recent_preemptions": sum(self._recent_preemption_deltas),
            "max_waiting_age_steps": max(
                self._waiting_age_steps.values(),
                default=0,
            ),
            "max_initial_waiting_age_steps": max(
                (
                    self._waiting_age_steps.get(seq.seq_id, 0)
                    for seq in self.waiting
                    if not seq.pending_recompute
                ),
                default=0,
            ),
        }
        critical = (
            signals["kv_utilization"] >= self.pressure_critical_utilization
            or signals["max_waiting_age_steps"]
            >= self.pressure_waiting_age_threshold
        )
        pressured = (
            signals["kv_utilization"] >= self.pressure_high_utilization
            or signals["recent_preemptions"]
            >= self.pressure_preemption_threshold
        )
        if critical:
            self._pressure_state = "critical"
            self._pressure_clean_steps = 0
        elif self._pressure_state == "critical":
            self._pressure_clean_steps += 1
            if self._pressure_clean_steps >= self.pressure_hysteresis_steps:
                self._pressure_state = "pressure" if pressured else "normal"
                self._pressure_clean_steps = 0
        elif pressured:
            self._pressure_state = "pressure"
            self._pressure_clean_steps = 0
        elif self._pressure_state != "normal":
            self._pressure_clean_steps += 1
            if self._pressure_clean_steps >= self.pressure_hysteresis_steps:
                self._pressure_state = "normal"
                self._pressure_clean_steps = 0
        return self._pressure_state, signals

    def _finish_pressure_decision(
        self,
        scheduled: list[Sequence],
        is_prefill: bool,
        state: str,
        signals: dict[str, int | float],
        decode_budget: int,
        decode_guard: int,
        forced_prefill: bool,
        preemption_count_before: int,
    ) -> tuple[list[Sequence], bool]:
        self._step_state = state
        scheduled_ids = {seq.seq_id for seq in scheduled}
        waiting_age_steps = {}
        for seq in self.waiting:
            if seq.seq_id in scheduled_ids:
                waiting_age_steps[seq.seq_id] = 0
            else:
                previous_age = self._waiting_age_steps.get(seq.seq_id)
                waiting_age_steps[seq.seq_id] = (
                    0 if previous_age is None else previous_age + 1
                )
        self._waiting_age_steps = waiting_age_steps
        self._recent_preemption_deltas.append(
            self.preemption_count - preemption_count_before
        )
        self._pressure_decision_count += 1
        self._pressure_last_decision = {
            **signals,
            "state": state,
            "phase": "prefill" if is_prefill else "decode",
            "mode": self._step_mode,
            "selected_decode_ids": (
                [] if is_prefill else sorted(scheduled_ids)
            ),
            "selected_prefill_ids": (
                sorted(scheduled_ids) if is_prefill else []
            ),
            "decode_token_budget": decode_budget,
            "decode_step_guard": decode_guard,
            "forced_prefill": forced_prefill,
        }
        if self.policy == "chunked_prefill_budgeted":
            self._pressure_last_decision.update(
                {
                    "prefill_chunk_token_budget": (
                        self.prefill_chunk_token_budget
                    ),
                    "effective_prefill_token_budget": (
                        self._last_effective_prefill_token_budget
                    ),
                    "scheduled_prefill_tokens": (
                        sum(seq.num_scheduled_tokens for seq in scheduled)
                        if is_prefill
                        else 0
                    ),
                }
            )
        return scheduled, is_prefill

    def _free_kv_for_waiting_head(
        self,
        *,
        reason: str = "waiting_prefill_allocation",
    ) -> None:
        seq = self.waiting[0]
        if seq.block_table:
            return
        while self.block_manager.can_allocate(seq) == -1 and self.running:
            victim = self._select_preemption_victim(self.running)
            self.running.remove(victim)
            self.preempt(
                victim,
                prepend=False,
                reason=reason,
                triggering_seq_id=seq.seq_id,
            )

    def _schedule_adaptive_prefill(
        self,
        configured_token_budget: int | None,
    ) -> list[Sequence]:
        token_budget = (
            configured_token_budget
            if configured_token_budget is not None and self.running
            else None
        )
        if self.policy == "chunked_prefill_budgeted":
            self._last_effective_prefill_token_budget = (
                self.max_num_batched_tokens
                if token_budget is None
                else token_budget
            )
        return self._schedule_prefill(token_budget)

    def _schedule_prefill(
        self,
        token_budget: int | None = None,
    ) -> list[Sequence]:
        scheduled_seqs = []
        num_batched_tokens = 0
        max_num_batched_tokens = (
            self.max_num_batched_tokens
            if token_budget is None
            else token_budget
        )

        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                cached_blocks = self.block_manager.num_cached_blocks(seq)
                cached_tokens = cached_blocks * self.block_size
                num_tokens = seq.num_tokens - cached_tokens
                scheduled_tokens = min(num_tokens, remaining)
                materialized_tokens = (
                    cached_tokens + scheduled_tokens
                    if self.incremental_kv_allocation
                    else None
                )
                num_cached_blocks = self.block_manager.can_allocate(
                    seq,
                    materialized_tokens,
                )
                if num_cached_blocks == -1:
                    break
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
                scheduled_tokens = min(num_tokens, remaining)
                materialized_tokens = (
                    seq.num_cached_tokens + scheduled_tokens
                )
                if (
                    self.incremental_kv_allocation
                    and not self.block_manager.can_materialize(
                        seq,
                        materialized_tokens,
                    )
                ):
                    break
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(
                    seq,
                    num_cached_blocks,
                    materialized_tokens,
                )
            elif self.incremental_kv_allocation:
                self.block_manager.materialize(seq, materialized_tokens)
            seq.num_scheduled_tokens = scheduled_tokens
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        return scheduled_seqs

    def _schedule_decode(
        self,
        policy_decode_budget: int,
        *,
        rotate: bool,
    ) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        decode_batch_size = min(
            self.max_num_seqs,
            self.max_num_batched_tokens,
            policy_decode_budget,
        )
        while self.running and len(scheduled_seqs) < decode_batch_size:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    victim = self._select_preemption_victim(self.running)
                    self.running.remove(victim)
                    self.preempt(
                        victim,
                        prepend=self.policy == "prefill_first",
                        reason="decode_kv_expansion",
                        triggering_seq_id=seq.seq_id,
                    )
                else:
                    self.preempt(
                        seq,
                        prepend=self.policy == "prefill_first",
                        reason="decode_kv_expansion_self",
                        triggering_seq_id=seq.seq_id,
                    )
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        if rotate:
            self.running.extend(scheduled_seqs)
        else:
            self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(
        self,
        seq: Sequence,
        *,
        prepend: bool = True,
        reason: str = "manual",
        triggering_seq_id: int | None = None,
    ):
        cost = self.resident_sequence_cost(seq)
        self._step_preemptions.append(
            PreemptionEvent(
                seq_id=seq.seq_id,
                reason=reason,
                triggering_seq_id=triggering_seq_id,
                logical_context_tokens=cost.logical_context_tokens,
                resident_kv_tokens=cost.resident_kv_tokens,
                resident_blocks=cost.resident_kv_blocks,
                releasable_blocks=cost.releasable_blocks,
                estimated_recompute_tokens=(
                    cost.estimated_recompute_tokens
                ),
                remaining_decode_tokens=cost.remaining_decode_tokens,
                scheduler_state=getattr(
                    self,
                    "_pressure_state",
                    "static",
                ),
            )
        )
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        seq.preemption_count += 1
        seq.pending_recompute = True
        self.block_manager.deallocate(seq)
        if prepend:
            self.waiting.appendleft(seq)
        else:
            self.waiting.append(seq)
        self.preemption_count += 1

    def reset_observability(self):
        self.preemption_count = 0
        self.actual_recompute_tokens = 0
        self.resume_count = 0
        self._mixed_step_count = 0
        self._mixed_total_step_count = 0
        self.last_step_event = None
        self._step_preemptions = []
        self.block_manager.reset_observability()
        if self.policy in {
            "pressure_aware_decode",
            "chunked_prefill_budgeted",
            "recompute_aware",
            "recompute_aware_bounded",
        }:
            self._pressure_state = "normal"
            self._pressure_clean_steps = 0
            self._recent_preemption_deltas.clear()
            self._waiting_age_steps = {
                seq.seq_id: 0 for seq in self.waiting
            }
            self._pressure_decision_count = 0
            self._pressure_last_decision = None
        if self.policy == "recompute_aware_bounded":
            self._scheduler_step_index = -1
            self._drain_episode_counter = 0
            self._active_drain_episode_id = None
            self._active_drain_steps = 0
            self._active_drain_tokens = 0
            self._active_drain_started_at = None
            self._drain_slo_watch = {}
            for seq in self._active_sequences():
                seq.last_progress_step = None
                seq.last_progress_gap_steps = None
                seq.last_progress_at = None

    def observability_snapshot(self) -> dict:
        snapshot = {
            "preemption_count": self.preemption_count,
            "actual_recompute_tokens": self.actual_recompute_tokens,
            "resume_count": self.resume_count,
            "kv_cache": self.block_manager.observability_snapshot(),
        }
        if self.policy in {
            "pressure_aware_decode",
            "chunked_prefill_budgeted",
            "recompute_aware",
            "recompute_aware_bounded",
        }:
            namespace = {
                "pressure_aware_decode": "pressure_aware",
                "chunked_prefill_budgeted": "chunked_prefill_budgeted",
                "recompute_aware": "recompute_aware",
                "recompute_aware_bounded": "recompute_aware_bounded",
            }[self.policy]
            snapshot[namespace] = {
                "state": self._pressure_state,
                "recent_preemptions": sum(self._recent_preemption_deltas),
                "max_waiting_age_steps": max(
                    self._waiting_age_steps.values(),
                    default=0,
                ),
                "clean_steps": self._pressure_clean_steps,
                "decision_count": self._pressure_decision_count,
                "last_decision": self._pressure_last_decision,
            }
            if self.policy == "recompute_aware_bounded":
                snapshot[namespace].update(
                    {
                        "scheduler_step_index": self._scheduler_step_index,
                        "drain_episode_count": self._drain_episode_counter,
                        "drain_active": (
                            self._active_drain_episode_id is not None
                        ),
                        "drain_episode_id": self._active_drain_episode_id,
                        "drain_steps": self._active_drain_steps,
                        "drain_tokens": self._active_drain_tokens,
                    }
                )
        return snapshot

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            scheduled_tokens = seq.num_scheduled_tokens
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += scheduled_tokens
            if is_prefill and seq.pending_recompute:
                seq.actual_recompute_tokens += scheduled_tokens
                self.actual_recompute_tokens += scheduled_tokens
            seq.num_scheduled_tokens = 0
            if self.policy == "recompute_aware_bounded":
                if seq.last_progress_step is None:
                    seq.last_progress_gap_steps = None
                else:
                    seq.last_progress_gap_steps = (
                        self._scheduler_step_index - seq.last_progress_step
                    )
                seq.last_progress_step = self._scheduler_step_index
                seq.last_progress_at = self._clock()
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            if is_prefill and seq.pending_recompute:
                seq.pending_recompute = False
                seq.resume_count += 1
                self.resume_count += 1
            seq.append_token(token_id)
            if self.policy == "recompute_aware_bounded":
                seq.next_token_deadline_at = (
                    self._clock() + self.itl_slo_s
                )
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)

    def postprocess_mixed(
        self,
        batch: MixedScheduleBatch,
        sampled_token_ids: dict[int, int],
    ) -> None:
        for item in batch.items:
            seq = item.sequence
            scheduled_tokens = item.num_scheduled_tokens
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += scheduled_tokens
            if item.phase == "prefill" and seq.pending_recompute:
                seq.actual_recompute_tokens += scheduled_tokens
                self.actual_recompute_tokens += scheduled_tokens
            seq.num_scheduled_tokens = 0
            if item.phase == "prefill" and seq.num_cached_tokens < seq.num_tokens:
                continue
            if item.phase == "prefill" and seq.pending_recompute:
                seq.pending_recompute = False
                seq.resume_count += 1
                self.resume_count += 1
            if not item.requires_sampling:
                continue
            token_id = sampled_token_ids[seq.seq_id]
            seq.append_token(token_id)
            if (
                (not seq.ignore_eos and token_id == self.eos)
                or seq.num_completion_tokens == seq.max_tokens
            ):
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
