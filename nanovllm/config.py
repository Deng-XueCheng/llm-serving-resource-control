import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    scheduler_policy: str = "prefill_first"
    decode_token_budget: int = 0
    decode_step_guard: int = 0
    cuda_event_timing: bool = False
    pressure_decode_token_budget: int = 4
    pressure_decode_step_guard: int = 0
    pressure_high_utilization: float = 0.75
    pressure_critical_utilization: float = 1.0
    pressure_preemption_window: int = 4
    pressure_preemption_threshold: int = 2
    pressure_hysteresis_steps: int = 2
    pressure_waiting_age_threshold: int = 4
    prefill_chunk_token_budget: int = 0
    max_drain_steps: int = 0
    waiting_age_limit: int = 0
    ttft_slo_ms: float = 0.0
    itl_slo_ms: float = 0.0
    mixed_min_prefill_tokens: int = 0
    mixed_waiting_age_threshold: int = 0
    mixed_slack_threshold_ms: float = 0.0
    distributed_init_port: int = 2333
    incremental_kv_allocation: bool = False

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        if (
            not isinstance(self.num_kvcache_blocks, int)
            or isinstance(self.num_kvcache_blocks, bool)
            or (
                self.num_kvcache_blocks != -1
                and self.num_kvcache_blocks <= 0
            )
        ):
            raise ValueError(
                "num_kvcache_blocks must be -1 or a positive integer"
            )
        if (
            not isinstance(self.scheduler_policy, str)
            or self.scheduler_policy
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
            raise ValueError(
                f"Unknown scheduler policy: {self.scheduler_policy}"
            )
        if self.scheduler_policy in {
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
            raise ValueError(
                "decode_token_budget must be a positive integer "
                "for a Decode-first policy"
            )
        if (
            self.scheduler_policy == "prefill_first"
            and (
                not isinstance(self.decode_token_budget, int)
                or isinstance(self.decode_token_budget, bool)
                or self.decode_token_budget != 0
            )
        ):
            raise ValueError(
                "decode_token_budget must be 0 for prefill_first"
            )
        if (
            not isinstance(self.decode_step_guard, int)
            or isinstance(self.decode_step_guard, bool)
            or self.decode_step_guard < 0
        ):
            raise ValueError("decode_step_guard must be a non-negative integer")
        if (
            self.scheduler_policy == "prefill_first"
            and self.decode_step_guard != 0
        ):
            raise ValueError(
                "decode_step_guard must be 0 for prefill_first"
            )
        if self.scheduler_policy in {
            "pressure_aware_decode",
            "chunked_prefill_budgeted",
            "recompute_aware",
            "recompute_aware_bounded",
        }:
            self._validate_pressure_aware_fields()
        if self.scheduler_policy == "mixed_slo_budget":
            for name, value in (
                ("mixed_min_prefill_tokens", self.mixed_min_prefill_tokens),
                ("mixed_waiting_age_threshold", self.mixed_waiting_age_threshold),
            ):
                if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                    raise ValueError(f"{name} must be a positive integer")
            if self.mixed_min_prefill_tokens >= self.max_num_batched_tokens:
                raise ValueError("mixed_min_prefill_tokens must leave one Decode token")
            for name, value in (
                ("mixed_slack_threshold_ms", self.mixed_slack_threshold_ms),
                ("ttft_slo_ms", self.ttft_slo_ms),
                ("itl_slo_ms", self.itl_slo_ms),
            ):
                if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                    raise ValueError(f"{name} must be positive")
        elif self.scheduler_policy == "recompute_aware_bounded":
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
                ("ttft_slo_ms", self.ttft_slo_ms),
                ("itl_slo_ms", self.itl_slo_ms),
            ):
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or value <= 0
                ):
                    raise ValueError(f"{name} must be positive")
        elif any(
            value != 0
            for value in (
                self.max_drain_steps,
                self.waiting_age_limit,
                self.ttft_slo_ms,
                self.itl_slo_ms,
            )
        ):
            raise ValueError(
                "bounded drain fields require "
                "scheduler_policy=recompute_aware_bounded"
            )
        if self.scheduler_policy == "chunked_prefill_budgeted":
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
        elif self.prefill_chunk_token_budget != 0:
            raise ValueError(
                "prefill_chunk_token_budget must be 0 unless "
                "scheduler_policy is chunked_prefill_budgeted"
            )
        if not isinstance(self.cuda_event_timing, bool):
            raise ValueError("cuda_event_timing must be a boolean")
        if not isinstance(self.incremental_kv_allocation, bool):
            raise ValueError("incremental_kv_allocation must be a boolean")
        if (
            not isinstance(self.distributed_init_port, int)
            or isinstance(self.distributed_init_port, bool)
            or not 1024 <= self.distributed_init_port <= 65535
        ):
            raise ValueError("distributed_init_port must be in [1024, 65535]")
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)

    def _validate_pressure_aware_fields(self) -> None:
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
            (
                "pressure_preemption_threshold",
                self.pressure_preemption_threshold,
            ),
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
