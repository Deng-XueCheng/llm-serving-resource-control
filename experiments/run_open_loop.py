from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import platform
import random
import re
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from contextlib import nullcontext
from math import isfinite
from pathlib import Path
from typing import Any

import torch
import flash_attn
import transformers
import triton
from nanovllm import LLM, SamplingParams

from experiments.benchmark.lifecycle import (
    RequestRecord,
    TerminalState,
    summarize_requests,
)
from experiments.benchmark.open_loop import (
    AdmissionConfig,
    OpenLoopResult,
    load_trace,
    prepare_requests,
    run_open_loop,
)
from experiments.snapshot_preflight import (
    LEGACY_UPSTREAM_COMMIT,
    verify_source_snapshot,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "experiments/results"
TOP_LEVEL_KEYS = {
    "schema_version",
    "upstream_commit",
    "model",
    "engine",
    "sampling",
    "workload",
    "slo",
    "measurement",
    "warmup",
    "profiling",
    "admission",
    "output",
}
MODEL_KEYS = {"path", "repo_id", "revision", "sha256"}
ENGINE_REQUIRED_KEYS = {
    "enforce_eager",
    "tensor_parallel_size",
    "max_model_len",
    "max_num_batched_tokens",
    "max_num_seqs",
    "gpu_memory_utilization",
    "kvcache_block_size",
}
ENGINE_OPTIONAL_KEYS = {
    "scheduler_policy",
    "decode_token_budget",
    "decode_step_guard",
    "num_kvcache_blocks",
    "cuda_event_timing",
    "pressure_decode_token_budget",
    "pressure_decode_step_guard",
    "pressure_high_utilization",
    "pressure_critical_utilization",
    "pressure_preemption_window",
    "pressure_preemption_threshold",
    "pressure_hysteresis_steps",
    "pressure_waiting_age_threshold",
    "prefill_chunk_token_budget",
    "max_drain_steps",
    "waiting_age_limit",
    "ttft_slo_ms",
    "itl_slo_ms",
    "mixed_min_prefill_tokens",
    "mixed_waiting_age_threshold",
    "mixed_slack_threshold_ms",
    "distributed_init_port",
    "incremental_kv_allocation",
}
SAMPLING_KEYS = {"temperature", "ignore_eos", "seed"}
WORKLOAD_KEYS = {"trace_path", "token_id_upper_bound"}
SLO_KEYS = {"ttft_slo_ms", "itl_slo_ms", "require_itl"}
MEASUREMENT_KEYS = {"start_seconds", "end_seconds", "max_run_seconds"}
WARMUP_KEYS = {
    "enabled",
    "prompt_length",
    "max_output_tokens",
    "seed",
    "batch_sizes",
}
OUTPUT_KEYS = {"directory", "run_id"}
PROFILING_KEYS = {
    "enabled",
    "output_directory",
    "record_shapes",
    "profile_memory",
    "with_stack",
    "wait_steps",
    "warmup_steps",
    "active_steps",
    "repeat",
    "cuda_events",
}
ADMISSION_KEYS = {
    "policy",
    "max_queue_wait_seconds",
    "observe_prefix_cache",
    "eta_prefill_seconds",
    "eta_decode_seconds_per_token",
    "eta_safety_margin_seconds",
}
RUN_LIMITATIONS = (
    "Single run; aggregate matched repeated seeds before drawing conclusions.",
    "This artifact contains one policy cell; compare policies only through "
    "hash-verified matched aggregation.",
    "Global Torch RNG seed is fixed and ignore_eos enforces fixed output "
    "lengths; output token identities are not paired across different "
    "scheduling orders.",
)


def validate_exact_keys(
    name: str,
    value: Any,
    expected: set[str],
) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    missing = sorted(expected - value.keys())
    unknown = sorted(value.keys() - expected)
    if missing or unknown:
        raise ValueError(
            f"{name} keys mismatch; missing={missing}, unknown={unknown}"
        )


def validate_engine_keys(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("engine must be an object")
    allowed = ENGINE_REQUIRED_KEYS | ENGINE_OPTIONAL_KEYS
    missing = sorted(ENGINE_REQUIRED_KEYS - value.keys())
    unknown = sorted(value.keys() - allowed)
    if missing or unknown:
        raise ValueError(
            f"engine keys mismatch; missing={missing}, unknown={unknown}"
        )


def require_positive_number(name: str, value: Any) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be positive")


def require_positive_integer(name: str, value: Any) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        config = json.load(file)
    config.setdefault(
        "profiling",
        {
            "enabled": False,
            "output_directory": "experiments/profiling",
            "record_shapes": True,
            "profile_memory": True,
            "with_stack": False,
            "wait_steps": 0,
            "warmup_steps": 1,
            "active_steps": 3,
            "repeat": 1,
            "cuda_events": False,
        },
    )
    config.setdefault(
        "admission",
        {"policy": "disabled", "max_queue_wait_seconds": 0.0},
    )
    if isinstance(config["admission"], dict):
        config["admission"].setdefault("observe_prefix_cache", False)
        config["admission"].setdefault("eta_prefill_seconds", 0.0)
        config["admission"].setdefault("eta_decode_seconds_per_token", 0.0)
        config["admission"].setdefault("eta_safety_margin_seconds", 0.0)
    validate_exact_keys("config", config, TOP_LEVEL_KEYS)
    validate_exact_keys("model", config["model"], MODEL_KEYS)
    validate_engine_keys(config["engine"])
    validate_exact_keys("sampling", config["sampling"], SAMPLING_KEYS)
    validate_exact_keys("workload", config["workload"], WORKLOAD_KEYS)
    validate_exact_keys("slo", config["slo"], SLO_KEYS)
    validate_exact_keys(
        "measurement",
        config["measurement"],
        MEASUREMENT_KEYS,
    )
    validate_exact_keys("warmup", config["warmup"], WARMUP_KEYS)
    validate_exact_keys("profiling", config["profiling"], PROFILING_KEYS)
    validate_exact_keys("admission", config["admission"], ADMISSION_KEYS)
    validate_exact_keys("output", config["output"], OUTPUT_KEYS)

    if config["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    upstream_commit = config["upstream_commit"]
    if (
        not isinstance(upstream_commit, str)
        or len(upstream_commit) != 40
        or any(character not in "0123456789abcdef" for character in upstream_commit)
    ):
        raise ValueError("upstream_commit must be a lowercase 40-character SHA")

    model = config["model"]
    model_path = Path(model["path"])
    if not model_path.is_absolute():
        raise ValueError("model.path must be absolute")
    for field in ("repo_id", "revision"):
        if not isinstance(model[field], str) or not model[field]:
            raise ValueError(f"model.{field} must be a non-empty string")
    if not isinstance(model["sha256"], dict) or not model["sha256"]:
        raise ValueError("model.sha256 must be a non-empty object")
    for filename, digest in model["sha256"].items():
        if Path(filename).name != filename:
            raise ValueError(f"Unsafe model hash filename: {filename}")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"Invalid model SHA-256 for {filename}")

    engine = config["engine"]
    engine.setdefault("scheduler_policy", "prefill_first")
    engine.setdefault("decode_token_budget", 0)
    engine.setdefault("decode_step_guard", 0)
    engine.setdefault("cuda_event_timing", config["profiling"]["cuda_events"])
    if not isinstance(engine["enforce_eager"], bool):
        raise ValueError("engine.enforce_eager must be a boolean")
    if not isinstance(engine["cuda_event_timing"], bool):
        raise ValueError("engine.cuda_event_timing must be a boolean")
    engine.setdefault("incremental_kv_allocation", False)
    if not isinstance(engine["incremental_kv_allocation"], bool):
        raise ValueError("engine.incremental_kv_allocation must be a boolean")
    if engine["cuda_event_timing"] != config["profiling"]["cuda_events"]:
        raise ValueError(
            "engine.cuda_event_timing must match profiling.cuda_events"
        )
    for field in (
        "tensor_parallel_size",
        "max_model_len",
        "max_num_batched_tokens",
        "max_num_seqs",
        "kvcache_block_size",
    ):
        require_positive_integer(f"engine.{field}", engine[field])
    if engine["tensor_parallel_size"] != 1:
        raise ValueError("benchmark v1 supports tensor_parallel_size=1 only")
    gpu_utilization = engine["gpu_memory_utilization"]
    require_positive_number(
        "engine.gpu_memory_utilization",
        gpu_utilization,
    )
    if gpu_utilization > 1:
        raise ValueError("engine.gpu_memory_utilization must not exceed 1")
    scheduler_policy = engine["scheduler_policy"]
    if (
        not isinstance(scheduler_policy, str)
        or scheduler_policy
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
            "engine.scheduler_policy must be prefill_first, "
            "decode_first_budgeted, pressure_aware_decode, "
            "chunked_prefill_budgeted, recompute_aware, or "
            "recompute_aware_bounded, mixed_token_budget, or mixed_slo_budget"
        )
    decode_token_budget = engine["decode_token_budget"]
    if scheduler_policy in {
        "decode_first_budgeted",
        "pressure_aware_decode",
        "chunked_prefill_budgeted",
        "recompute_aware",
        "recompute_aware_bounded",
        "mixed_token_budget",
        "mixed_slo_budget",
    }:
        require_positive_integer(
            "engine.decode_token_budget",
            decode_token_budget,
        )
    elif (
        not isinstance(decode_token_budget, int)
        or isinstance(decode_token_budget, bool)
        or decode_token_budget != 0
    ):
        raise ValueError(
            "engine.decode_token_budget must be 0 for prefill_first"
        )
    decode_step_guard = engine["decode_step_guard"]
    if (
        not isinstance(decode_step_guard, int)
        or isinstance(decode_step_guard, bool)
        or decode_step_guard < 0
    ):
        raise ValueError(
            "engine.decode_step_guard must be a non-negative integer"
        )
    if scheduler_policy == "prefill_first" and decode_step_guard != 0:
        raise ValueError(
            "engine.decode_step_guard must be 0 for prefill_first"
        )
    if "num_kvcache_blocks" in engine:
        require_positive_integer(
            "engine.num_kvcache_blocks",
            engine["num_kvcache_blocks"],
        )

    if scheduler_policy in {
        "pressure_aware_decode",
        "chunked_prefill_budgeted",
        "recompute_aware",
        "recompute_aware_bounded",
    }:
        validate_pressure_aware_engine(engine)
    elif any(field.startswith("pressure_") for field in engine):
        raise ValueError(
            "pressure_* fields are only valid for pressure-aware policies"
        )
    if scheduler_policy == "chunked_prefill_budgeted":
        if "prefill_chunk_token_budget" not in engine:
            raise ValueError(
                "chunked_prefill_budgeted requires "
                "engine.prefill_chunk_token_budget"
            )
        require_positive_integer(
            "engine.prefill_chunk_token_budget",
            engine["prefill_chunk_token_budget"],
        )
        if (
            engine["prefill_chunk_token_budget"]
            > engine["max_num_batched_tokens"]
        ):
            raise ValueError(
                "engine.prefill_chunk_token_budget must not exceed "
                "engine.max_num_batched_tokens"
            )
    elif "prefill_chunk_token_budget" in engine:
        raise ValueError(
            "engine.prefill_chunk_token_budget is only valid for "
            "chunked_prefill_budgeted"
        )
    bounded_fields = {
        "max_drain_steps",
        "waiting_age_limit",
        "ttft_slo_ms",
        "itl_slo_ms",
    }
    if scheduler_policy == "recompute_aware_bounded":
        missing = sorted(bounded_fields - engine.keys())
        if missing:
            raise ValueError(
                "recompute_aware_bounded requires engine fields: "
                + ", ".join(missing)
            )
        require_positive_integer(
            "engine.max_drain_steps",
            engine["max_drain_steps"],
        )
        require_positive_integer(
            "engine.waiting_age_limit",
            engine["waiting_age_limit"],
        )
        for name in ("ttft_slo_ms", "itl_slo_ms"):
            require_positive_number(f"engine.{name}", engine[name])
            if engine[name] != config["slo"][name]:
                raise ValueError(
                    f"engine.{name} must match slo.{name}"
                )
    elif scheduler_policy == "mixed_slo_budget":
        required = {
            "ttft_slo_ms",
            "itl_slo_ms",
            "mixed_min_prefill_tokens",
            "mixed_waiting_age_threshold",
            "mixed_slack_threshold_ms",
        }
        missing = sorted(required - engine.keys())
        if missing:
            raise ValueError(
                "mixed_slo_budget requires engine fields: "
                + ", ".join(missing)
            )
        require_positive_integer(
            "engine.mixed_min_prefill_tokens",
            engine["mixed_min_prefill_tokens"],
        )
        require_positive_integer(
            "engine.mixed_waiting_age_threshold",
            engine["mixed_waiting_age_threshold"],
        )
        require_positive_number(
            "engine.mixed_slack_threshold_ms",
            engine["mixed_slack_threshold_ms"],
        )
        for name in ("ttft_slo_ms", "itl_slo_ms"):
            require_positive_number(f"engine.{name}", engine[name])
            if engine[name] != config["slo"][name]:
                raise ValueError(f"engine.{name} must match slo.{name}")
    elif bounded_fields & engine.keys():
        raise ValueError(
            "bounded drain fields are only valid for "
            "recompute_aware_bounded"
        )

    sampling = config["sampling"]
    require_positive_number(
        "sampling.temperature",
        sampling["temperature"],
    )
    if not isinstance(sampling["ignore_eos"], bool):
        raise ValueError("sampling.ignore_eos must be a boolean")
    if not sampling["ignore_eos"]:
        raise ValueError(
            "benchmark v1 requires sampling.ignore_eos=true "
            "to keep output lengths matched across policies"
        )
    if (
        not isinstance(sampling["seed"], int)
        or isinstance(sampling["seed"], bool)
        or sampling["seed"] < 0
    ):
        raise ValueError("sampling.seed must be a non-negative integer")

    require_positive_integer(
        "workload.token_id_upper_bound",
        config["workload"]["token_id_upper_bound"],
    )
    for field in SLO_KEYS:
        if field != "require_itl":
            require_positive_number(f"slo.{field}", config["slo"][field])
    if not isinstance(config["slo"]["require_itl"], bool):
        raise ValueError("slo.require_itl must be a boolean")

    measurement = config["measurement"]
    if (
        not isinstance(measurement["start_seconds"], (int, float))
        or isinstance(measurement["start_seconds"], bool)
        or not isfinite(measurement["start_seconds"])
        or measurement["start_seconds"] < 0
    ):
        raise ValueError("measurement.start_seconds must be non-negative")
    require_positive_number(
        "measurement.end_seconds",
        measurement["end_seconds"],
    )
    require_positive_number(
        "measurement.max_run_seconds",
        measurement["max_run_seconds"],
    )
    if measurement["end_seconds"] <= measurement["start_seconds"]:
        raise ValueError("measurement end must be greater than start")
    if measurement["max_run_seconds"] < measurement["end_seconds"]:
        raise ValueError("max_run_seconds must cover the measurement window")

    warmup = config["warmup"]
    if not isinstance(warmup["enabled"], bool):
        raise ValueError("warmup.enabled must be a boolean")
    for field in ("prompt_length", "max_output_tokens"):
        require_positive_integer(f"warmup.{field}", warmup[field])
    if not isinstance(warmup["seed"], int) or isinstance(warmup["seed"], bool):
        raise ValueError("warmup.seed must be an integer")
    batch_sizes = warmup["batch_sizes"]
    if (
        not isinstance(batch_sizes, list)
        or not batch_sizes
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            for value in batch_sizes
        )
        or len(batch_sizes) != len(set(batch_sizes))
    ):
        raise ValueError(
            "warmup.batch_sizes must be unique positive integers"
        )
    if any(value > engine["max_num_seqs"] for value in batch_sizes):
        raise ValueError("warmup batch size exceeds engine.max_num_seqs")

    validate_profiling_config(config["profiling"])
    validate_admission_config(config["admission"], engine)

    trace_path = resolve_repo_path(config["workload"]["trace_path"])
    resolved_outputs = list(
        resolve_output_paths(
            config["output"],
            include_phase_timings=config["profiling"]["cuda_events"],
            include_admission_events=(
                config["admission"]["policy"]
                in {
                    "kv_aware_fifo",
                    "prefix_aware_fifo",
                    "slack_aware_prefix_fifo",
                }
            ),
            include_cache_events=config["admission"]["observe_prefix_cache"],
            include_cache_states=config["admission"]["observe_prefix_cache"],
            include_slack_events=(
                config["admission"]["policy"] == "slack_aware_prefix_fifo"
            ),
        ).values()
    )
    all_paths = [path.resolve(), trace_path, *resolved_outputs]
    if len(all_paths) != len(set(all_paths)):
        raise ValueError(
            "Config, trace, and output paths must be globally unique"
        )
    return config


def validate_pressure_aware_engine(engine: dict[str, Any]) -> None:
    required = {
        "pressure_decode_token_budget",
        "pressure_decode_step_guard",
        "pressure_high_utilization",
        "pressure_critical_utilization",
        "pressure_preemption_window",
        "pressure_preemption_threshold",
        "pressure_hysteresis_steps",
        "pressure_waiting_age_threshold",
    }
    missing = sorted(required - engine.keys())
    if missing:
        raise ValueError(
            "pressure_aware_decode requires engine fields: "
            + ", ".join(missing)
        )
    if (
        not isinstance(engine["pressure_decode_token_budget"], int)
        or isinstance(engine["pressure_decode_token_budget"], bool)
        or engine["pressure_decode_token_budget"] <= 0
        or engine["pressure_decode_token_budget"]
        > engine["decode_token_budget"]
    ):
        raise ValueError("pressure_decode_token_budget must not exceed normal budget")
    for field in (
        "pressure_preemption_window",
        "pressure_preemption_threshold",
        "pressure_hysteresis_steps",
        "pressure_waiting_age_threshold",
    ):
        require_positive_integer(
            f"engine.{field}",
            engine[field],
        )
    if (
        not isinstance(engine["pressure_decode_step_guard"], int)
        or isinstance(engine["pressure_decode_step_guard"], bool)
        or engine["pressure_decode_step_guard"] < 0
    ):
        raise ValueError(
            "engine.pressure_decode_step_guard must be non-negative"
        )
    require_positive_number(
        "engine.pressure_high_utilization",
        engine["pressure_high_utilization"],
    )
    if engine["pressure_high_utilization"] >= 1:
        raise ValueError(
            "engine.pressure_high_utilization must be below 1"
        )
    require_positive_number(
        "engine.pressure_critical_utilization",
        engine["pressure_critical_utilization"],
    )
    if engine["pressure_critical_utilization"] > 1:
        raise ValueError(
            "engine.pressure_critical_utilization must not exceed 1"
        )
    if (
        engine["pressure_high_utilization"]
        > engine["pressure_critical_utilization"]
    ):
        raise ValueError(
            "pressure_critical_utilization must be >= high_utilization"
        )


def validate_profiling_config(profiling: dict[str, Any]) -> None:
    if not isinstance(profiling["enabled"], bool):
        raise ValueError("profiling.enabled must be a boolean")
    if not isinstance(profiling["output_directory"], str):
        raise ValueError("profiling.output_directory must be a string")
    resolve_repo_path(profiling["output_directory"])
    for field in ("record_shapes", "profile_memory", "with_stack", "cuda_events"):
        if not isinstance(profiling[field], bool):
            raise ValueError(f"profiling.{field} must be a boolean")
    for field in (
        "wait_steps",
        "warmup_steps",
        "active_steps",
        "repeat",
    ):
        if (
            not isinstance(profiling[field], int)
            or isinstance(profiling[field], bool)
            or profiling[field] < 0
        ):
            raise ValueError(f"profiling.{field} must be a non-negative integer")
    if profiling["active_steps"] <= 0:
        raise ValueError("profiling.active_steps must be positive")
    if profiling["repeat"] <= 0:
        raise ValueError("profiling.repeat must be positive")


def validate_admission_config(
    admission: dict[str, Any],
    engine: dict[str, Any],
) -> None:
    if admission["policy"] not in {
        "disabled",
        "kv_aware_fifo",
        "prefix_aware_fifo",
        "slack_aware_prefix_fifo",
    }:
        raise ValueError(
            "admission.policy must be disabled, kv_aware_fifo, "
            "prefix_aware_fifo, or slack_aware_prefix_fifo"
        )
    observe_prefix_cache = admission.get("observe_prefix_cache", False)
    if not isinstance(observe_prefix_cache, bool):
        raise ValueError("admission.observe_prefix_cache must be a boolean")
    wait = admission["max_queue_wait_seconds"]
    if admission["policy"] == "disabled":
        if (
            not isinstance(wait, (int, float))
            or isinstance(wait, bool)
            or not isfinite(float(wait))
            or wait != 0
            or any(
                admission[field] != 0
                for field in (
                    "eta_prefill_seconds",
                    "eta_decode_seconds_per_token",
                    "eta_safety_margin_seconds",
                )
            )
        ):
            raise ValueError("disabled admission requires zero queue wait and ETA")
        return
    if (
        admission["policy"] in {"prefix_aware_fifo", "slack_aware_prefix_fifo"}
        and not observe_prefix_cache
    ):
        raise ValueError(
            "prefix-aware admission requires prefix cache observation"
        )
    require_positive_number("admission.max_queue_wait_seconds", wait)
    eta_fields = (
        "eta_prefill_seconds",
        "eta_decode_seconds_per_token",
        "eta_safety_margin_seconds",
    )
    if admission["policy"] == "slack_aware_prefix_fifo":
        for field in eta_fields:
            require_positive_number(f"admission.{field}", admission[field])
    elif any(admission[field] != 0 for field in eta_fields):
        raise ValueError("ETA fields are only valid for slack-aware admission")
    if "num_kvcache_blocks" not in engine:
        raise ValueError("KV-aware admission requires num_kvcache_blocks")


def build_pytorch_profiler(
    profiling: dict[str, Any],
    *,
    run_id: str,
) -> tuple[Any, Path | None]:
    if not profiling["enabled"]:
        return nullcontext(None), None
    if not torch.cuda.is_available():
        raise RuntimeError("profiling.enabled requires CUDA")
    trace_directory = (
        resolve_repo_path(profiling["output_directory"]) / run_id
    ).resolve()
    if trace_directory.exists() and any(trace_directory.iterdir()):
        raise FileExistsError(
            "Profiler output would overwrite existing traces: "
            f"{trace_directory}"
        )
    trace_directory.mkdir(parents=True, exist_ok=True)
    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=profiling["wait_steps"],
            warmup=profiling["warmup_steps"],
            active=profiling["active_steps"],
            repeat=profiling["repeat"],
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            str(trace_directory)
        ),
        record_shapes=profiling["record_shapes"],
        profile_memory=profiling["profile_memory"],
        with_stack=profiling["with_stack"],
    )
    return profiler, trace_directory


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError("Repository artifact paths must be relative")
    resolved = (REPO_ROOT / path).resolve()
    if not resolved.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError(f"Path leaves repository: {value}")
    return resolved


def resolve_result_path(value: str, suffix: str) -> Path:
    resolved = resolve_repo_path(value)
    if not resolved.is_relative_to(RESULTS_ROOT.resolve()):
        raise ValueError("Result paths must be inside experiments/results")
    if resolved.suffix != suffix:
        raise ValueError(f"Result path must end in {suffix}: {value}")
    return resolved


def resolve_output_paths(
    output: dict[str, Any],
    *,
    include_phase_timings: bool = False,
    include_admission_events: bool = False,
    include_cache_events: bool = False,
    include_cache_states: bool = False,
    include_slack_events: bool = False,
) -> dict[str, Path]:
    run_id = output["run_id"]
    if (
        not isinstance(run_id, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id)
        or ".." in run_id
    ):
        raise ValueError("output.run_id contains unsafe characters")
    directory = resolve_repo_path(output["directory"])
    if not directory.is_relative_to(RESULTS_ROOT.resolve()):
        raise ValueError("output.directory must be inside experiments/results")
    paths = {
        "summary": directory / f"{run_id}.summary.json",
        "requests": directory / f"{run_id}.requests.jsonl",
        "steps": directory / f"{run_id}.steps.jsonl",
    }
    if include_phase_timings:
        paths["phase_timings"] = (
            directory / f"{run_id}.phase_timings.jsonl"
        )
    if include_admission_events:
        paths["admission_events"] = (
            directory / f"{run_id}.admission.jsonl"
        )
    if include_cache_events:
        paths["cache_events"] = directory / f"{run_id}.cache.jsonl"
    if include_cache_states:
        paths["cache_states"] = directory / f"{run_id}.cache_states.jsonl"
    if include_slack_events:
        paths["slack_events"] = directory / f"{run_id}.slack.jsonl"
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Run ID would overwrite existing artifacts: "
            + ", ".join(existing)
        )
    return paths


def required_kv_blocks(
    prompt_length: int,
    max_output_tokens: int,
    block_size: int,
) -> int:
    cache_tokens = prompt_length + max_output_tokens - 1
    return (cache_tokens + block_size - 1) // block_size


def require_kv_footprint(
    label: str,
    prompt_length: int,
    max_output_tokens: int,
    *,
    block_size: int,
    total_blocks: int,
) -> None:
    required_blocks = required_kv_blocks(
        prompt_length,
        max_output_tokens,
        block_size,
    )
    if required_blocks > total_blocks:
        raise ValueError(
            f"{label} requires {required_blocks} KV blocks "
            f"but engine provides {total_blocks}"
        )


def validate_kv_request_feasibility(
    specs,
    engine: dict[str, Any],
    *,
    warmup: dict[str, Any] | None = None,
    admission: dict[str, Any] | None = None,
) -> None:
    total_blocks = engine.get("num_kvcache_blocks")
    if total_blocks is None:
        return
    block_size = engine["kvcache_block_size"]
    for spec in specs:
        if (
            admission is not None
            and admission["policy"]
            in {"prefix_aware_fifo", "slack_aware_prefix_fifo"}
            and spec.prefix_group is not None
            and spec.shared_prefix_length % block_size != 0
        ):
            raise ValueError(
                f"Request {spec.request_id} shared_prefix_length must be "
                "block-aligned for prefix-aware admission"
            )
        require_kv_footprint(
            f"Request {spec.request_id}",
            spec.prompt_length,
            spec.max_output_tokens,
            block_size=block_size,
            total_blocks=total_blocks,
        )
    if warmup is not None and warmup["enabled"]:
        require_kv_footprint(
            "warmup",
            warmup["prompt_length"],
            warmup["max_output_tokens"],
            block_size=block_size,
            total_blocks=total_blocks,
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def python_package_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(directory.rglob("*.py"))
    if not paths:
        raise ValueError(f"No Python files found under {directory}")
    for path in paths:
        relative = path.relative_to(directory).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def verify_model_files(
    model_path: Path,
    expected_hashes: dict[str, str],
) -> dict[str, str]:
    actual = {}
    for filename, expected in expected_hashes.items():
        path = model_path / filename
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
        digest = file_sha256(path)
        if digest != expected:
            raise RuntimeError(
                f"Model hash mismatch for {filename}: "
                f"expected {expected}, got {digest}"
            )
        actual[filename] = digest
    return actual


def git_output(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def verify_upstream_commit(upstream_commit: str) -> None:
    if upstream_commit != LEGACY_UPSTREAM_COMMIT:
        raise RuntimeError(
            "Configured upstream commit does not match the clean snapshot "
            f"lineage: expected {LEGACY_UPSTREAM_COMMIT}, got {upstream_commit}"
        )
    verify_source_snapshot()


def runtime_metadata() -> dict[str, Any]:
    metadata_errors = []
    try:
        driver_version = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip().splitlines()[0]
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError) as exception:
        driver_version = None
        metadata_errors.append(f"nvidia_driver: {exception}")
    try:
        cuda_available = torch.cuda.is_available()
    except Exception as exception:
        cuda_available = False
        metadata_errors.append(f"cuda_available: {exception}")
    gpu_name = None
    gpu_capability = None
    if cuda_available:
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_capability = list(torch.cuda.get_device_capability(0))
        except Exception as exception:
            cuda_available = False
            metadata_errors.append(f"gpu_metadata: {exception}")
    lock_hashes = {}
    for name, path in (
        (
            "requirements_lock_sha256",
            REPO_ROOT / "reproduction/requirements.lock.txt",
        ),
        ("pylock_sha256", REPO_ROOT / "reproduction/pylock.toml"),
    ):
        try:
            lock_hashes[name] = file_sha256(path)
        except Exception as exception:
            lock_hashes[name] = None
            metadata_errors.append(f"{name}: {exception}")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": cuda_available,
        "gpu": gpu_name,
        "gpu_capability": gpu_capability,
        "nvidia_driver": driver_version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "triton": triton.__version__,
        "flash_attn": flash_attn.__version__,
        **lock_hashes,
        "metadata_errors": metadata_errors,
    }


def reset_pressure_observability(llm: LLM) -> None:
    torch.cuda.synchronize()
    llm.reset_observability()
    torch.cuda.reset_peak_memory_stats()


def gpu_memory_snapshot() -> dict[str, int]:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "allocated_bytes_current": torch.cuda.memory_allocated(),
        "allocated_bytes_peak": torch.cuda.max_memory_allocated(),
        "reserved_bytes_current": torch.cuda.memory_reserved(),
        "reserved_bytes_peak": torch.cuda.max_memory_reserved(),
        "device_free_bytes": free_bytes,
        "device_total_bytes": total_bytes,
    }


def exception_record(exception: Exception) -> dict[str, Any]:
    return {
        "type": type(exception).__name__,
        "message": str(exception),
        "is_cuda_oom": isinstance(
            exception,
            torch.cuda.OutOfMemoryError,
        ),
    }


def error_has_cuda_oom(error: Any) -> bool:
    if not isinstance(error, dict):
        return False
    if error.get("is_cuda_oom") is True:
        return True
    return any(error_has_cuda_oom(value) for value in error.values())


def fail_result(
    result: OpenLoopResult,
    secondary_error: dict[str, Any],
    *,
    label: str,
) -> OpenLoopResult:
    if result.error is None:
        error = {
            "type": secondary_error["type"],
            "message": f"{label}: {secondary_error['message']}",
            "is_cuda_oom": error_has_cuda_oom(secondary_error),
            label: secondary_error,
        }
    else:
        error = {
            **result.error,
            label: secondary_error,
            "is_cuda_oom": (
                error_has_cuda_oom(result.error)
                or error_has_cuda_oom(secondary_error)
            ),
        }
    return OpenLoopResult(
        status="failed",
        records=result.records,
        step_events=result.step_events,
        summary=result.summary,
        error=error,
        phase_timings=result.phase_timings,
        admission=result.admission,
        admission_events=result.admission_events,
    )


def build_pressure_summary(
    llm,
    *,
    terminal_counts: dict[str, Any],
    error: dict[str, Any] | None,
    gpu_memory: dict[str, int] | None,
    observation_active: bool,
    admission_rejection_supported: bool,
    collection_error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    engine = llm.observability_snapshot() if llm is not None else None
    return {
        "schema_version": 1,
        "observation_active": observation_active,
        "observation_complete": (
            observation_active
            and engine is not None
            and gpu_memory is not None
            and collection_error is None
        ),
        "kv_cache": None if engine is None else engine["kv_cache"],
        "preemption_count": (
            None if engine is None else engine["preemption_count"]
        ),
        "gpu_memory": gpu_memory,
        "rejected_requests": int(terminal_counts["Rejected"]),
        "admission_rejection_supported": admission_rejection_supported,
        "oom_detected": (
            error_has_cuda_oom(error)
            or error_has_cuda_oom(collection_error)
        ),
        "collection_error": collection_error,
    }


def build_pressure_aware_summary(llm) -> dict[str, Any] | None:
    if llm is None:
        return None
    return llm.observability_snapshot().get("pressure_aware")


def build_chunked_prefill_summary(llm) -> dict[str, Any] | None:
    if llm is None:
        return None
    return llm.observability_snapshot().get(
        "chunked_prefill_budgeted"
    )


def build_recompute_aware_summary(llm) -> dict[str, Any] | None:
    if llm is None:
        return None
    snapshot = llm.observability_snapshot()
    policy = snapshot.get("recompute_aware")
    if policy is None:
        return None
    return {
        **policy,
        "actual_recompute_tokens": snapshot["actual_recompute_tokens"],
        "resume_count": snapshot["resume_count"],
    }


def build_bounded_recompute_summary(llm) -> dict[str, Any] | None:
    if llm is None:
        return None
    snapshot = llm.observability_snapshot()
    policy = snapshot.get("recompute_aware_bounded")
    if policy is None:
        return None
    return {
        **policy,
        "actual_recompute_tokens": snapshot["actual_recompute_tokens"],
        "resume_count": snapshot["resume_count"],
    }


def build_recompute_observability(llm) -> dict[str, Any] | None:
    if llm is None:
        return None
    snapshot = llm.observability_snapshot()
    return {
        "schema_version": 1,
        "preemption_count": snapshot["preemption_count"],
        "actual_recompute_tokens": snapshot["actual_recompute_tokens"],
        "resume_count": snapshot["resume_count"],
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


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
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
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False))
            file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    temporary_path.replace(path)


def startup_failure_result(
    specs,
    config: dict[str, Any],
    exception: Exception,
) -> OpenLoopResult:
    error = exception_record(exception)
    records = [
        RequestRecord(
            request_id=spec.request_id,
            request_class=spec.request_class,
            arrival_at=spec.arrival_time,
        )
        for spec in specs
    ]
    reason = f"{error['type']}: {error['message']}"
    for record in records:
        record.mark_terminal(
            TerminalState.CANCELLED,
            record.arrival_at,
            reason=reason,
        )
    summary = summarize_requests(
        records,
        measurement_start=config["measurement"]["start_seconds"],
        measurement_end=config["measurement"]["end_seconds"],
        ttft_slo_ms=config["slo"]["ttft_slo_ms"],
        itl_slo_ms=config["slo"]["itl_slo_ms"],
        require_itl=config["slo"]["require_itl"],
    )
    summary["runtime"] = {
        "elapsed_seconds": 0.0,
        "steps": 0,
        "timed_out": False,
    }
    return OpenLoopResult(
        status="failed",
        records=records,
        step_events=[],
        summary=summary,
        error=error,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "experiments/configs/open_loop_smoke.json",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Override the model path recorded in the frozen config.",
    )
    parser.add_argument(
        "--stage12-manifest",
        type=Path,
        help=(
            "Pre-run Stage 12 smoke manifest. When supplied, config, trace, "
            "execution code and calibration evidence are verified before CUDA work."
        ),
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    if args.model_path is not None:
        config["model"]["path"] = str(args.model_path.resolve())
    stage12_manifest = None
    if config["admission"]["policy"] == "slack_aware_prefix_fifo":
        if args.stage12_manifest is None:
            raise ValueError("slack-aware Stage 12 runs require --stage12-manifest")
        from experiments.aggregate_stage12 import load_frozen_manifest

        manifest_path = args.stage12_manifest.resolve()
        load_frozen_manifest(
            manifest_path=manifest_path,
            config_path=config_path,
        )
        stage12_manifest = {
            "path": str(manifest_path),
            "sha256": file_sha256(manifest_path),
        }
    elif args.stage12_manifest is not None:
        raise ValueError("--stage12-manifest is only valid for slack-aware Stage 12 runs")
    verify_upstream_commit(config["upstream_commit"])

    trace_path = resolve_repo_path(config["workload"]["trace_path"])
    specs = load_trace(trace_path)
    max_model_len = config["engine"]["max_model_len"]
    for spec in specs:
        if spec.prompt_length + spec.max_output_tokens > max_model_len:
            raise ValueError(
                f"Request {spec.request_id} exceeds max_model_len"
            )
        if (
            config["slo"]["require_itl"]
            and spec.request_class == "interactive"
            and spec.max_output_tokens < 2
        ):
            raise ValueError(
                f"Interactive request {spec.request_id} must emit at least "
                "two tokens when ITL is required"
            )
    validate_kv_request_feasibility(
        specs,
        config["engine"],
        warmup=config["warmup"],
        admission=config["admission"],
    )
    prepared = prepare_requests(
        specs,
        token_id_upper_bound=config["workload"]["token_id_upper_bound"],
    )
    output_paths = resolve_output_paths(
        config["output"],
        include_phase_timings=config["profiling"]["cuda_events"],
        include_admission_events=(
            config["admission"]["policy"]
                in {
                    "kv_aware_fifo",
                    "prefix_aware_fifo",
                    "slack_aware_prefix_fifo",
                }
        ),
        include_cache_events=config["admission"]["observe_prefix_cache"],
        include_cache_states=config["admission"]["observe_prefix_cache"],
        include_slack_events=(
            config["admission"]["policy"] == "slack_aware_prefix_fifo"
        ),
    )
    profiler_context, profiler_trace_directory = build_pytorch_profiler(
        config["profiling"],
        run_id=config["output"]["run_id"],
    )

    model_path = Path(config["model"]["path"])
    model_hashes = None
    llm = None
    observation_active = False
    gpu_memory = None
    pressure_collection_error = None
    pressure_aware = None
    chunked_prefill_budgeted = None
    recompute_aware = None
    recompute_aware_bounded = None
    recompute_observability = None
    try:
        if not model_path.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        model_hashes = verify_model_files(
            model_path,
            config["model"]["sha256"],
        )
        llm = LLM(str(model_path), **config["engine"])
        warmup = config["warmup"]
        if warmup["enabled"]:
            generator = random.Random(warmup["seed"])
            for batch_size in warmup["batch_sizes"]:
                warmup_prompts = [
                    [
                        generator.randrange(
                            1,
                            config["workload"]["token_id_upper_bound"],
                        )
                        for _ in range(warmup["prompt_length"])
                    ]
                    for _ in range(batch_size)
                ]
                llm.generate(
                    warmup_prompts,
                    SamplingParams(
                        temperature=config["sampling"]["temperature"],
                        max_tokens=warmup["max_output_tokens"],
                        ignore_eos=True,
                    ),
                    use_tqdm=False,
                )

        torch.manual_seed(config["sampling"]["seed"])
        torch.cuda.manual_seed_all(config["sampling"]["seed"])
        reset_pressure_observability(llm)
        observation_active = True

        with profiler_context as profiler:
            result = run_open_loop(
                llm,
                prepared,
                temperature=config["sampling"]["temperature"],
                ignore_eos=config["sampling"]["ignore_eos"],
                measurement_start=config["measurement"]["start_seconds"],
                measurement_end=config["measurement"]["end_seconds"],
                ttft_slo_ms=config["slo"]["ttft_slo_ms"],
                itl_slo_ms=config["slo"]["itl_slo_ms"],
                require_itl=config["slo"]["require_itl"],
                max_run_seconds=config["measurement"]["max_run_seconds"],
                synchronize=torch.cuda.synchronize,
                profiler=profiler,
                admission=AdmissionConfig(
                    policy=config["admission"]["policy"],
                    total_kv_blocks=config["engine"].get("num_kvcache_blocks"),
                    kvcache_block_size=config["engine"]["kvcache_block_size"],
                    max_queue_wait_seconds=(
                        config["admission"]["max_queue_wait_seconds"]
                    ),
                    observe_prefix_cache=(
                        config["admission"]["observe_prefix_cache"]
                    ),
                    eta_prefill_seconds=(
                        config["admission"]["eta_prefill_seconds"]
                    ),
                    eta_decode_seconds_per_token=(
                        config["admission"]["eta_decode_seconds_per_token"]
                    ),
                    eta_safety_margin_seconds=(
                        config["admission"]["eta_safety_margin_seconds"]
                    ),
                ),
            )
    except Exception as exception:
        result = startup_failure_result(specs, config, exception)
    finally:
        if llm is not None:
            try:
                torch.cuda.synchronize()
                gpu_memory = gpu_memory_snapshot()
            except Exception as exception:
                pressure_collection_error = exception_record(exception)
            try:
                atexit.unregister(llm.exit)
                llm.exit()
            except Exception as exception:
                result = fail_result(
                    result,
                    exception_record(exception),
                    label="cleanup_error",
                )
        try:
            pressure = build_pressure_summary(
                llm,
                terminal_counts=result.summary["terminal_counts"],
                error=result.error,
                gpu_memory=gpu_memory,
                observation_active=observation_active,
                admission_rejection_supported=(
                    config["admission"]["policy"]
                    in {
                        "kv_aware_fifo",
                        "prefix_aware_fifo",
                        "slack_aware_prefix_fifo",
                    }
                ),
                collection_error=pressure_collection_error,
            )
            recompute_observability = build_recompute_observability(llm)
            if llm is not None and recompute_observability is None:
                raise RuntimeError(
                    "scheduler recompute observation is missing"
                )
            if config["engine"]["scheduler_policy"] == "pressure_aware_decode":
                pressure_aware = build_pressure_aware_summary(llm)
                if pressure_aware is None:
                    raise RuntimeError(
                        "pressure_aware scheduler observation is missing"
                    )
            if (
                config["engine"]["scheduler_policy"]
                == "chunked_prefill_budgeted"
            ):
                chunked_prefill_budgeted = (
                    build_chunked_prefill_summary(llm)
                )
                if chunked_prefill_budgeted is None:
                    raise RuntimeError(
                        "chunked_prefill_budgeted scheduler observation "
                        "is missing"
                    )
            if config["engine"]["scheduler_policy"] == "recompute_aware":
                recompute_aware = build_recompute_aware_summary(llm)
                if recompute_aware is None:
                    raise RuntimeError(
                        "recompute_aware scheduler observation is missing"
                    )
            if (
                config["engine"]["scheduler_policy"]
                == "recompute_aware_bounded"
            ):
                recompute_aware_bounded = (
                    build_bounded_recompute_summary(llm)
                )
                if recompute_aware_bounded is None:
                    raise RuntimeError(
                        "recompute_aware_bounded scheduler observation "
                        "is missing"
                    )
        except Exception as exception:
            pressure_collection_error = exception_record(exception)
            pressure = build_pressure_summary(
                None,
                terminal_counts=result.summary["terminal_counts"],
                error=result.error,
                gpu_memory=gpu_memory,
                observation_active=observation_active,
                admission_rejection_supported=(
                    config["admission"]["policy"]
                    in {
                        "kv_aware_fifo",
                        "prefix_aware_fifo",
                        "slack_aware_prefix_fifo",
                    }
                ),
                collection_error=pressure_collection_error,
            )
        if pressure_collection_error is not None:
            result = fail_result(
                result,
                pressure_collection_error,
                label="pressure_collection_error",
            )

    requests_path = output_paths["requests"]
    steps_path = output_paths["steps"]
    summary_path = output_paths["summary"]
    write_jsonl_atomic(
        requests_path,
        [record.to_dict() for record in result.records],
    )
    write_jsonl_atomic(steps_path, result.step_events)
    phase_timings_path = output_paths.get("phase_timings")
    if phase_timings_path is not None:
        write_jsonl_atomic(phase_timings_path, result.phase_timings)
    admission_events_path = output_paths.get("admission_events")
    if admission_events_path is not None:
        write_jsonl_atomic(admission_events_path, result.admission_events)
    cache_events_path = output_paths.get("cache_events")
    if cache_events_path is not None:
        write_jsonl_atomic(cache_events_path, result.cache_events)
    cache_states_path = output_paths.get("cache_states")
    if cache_states_path is not None:
        write_jsonl_atomic(cache_states_path, result.cache_states)
    slack_events_path = output_paths.get("slack_events")
    if slack_events_path is not None:
        write_jsonl_atomic(slack_events_path, result.slack_events)

    try:
        runtime = runtime_metadata()
    except Exception as exception:
        runtime = {
            "metadata_errors": [
                f"runtime_metadata: {type(exception).__name__}: {exception}"
            ]
        }
    summary = {
        "status": result.status,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(
            [
                sys.executable,
                "-m",
                "experiments.run_open_loop",
                "--config",
                str(config_path),
            ]
        ),
        "working_directory": str(REPO_ROOT),
        "upstream_commit": config["upstream_commit"],
        "repository_head": git_output("rev-parse", "HEAD"),
        "repository_status": git_output("status", "--short").splitlines(),
        "config_path": str(config_path),
        "provenance": {
            "config_sha256": file_sha256(config_path),
            "trace_path": str(trace_path),
            "trace_sha256": file_sha256(trace_path),
            "runner_sha256": file_sha256(Path(__file__)),
            "open_loop_sha256": file_sha256(
                REPO_ROOT / "experiments/benchmark/open_loop.py"
            ),
            "lifecycle_sha256": file_sha256(
                REPO_ROOT / "experiments/benchmark/lifecycle.py"
            ),
            "llm_engine_sha256": file_sha256(
                REPO_ROOT / "nanovllm/engine/llm_engine.py"
            ),
            "scheduler_sha256": file_sha256(
                REPO_ROOT / "nanovllm/engine/scheduler.py"
            ),
            "sequence_sha256": file_sha256(
                REPO_ROOT / "nanovllm/engine/sequence.py"
            ),
            "nanovllm_package_sha256": python_package_sha256(
                REPO_ROOT / "nanovllm"
            ),
            "block_manager_sha256": file_sha256(
                REPO_ROOT / "nanovllm/engine/block_manager.py"
            ),
            "model_runner_sha256": file_sha256(
                REPO_ROOT / "nanovllm/engine/model_runner.py"
            ),
            "config_module_sha256": file_sha256(
                REPO_ROOT / "nanovllm/config.py"
            ),
        },
        "model": {
            **config["model"],
            "verified_sha256": model_hashes,
        },
        "runtime": runtime,
        "engine": config["engine"],
        "sampling": config["sampling"],
        "slo": config["slo"],
        "measurement": config["measurement"],
        "profiling": {
            **config["profiling"],
            "trace_directory": (
                None
                if profiler_trace_directory is None
                else str(profiler_trace_directory)
            ),
        },
        "admission": result.admission,
        "summary": result.summary,
        "pressure": pressure,
        "error": result.error,
        "artifacts": {
            "requests_path": str(requests_path),
            "requests_sha256": file_sha256(requests_path),
            "steps_path": str(steps_path),
            "steps_sha256": file_sha256(steps_path),
        },
        "limitations": list(RUN_LIMITATIONS),
    }
    if phase_timings_path is not None:
        summary["artifacts"]["phase_timings_path"] = str(phase_timings_path)
        summary["artifacts"]["phase_timings_sha256"] = file_sha256(
            phase_timings_path
        )
    if admission_events_path is not None:
        summary["artifacts"]["admission_events_path"] = str(
            admission_events_path
        )
        summary["artifacts"]["admission_events_sha256"] = file_sha256(
            admission_events_path
        )
    if cache_events_path is not None:
        summary["artifacts"]["cache_events_path"] = str(cache_events_path)
        summary["artifacts"]["cache_events_sha256"] = file_sha256(
            cache_events_path
        )
    if cache_states_path is not None:
        summary["artifacts"]["cache_states_path"] = str(cache_states_path)
        summary["artifacts"]["cache_states_sha256"] = file_sha256(
            cache_states_path
        )
    if slack_events_path is not None:
        summary["artifacts"]["slack_events_path"] = str(slack_events_path)
        summary["artifacts"]["slack_events_sha256"] = file_sha256(
            slack_events_path
        )
    if stage12_manifest is not None:
        summary["stage12_manifest"] = stage12_manifest
    if pressure_aware is not None:
        summary["pressure_aware"] = pressure_aware
    if chunked_prefill_budgeted is not None:
        summary["chunked_prefill_budgeted"] = (
            chunked_prefill_budgeted
        )
    if recompute_aware is not None:
        summary["recompute_aware"] = recompute_aware
    if recompute_aware_bounded is not None:
        summary["recompute_aware_bounded"] = recompute_aware_bounded
    if recompute_observability is not None:
        summary["recompute_observability"] = recompute_observability
    write_json_atomic(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if result.status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
