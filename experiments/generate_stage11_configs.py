from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.generate_matrix import write_json_idempotent
from experiments.run_open_loop import load_config


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIRECTORY = "experiments/configs/stage11_formal_r1"
TRACE_DIRECTORY = "experiments/data/stage11_formal_r1"
OUTPUT_DIRECTORY = "experiments/results"
MANIFEST_PATH = REPO_ROOT / "experiments/configs/stage11_formal_r1_manifest.json"
CAPACITIES = (4, 6, 8)
REUSE_RATIOS = (0, 50, 90)
SEEDS = (1, 2, 3)
POLICIES = {
    "fifo": {
        "policy": "kv_aware_fifo",
        "max_queue_wait_seconds": 1.0,
        "observe_prefix_cache": True,
    },
    "prefix": {
        "policy": "prefix_aware_fifo",
        "max_queue_wait_seconds": 1.0,
        "observe_prefix_cache": True,
    },
}
CODE_PATHS = {
    "runner_sha256": "experiments/run_open_loop.py",
    "open_loop_sha256": "experiments/benchmark/open_loop.py",
    "lifecycle_sha256": "experiments/benchmark/lifecycle.py",
    "llm_engine_sha256": "nanovllm/engine/llm_engine.py",
    "scheduler_sha256": "nanovllm/engine/scheduler.py",
    "block_manager_sha256": "nanovllm/engine/block_manager.py",
    "model_runner_sha256": "nanovllm/engine/model_runner.py",
    "config_module_sha256": "nanovllm/config.py",
}
MODEL = {
    "path": "/absolute/path/to/Qwen3-0.6B",
    "repo_id": "Qwen/Qwen3-0.6B",
    "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
    "sha256": {
        "model.safetensors": (
            "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e"
            "382306f42996874b"
        ),
        "config.json": (
            "660db3b73d788119c04535e48cf9be5f55bc3100841a718637"
            "ae695b442f27dd"
        ),
        "tokenizer.json": (
            "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe"
            "794b4492dae4"
        ),
    },
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trace_path(reuse_ratio: int, seed: int) -> Path:
    return REPO_ROOT / TRACE_DIRECTORY / (
        f"stage11_formal_r1_reuse{reuse_ratio}_seed{seed}.json"
    )


def build_trace(reuse_ratio: int, seed: int) -> dict[str, Any]:
    if reuse_ratio not in REUSE_RATIOS:
        raise ValueError("Unknown prefix reuse ratio")
    if seed not in SEEDS:
        raise ValueError("Unknown formal seed")
    shared_interactive_count = 10 * reuse_ratio // 100
    prefix_group = f"stage11-system-prefix-seed{seed}"
    requests: list[dict[str, Any]] = [
        {
            "request_id": "long-000",
            "request_class": "long",
            "arrival_time": 0.0,
            "prompt_length": 768,
            "max_output_tokens": 64,
            "seed": 110000 + seed,
            "prefix_group": prefix_group if shared_interactive_count else None,
            "shared_prefix_length": 512 if shared_interactive_count else 0,
        }
    ]
    for index in range(10):
        shared = index < shared_interactive_count
        requests.append(
            {
                "request_id": f"interactive-{index:03d}",
                "request_class": "interactive",
                "arrival_time": 0.05 + index * 0.01,
                "prompt_length": 768,
                "max_output_tokens": 16,
                "seed": 111000 + seed * 100 + index,
                "prefix_group": prefix_group if shared else None,
                "shared_prefix_length": 512 if shared else 0,
            }
        )
    return {
        "schema_version": 2,
        "description": (
            "Stage 11 matched mixed workload: one long prefix owner and "
            f"{reuse_ratio}% shared-prefix interactive requests (seed {seed})."
        ),
        "time_unit": "seconds",
        "requests": requests,
    }


def build_config(
    *,
    policy_name: str,
    reuse_ratio: int,
    capacity: int,
    seed: int,
) -> dict[str, Any]:
    if policy_name not in POLICIES:
        raise ValueError("Unknown admission policy")
    if capacity not in CAPACITIES:
        raise ValueError("Unknown KV capacity")
    trace = trace_path(reuse_ratio, seed)
    run_id = (
        f"stage11_formal_r1_{policy_name}_reuse{reuse_ratio}_kv{capacity}_"
        f"seed{seed}"
    )
    return {
        "schema_version": 1,
        "upstream_commit": "bb823b3e06983d71485a8e1f23715ebd87d98ef8",
        "model": MODEL,
        "engine": {
            "enforce_eager": True,
            "tensor_parallel_size": 1,
            "max_model_len": 1024,
            "max_num_batched_tokens": 768,
            "max_num_seqs": 8,
            "gpu_memory_utilization": 0.8,
            "kvcache_block_size": 256,
            "scheduler_policy": "prefill_first",
            "decode_token_budget": 0,
            "decode_step_guard": 0,
            "num_kvcache_blocks": capacity,
        },
        "sampling": {
            "temperature": 0.6,
            "ignore_eos": True,
            "seed": 20260811 + seed,
        },
        "workload": {
            "trace_path": str(trace.relative_to(REPO_ROOT)),
            "token_id_upper_bound": 10000,
        },
        "slo": {
            "ttft_slo_ms": 10000.0,
            "itl_slo_ms": 10000.0,
            "require_itl": True,
        },
        "measurement": {
            "start_seconds": 0.0,
            "end_seconds": 30.0,
            "max_run_seconds": 45.0,
        },
        "warmup": {
            "enabled": True,
            "prompt_length": 32,
            "max_output_tokens": 2,
            "seed": 111100 + seed,
            "batch_sizes": [1, 2],
        },
        "profiling": {
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
        "admission": POLICIES[policy_name],
        "output": {"directory": OUTPUT_DIRECTORY, "run_id": run_id},
    }


def planned_configs() -> list[tuple[Path, dict[str, Any]]]:
    planned = []
    for reuse_ratio in REUSE_RATIOS:
        for capacity in CAPACITIES:
            for seed in SEEDS:
                for policy_name in POLICIES:
                    config = build_config(
                        policy_name=policy_name,
                        reuse_ratio=reuse_ratio,
                        capacity=capacity,
                        seed=seed,
                    )
                    path = REPO_ROOT / CONFIG_DIRECTORY / (
                        f"{config['output']['run_id']}.json"
                    )
                    planned.append((path, config))
    return planned


def manifest() -> dict[str, Any]:
    for reuse_ratio in REUSE_RATIOS:
        for seed in SEEDS:
            path = trace_path(reuse_ratio, seed)
            if json.loads(path.read_text(encoding="utf-8")) != build_trace(
                reuse_ratio, seed
            ):
                raise ValueError("Stage 11 formal trace differs from generator")
    traces = {
        f"reuse{reuse_ratio}_seed{seed}": {
            "path": str(trace_path(reuse_ratio, seed).relative_to(REPO_ROOT)),
            "sha256": sha256_file(trace_path(reuse_ratio, seed)),
        }
        for reuse_ratio in REUSE_RATIOS
        for seed in SEEDS
    }
    return {
        "schema_version": 1,
        "revision": "stage11_formal_r1",
        "description": (
            "Matched Prefix-Aware KV admission ablation with real GPU "
            "Prefix Cache preview on a frozen single-GPU Nano-vLLM setup."
        ),
        "config_directory": CONFIG_DIRECTORY,
        "trace_directory": TRACE_DIRECTORY,
        "output_directory": OUTPUT_DIRECTORY,
        "capacities": list(CAPACITIES),
        "reuse_ratios": list(REUSE_RATIOS),
        "seeds": list(SEEDS),
        "policies": POLICIES,
        "traces": traces,
        "required_cells": len(CAPACITIES)
        * len(REUSE_RATIOS)
        * len(SEEDS)
        * len(POLICIES),
        "execution_code": {
            key: sha256_file(REPO_ROOT / path)
            for key, path in CODE_PATHS.items()
        },
        "generation_code": {
            "generator_sha256": sha256_file(Path(__file__)),
            "aggregator_sha256": sha256_file(
                REPO_ROOT / "experiments/aggregate_stage11.py"
            ),
        },
        "completion_gates": {
            "status": "passed",
            "terminal_states": ["Finished", "Rejected"],
            "timed_out": False,
            "oom_detected": False,
            "final_kv_used_blocks": 0,
            "final_reserved_blocks": 0,
        },
    }


def main() -> None:
    for reuse_ratio in REUSE_RATIOS:
        for seed in SEEDS:
            write_json_idempotent(
                trace_path(reuse_ratio, seed), build_trace(reuse_ratio, seed)
            )
    for path, config in planned_configs():
        write_json_idempotent(path, config)
        load_config(path)
    write_json_idempotent(MANIFEST_PATH, manifest())


if __name__ == "__main__":
    main()
