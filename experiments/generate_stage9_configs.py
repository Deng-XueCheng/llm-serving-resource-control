from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from experiments.generate_matrix import write_json_idempotent
from experiments.run_open_loop import load_config


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIRECTORY = "experiments/configs/stage9_formal_r1"
OUTPUT_DIRECTORY = "experiments/results"
SEEDS = (1, 2, 3)
POLICIES = {
    "disabled": {"policy": "disabled", "max_queue_wait_seconds": 0.0},
    "fifo": {"policy": "kv_aware_fifo", "max_queue_wait_seconds": 0.25},
}
PROFILING_DISABLED = {
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
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_path(seed: int) -> Path:
    return REPO_ROOT / (
        "experiments/configs/stage10_formal_r2_kv4/"
        f"stage10_formal_r2_kv4_adaptive_overload_seed{seed}.json"
    )


def planned_configs() -> list[tuple[Path, dict]]:
    planned = []
    for seed in SEEDS:
        source = json.loads(source_path(seed).read_text(encoding="utf-8"))
        if (
            source["engine"].get("num_kvcache_blocks") != 4
            or source["engine"]["scheduler_policy"] != "pressure_aware_decode"
            or source["measurement"]["max_run_seconds"] != 45.0
        ):
            raise ValueError("Stage 9 source config does not match frozen setup")
        for name, admission in POLICIES.items():
            value = copy.deepcopy(source)
            run_id = f"stage9_formal_r1_{name}_overload_seed{seed}"
            value["admission"] = copy.deepcopy(admission)
            value["output"] = {"directory": OUTPUT_DIRECTORY, "run_id": run_id}
            value["profiling"] = copy.deepcopy(PROFILING_DISABLED)
            path = REPO_ROOT / CONFIG_DIRECTORY / f"{run_id}.json"
            planned.append((path, value))
    return planned


def manifest() -> dict:
    sources = {
        str(seed): {
            "path": str(source_path(seed).relative_to(REPO_ROOT)),
            "sha256": sha256_file(source_path(seed)),
        }
        for seed in SEEDS
    }
    return {
        "schema_version": 1,
        "revision": "stage9_formal_r1",
        "description": "KV-aware FIFO admission ablation on the frozen Stage 10 r2 4-block adaptive overload setup.",
        "source_configs": sources,
        "config_directory": CONFIG_DIRECTORY,
        "output_directory": OUTPUT_DIRECTORY,
        "seeds": list(SEEDS),
        "policies": POLICIES,
        "required_cells": len(SEEDS) * len(POLICIES),
        "execution_code": {
            "runner_sha256": sha256_file(REPO_ROOT / "experiments/run_open_loop.py"),
            "open_loop_sha256": sha256_file(REPO_ROOT / "experiments/benchmark/open_loop.py"),
            "lifecycle_sha256": sha256_file(REPO_ROOT / "experiments/benchmark/lifecycle.py"),
            "llm_engine_sha256": sha256_file(REPO_ROOT / "nanovllm/engine/llm_engine.py"),
            "scheduler_sha256": sha256_file(REPO_ROOT / "nanovllm/engine/scheduler.py"),
            "block_manager_sha256": sha256_file(REPO_ROOT / "nanovllm/engine/block_manager.py"),
            "model_runner_sha256": sha256_file(REPO_ROOT / "nanovllm/engine/model_runner.py"),
            "config_module_sha256": sha256_file(REPO_ROOT / "nanovllm/config.py"),
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
    for path, value in planned_configs():
        write_json_idempotent(path, value)
        load_config(path)
    write_json_idempotent(
        REPO_ROOT / "experiments/configs/stage9_formal_r1_manifest.json",
        manifest(),
    )


if __name__ == "__main__":
    main()
