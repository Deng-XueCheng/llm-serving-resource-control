from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.generate_matrix import write_json_idempotent
from experiments.generate_stage15_configs import source_config_path


REPO_ROOT = Path(__file__).resolve().parents[1]
CAPACITIES = (4, 6, 8)
SEEDS = (1, 2, 3)
POLICIES = {
    "pressure": "pressure_aware_decode",
    "recompute": "recompute_aware",
    "bounded": "recompute_aware_bounded",
}
REVISION = "stage16_diagnostic_r2"
MAX_DRAIN_STEPS = 16
WAITING_AGE_LIMIT = 32
EXECUTION_SOURCES = {
    "runner_sha256": "experiments/run_open_loop.py",
    "open_loop_sha256": "experiments/benchmark/open_loop.py",
    "lifecycle_sha256": "experiments/benchmark/lifecycle.py",
    "llm_engine_sha256": "nanovllm/engine/llm_engine.py",
    "scheduler_sha256": "nanovllm/engine/scheduler.py",
    "sequence_sha256": "nanovllm/engine/sequence.py",
    "block_manager_sha256": "nanovllm/engine/block_manager.py",
    "model_runner_sha256": "nanovllm/engine/model_runner.py",
    "config_module_sha256": "nanovllm/config.py",
}
EXECUTION_FINGERPRINT_KEYS = (
    "runner_sha256",
    "open_loop_sha256",
    "lifecycle_sha256",
    "llm_engine_sha256",
    "scheduler_sha256",
    "sequence_sha256",
    "nanovllm_package_sha256",
    "block_manager_sha256",
    "model_runner_sha256",
    "config_module_sha256",
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


def execution_fingerprint() -> dict[str, str]:
    fingerprint = {
        key: file_sha256(REPO_ROOT / relative_path)
        for key, relative_path in EXECUTION_SOURCES.items()
    }
    fingerprint["nanovllm_package_sha256"] = python_package_sha256(
        REPO_ROOT / "nanovllm"
    )
    return {
        key: fingerprint[key]
        for key in EXECUTION_FINGERPRINT_KEYS
    }


def trace_binding(config: dict[str, Any], seed: int) -> dict[str, Any]:
    trace_path = Path(config["workload"]["trace_path"])
    if not trace_path.is_absolute():
        trace_path = (REPO_ROOT / trace_path).resolve()
    if not trace_path.is_file():
        raise FileNotFoundError(f"Stage 16 trace not found: {trace_path}")
    return {
        "seed": seed,
        "trace_path": str(trace_path.relative_to(REPO_ROOT)),
        "trace_sha256": file_sha256(trace_path),
    }


def run_id(capacity: int, policy: str, seed: int) -> str:
    return f"{REVISION}_kv{capacity}_{policy}_seed{seed}"


def plan_configs() -> list[dict[str, Any]]:
    directory = REPO_ROOT / f"experiments/configs/{REVISION}"
    planned = []
    for capacity in CAPACITIES:
        for seed in SEEDS:
            source_path = source_config_path(capacity, seed)
            source = json.loads(source_path.read_text(encoding="utf-8"))
            for policy_name, scheduler_policy in POLICIES.items():
                config = copy.deepcopy(source)
                config["engine"]["scheduler_policy"] = scheduler_policy
                if policy_name == "bounded":
                    config["engine"].update(
                        {
                            "max_drain_steps": MAX_DRAIN_STEPS,
                            "waiting_age_limit": WAITING_AGE_LIMIT,
                            "ttft_slo_ms": config["slo"]["ttft_slo_ms"],
                            "itl_slo_ms": config["slo"]["itl_slo_ms"],
                        }
                    )
                config["output"]["run_id"] = run_id(
                    capacity,
                    policy_name,
                    seed,
                )
                path = directory / f"{config['output']['run_id']}.json"
                planned.append(
                    {
                        "capacity": capacity,
                        "seed": seed,
                        "policy": policy_name,
                        "source": source_path,
                        "path": path,
                        "config": config,
                    }
                )
    return planned


def normalized_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(config)
    normalized["engine"]["scheduler_policy"] = "<policy>"
    for field in (
        "max_drain_steps",
        "waiting_age_limit",
        "ttft_slo_ms",
        "itl_slo_ms",
    ):
        normalized["engine"].pop(field, None)
    normalized["output"]["run_id"] = "<run-id>"
    return normalized


def validate_plan(planned: list[dict[str, Any]]) -> None:
    if len(planned) != 27:
        raise ValueError("Stage 16 plan must contain exactly 27 cells")
    indexed = {
        (item["capacity"], item["seed"], item["policy"]): item
        for item in planned
    }
    if len(indexed) != 27:
        raise ValueError("Stage 16 plan contains duplicate cells")
    for capacity in CAPACITIES:
        for seed in SEEDS:
            normalized = set()
            for policy_name, scheduler_policy in POLICIES.items():
                item = indexed[(capacity, seed, policy_name)]
                config = item["config"]
                if (
                    config["engine"]["num_kvcache_blocks"] != capacity
                    or config["engine"]["scheduler_policy"]
                    != scheduler_policy
                    or config["sampling"]["seed"] != 20262000 + seed
                    or config["output"]["run_id"] != item["path"].stem
                    or not config["workload"]["trace_path"].endswith(
                        f"overload_seed{seed}.json"
                    )
                ):
                    raise ValueError("Stage 16 cell label/config differs")
                if policy_name == "bounded" and (
                    config["engine"]["max_drain_steps"]
                    != MAX_DRAIN_STEPS
                    or config["engine"]["waiting_age_limit"]
                    != WAITING_AGE_LIMIT
                    or config["engine"]["ttft_slo_ms"]
                    != config["slo"]["ttft_slo_ms"]
                    or config["engine"]["itl_slo_ms"]
                    != config["slo"]["itl_slo_ms"]
                ):
                    raise ValueError("Stage 16 bounded contract differs")
                normalized.add(
                    json.dumps(
                        normalized_config(config),
                        sort_keys=True,
                    )
                )
            if len(normalized) != 1:
                raise ValueError("Stage 16 matched configs differ")


def build_manifest(planned: list[dict[str, Any]]) -> dict[str, Any]:
    traces = {
        item["seed"]: trace_binding(item["config"], item["seed"])
        for item in planned
    }
    return {
        "schema_version": 2,
        "revision": REVISION,
        "description": (
            "Frozen 27-cell matched diagnostic comparing pressure-aware, "
            "unbounded recompute-aware and SLO-aware bounded drain."
        ),
        "capacities": list(CAPACITIES),
        "seeds": list(SEEDS),
        "policies": list(POLICIES),
        "required_cells": 27,
        "controlled_difference": [
            "engine.scheduler_policy",
            "engine.max_drain_steps",
            "engine.waiting_age_limit",
            "engine.ttft_slo_ms",
            "engine.itl_slo_ms",
            "output.run_id",
        ],
        "bounded_contract": {
            "max_drain_steps": MAX_DRAIN_STEPS,
            "waiting_age_limit": WAITING_AGE_LIMIT,
            "ttft_slo_source": "slo.ttft_slo_ms",
            "itl_slo_source": "slo.itl_slo_ms",
            "episode_entry_positive_slack_only": True,
        },
        "completion_gates": {
            "status": "passed",
            "all_requests_finished": True,
            "terminal_reconciled": True,
            "timed_out": False,
            "oom_detected": False,
            "final_kv_used_blocks": 0,
            "raw_summary_counters_match": True,
        },
        "acceptance": {
            "fairness_reduction_min": 0.50,
            "itl_p99_reduction_min": 0.30,
            "resource_reduction_min": 0.50,
            "ttft_p99_regression_max": 0.10,
            "improved_matched_pairs_min": 6,
            "matched_pairs": 9,
        },
        "execution_fingerprint": execution_fingerprint(),
        "traces": [traces[seed] for seed in SEEDS],
        "cells": [
            {
                "capacity": item["capacity"],
                "seed": item["seed"],
                "policy": item["policy"],
                "run_id": item["config"]["output"]["run_id"],
                "config_path": str(item["path"].relative_to(REPO_ROOT)),
                "source_config_path": str(
                    item["source"].relative_to(REPO_ROOT)
                ),
                "config_sha256": hashlib.sha256(
                    (
                        json.dumps(
                            item["config"],
                            ensure_ascii=False,
                            indent=2,
                        )
                        + "\n"
                    ).encode("utf-8")
                ).hexdigest(),
                "source_config_sha256": file_sha256(item["source"]),
                "trace_path": traces[item["seed"]]["trace_path"],
                "trace_sha256": traces[item["seed"]]["trace_sha256"],
            }
            for item in planned
        ],
    }


def generate() -> list[tuple[Path, str]]:
    planned = plan_configs()
    validate_plan(planned)
    outputs = [
        (item["path"], write_json_idempotent(item["path"], item["config"]))
        for item in planned
    ]
    manifest_path = REPO_ROOT / f"experiments/configs/{REVISION}_matrix.json"
    outputs.append(
        (
            manifest_path,
            write_json_idempotent(manifest_path, build_manifest(planned)),
        )
    )
    return outputs


def main() -> None:
    argparse.ArgumentParser().parse_args()
    for path, status in generate():
        print(f"{status}: {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
