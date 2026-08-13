from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.generate_matrix import write_json_idempotent


REPO_ROOT = Path(__file__).resolve().parents[1]
CAPACITIES = (4, 6, 8)
SEEDS = (1, 2, 3)
POLICIES = {
    "pressure": "pressure_aware_decode",
    "recompute": "recompute_aware",
}
DEFAULT_REVISION = "stage15_diagnostic_r1"
SUPPORTED_REVISIONS = {
    DEFAULT_REVISION,
    "stage15_diagnostic_r2",
    "stage15_diagnostic_r3",
}


def source_config_path(capacity: int, seed: int) -> Path:
    return (
        REPO_ROOT
        / f"experiments/configs/stage10_formal_r2_kv{capacity}"
        / (
            f"stage10_formal_r2_kv{capacity}_adaptive_"
            f"overload_seed{seed}.json"
        )
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_id(
    capacity: int,
    policy: str,
    seed: int,
    *,
    revision: str = DEFAULT_REVISION,
) -> str:
    return f"{revision}_kv{capacity}_{policy}_seed{seed}"


def plan_configs(
    revision: str = DEFAULT_REVISION,
) -> list[dict[str, Any]]:
    if revision not in SUPPORTED_REVISIONS:
        raise ValueError("Unsupported Stage 15 revision")
    config_directory = REPO_ROOT / f"experiments/configs/{revision}"
    planned = []
    for capacity in CAPACITIES:
        for seed in SEEDS:
            source_path = source_config_path(capacity, seed)
            source = json.loads(source_path.read_text(encoding="utf-8"))
            for policy_name, scheduler_policy in POLICIES.items():
                config = copy.deepcopy(source)
                config["engine"]["scheduler_policy"] = scheduler_policy
                config["output"]["run_id"] = run_id(
                    capacity,
                    policy_name,
                    seed,
                    revision=revision,
                )
                planned.append(
                    {
                        "capacity": capacity,
                        "seed": seed,
                        "policy": policy_name,
                        "source": source_path,
                        "path": config_directory
                        / f"{config['output']['run_id']}.json",
                        "config": config,
                    }
                )
    return planned


def normalized_pair_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(config)
    normalized["engine"]["scheduler_policy"] = "<policy>"
    normalized["output"]["run_id"] = "<run-id>"
    return normalized


def validate_plan(planned: list[dict[str, Any]]) -> None:
    if len(planned) != 18:
        raise ValueError("Stage 15 plan must contain exactly 18 cells")
    indexed = {
        (item["capacity"], item["seed"], item["policy"]): item
        for item in planned
    }
    if len(indexed) != 18:
        raise ValueError("Stage 15 plan contains duplicate cells")
    for capacity in CAPACITIES:
        for seed in SEEDS:
            baseline_item = indexed[(capacity, seed, "pressure")]
            candidate_item = indexed[(capacity, seed, "recompute")]
            baseline = baseline_item["config"]
            candidate = candidate_item["config"]
            for policy_name, item in (
                ("pressure", baseline_item),
                ("recompute", candidate_item),
            ):
                config = item["config"]
                expected_policy = POLICIES[policy_name]
                if (
                    config["engine"]["num_kvcache_blocks"] != capacity
                    or config["engine"]["scheduler_policy"]
                    != expected_policy
                    or config["sampling"]["seed"] != 20262000 + seed
                    or config["output"]["run_id"] != item["path"].stem
                    or not config["workload"]["trace_path"].endswith(
                        f"overload_seed{seed}.json"
                    )
                ):
                    raise ValueError("Stage 15 cell label/config differs")
            if normalized_pair_config(baseline) != normalized_pair_config(
                candidate
            ):
                raise ValueError("Stage 15 matched configs differ")


def build_manifest(
    planned: list[dict[str, Any]],
    *,
    revision: str = DEFAULT_REVISION,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "revision": revision,
        "description": (
            "Frozen 18-cell matched diagnostic comparing the existing "
            "pressure-aware baseline with recompute-aware KV scheduling."
        ),
        "capacities": list(CAPACITIES),
        "seeds": list(SEEDS),
        "policies": list(POLICIES),
        "required_cells": 18,
        "controlled_difference": [
            "engine.scheduler_policy",
            "output.run_id",
        ],
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
            "aggregate_actual_recompute_tokens_reduction_min": 0.20,
            "improved_matched_pairs_min": 6,
            "matched_pairs": 9,
        },
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
                    (json.dumps(
                        item["config"],
                        ensure_ascii=False,
                        indent=2,
                    ) + "\n").encode("utf-8")
                ).hexdigest(),
                "source_config_sha256": file_sha256(item["source"]),
            }
            for item in planned
        ],
    }


def generate(
    revision: str = DEFAULT_REVISION,
) -> list[tuple[Path, str]]:
    planned = plan_configs(revision)
    validate_plan(planned)
    outputs = [
        (item["path"], write_json_idempotent(item["path"], item["config"]))
        for item in planned
    ]
    manifest = build_manifest(planned, revision=revision)
    manifest_path = REPO_ROOT / f"experiments/configs/{revision}_matrix.json"
    outputs.append(
        (manifest_path, write_json_idempotent(manifest_path, manifest))
    )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--revision",
        choices=sorted(SUPPORTED_REVISIONS),
        default=DEFAULT_REVISION,
    )
    args = parser.parse_args()
    for path, status in generate(args.revision):
        print(f"{status}: {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
