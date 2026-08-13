from __future__ import annotations

"""Generate the frozen Stage 12 r3 matched-pair configs and manifests."""

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.aggregate_stage12 import execution_code_fingerprint
from experiments.run_open_loop import load_config


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "experiments/configs/stage12_formal_r3"
TRACE_DIR = REPO_ROOT / "experiments/data/stage12_formal_r3"
RESULTS = REPO_ROOT / "experiments/results"
CAL_CONFIG = REPO_ROOT / "experiments/configs/stage12_calibration_r1.json"
CAL_TRACE = REPO_ROOT / "experiments/data/stage12_calibration_r1.json"
CAL_RESULTS = RESULTS / "stage12_calibration_r1"
CAL_SCRIPT = REPO_ROOT / "experiments/derive_stage12_eta.py"
CAL_ETA = RESULTS / "stage12_calibration_r1.eta.json"
MANIFEST = REPO_ROOT / "experiments/configs/stage12_formal_r3_manifest.json"
SEEDS = (1, 2, 3)
POLICIES = ("prefix", "slack")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path: Path, document: dict[str, Any]) -> None:
    text = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError(f"Refusing to overwrite frozen document: {path}")
    path.write_text(text, encoding="utf-8")


def trace(seed: int) -> dict[str, Any]:
    group = f"stage12-formal-system-{seed}"
    return {
        "schema_version": 2,
        "description": (
            "Stage 12 r3 matched workload: a 3-block head is infeasible while "
            "a following 2-block shared-prefix request remains admissible."
        ),
        "time_unit": "seconds",
        "requests": [
            {"request_id": "long-000", "request_class": "long", "arrival_time": 0.0,
             "prompt_length": 768, "max_output_tokens": 64, "seed": 121000 + seed,
             "prefix_group": group, "shared_prefix_length": 512},
            {"request_id": "interactive-000", "request_class": "interactive", "arrival_time": 0.05,
             "prompt_length": 768, "max_output_tokens": 16, "seed": 122000 + seed,
             "prefix_group": group, "shared_prefix_length": 256},
            {"request_id": "interactive-001", "request_class": "interactive", "arrival_time": 0.05,
             "prompt_length": 768, "max_output_tokens": 16, "seed": 123000 + seed,
             "prefix_group": group, "shared_prefix_length": 512},
        ],
    }


def calibration(eta: dict[str, Any]) -> dict[str, Any]:
    names = ("config", "trace", "requests", "steps", "summary", "phase_timings")
    paths = {
        "config": CAL_CONFIG, "trace": CAL_TRACE,
        "requests": CAL_RESULTS.with_suffix(".requests.jsonl"),
        "steps": CAL_RESULTS.with_suffix(".steps.jsonl"),
        "summary": CAL_RESULTS.with_suffix(".summary.json"),
        "phase_timings": CAL_RESULTS.with_suffix(".phase_timings.jsonl"),
        "derivation_script": CAL_SCRIPT, "derivation_output": CAL_ETA,
    }
    result = {f"{name}_path": str(path.resolve()) for name, path in paths.items()}
    result.update({f"{name}_sha256": sha256(path) for name, path in paths.items()})
    result.update({key: eta[key] for key in (
        "eta_prefill_seconds", "eta_decode_seconds_per_token", "eta_safety_margin_seconds"
    )})
    return result


def build_config(base: dict[str, Any], *, seed: int, policy: str) -> dict[str, Any]:
    document = copy.deepcopy(base)
    run_id = f"stage12_formal_r3_{policy}_seed{seed}"
    document["sampling"]["seed"] = 20261200 + seed
    document["workload"]["trace_path"] = str(
        (TRACE_DIR / f"stage12_formal_r3_seed{seed}.json").relative_to(REPO_ROOT)
    ).replace("\\", "/")
    document["output"] = {"directory": "experiments/results", "run_id": run_id}
    if policy == "prefix":
        document["admission"] = {
            "policy": "prefix_aware_fifo", "max_queue_wait_seconds": 0.2,
            "observe_prefix_cache": True, "eta_prefill_seconds": 0.0,
            "eta_decode_seconds_per_token": 0.0, "eta_safety_margin_seconds": 0.0,
        }
    else:
        document["admission"]["policy"] = "slack_aware_prefix_fifo"
        document["admission"]["max_queue_wait_seconds"] = 0.2
        document["admission"]["observe_prefix_cache"] = True
    return document


def main() -> None:
    base = json.loads(CAL_CONFIG.read_text(encoding="utf-8"))
    eta = json.loads(CAL_ETA.read_text(encoding="utf-8"))
    for key in ("eta_prefill_seconds", "eta_decode_seconds_per_token", "eta_safety_margin_seconds"):
        base["admission"][key] = eta[key]
    cells = []
    for seed in SEEDS:
        trace_path = TRACE_DIR / f"stage12_formal_r3_seed{seed}.json"
        write_once(trace_path, trace(seed))
        for policy in POLICIES:
            config = build_config(base, seed=seed, policy=policy)
            config_path = CONFIG_DIR / f"{config['output']['run_id']}.json"
            write_once(config_path, config)
            load_config(config_path)
            cell = {"seed": seed, "policy": policy, "config_path": str(config_path.resolve()),
                    "config_sha256": sha256(config_path), "trace_path": str(trace_path.resolve()),
                    "trace_sha256": sha256(trace_path)}
            if policy == "slack":
                manifest = {
                    "schema_version": 1, "stage": "stage12", "kind": "formal",
                    "run_id": config["output"]["run_id"], "config_path": str(config_path.resolve()),
                    "config_sha256": sha256(config_path), "trace_path": str(trace_path.resolve()),
                    "trace_sha256": sha256(trace_path), "execution_code": execution_code_fingerprint(),
                    "aggregate_sha256": sha256(REPO_ROOT / "experiments/aggregate_stage12.py"),
                    "upstream_commit": config["upstream_commit"], "calibration": calibration(eta),
                }
                manifest_path = CONFIG_DIR / f"{config['output']['run_id']}.manifest.json"
                write_once(manifest_path, manifest)
                cell["manifest_path"] = str(manifest_path.resolve())
                cell["manifest_sha256"] = sha256(manifest_path)
            cells.append(cell)
    write_once(MANIFEST, {"schema_version": 1, "stage": "stage12", "revision": "r3",
                          "seeds": list(SEEDS), "policies": list(POLICIES), "cells": cells})
    print(f"Generated {len(cells)} Stage 12 formal r3 cells")


if __name__ == "__main__":
    main()
