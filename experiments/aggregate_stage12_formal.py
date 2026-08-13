from __future__ import annotations

"""Aggregate the frozen Stage 12 r3 matched single-GPU experiment.

This module deliberately treats the global formal manifest as an input plan,
not as a source of truth for results.  Every completed cell is rebound to its
frozen config and trace; raw serving events are replayed before a metric is
reported.  The slack policy additionally uses the Stage 12 ETA/cache replay.
"""

import argparse
import copy
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.aggregate_stage11 import validate_events, validate_pressure
from experiments.aggregate_stage12 import (
    CODE_PATHS,
    REPO_ROOT,
    execution_code_fingerprint,
    read_jsonl,
    resolve_recorded_manifest_path,
    sha256,
    validate_raw,
    validate_run_artifacts,
)
from experiments.benchmark.calibration_evidence import recorded_path_matches


RESULTS = REPO_ROOT / "experiments/results"
FORMAL_MANIFEST = REPO_ROOT / "experiments/configs/stage12_formal_r3_manifest.json"
FORMAL_MANIFEST_SHA256 = "ec37c959db99297c9e7ac5699fb2394c72c896adda9dfb14fb184f647989c1d3"
OUTPUT = RESULTS / "stage12_formal_r3_aggregate.json"
SEEDS = (1, 2, 3)
POLICIES = {"prefix": "prefix_aware_fifo", "slack": "slack_aware_prefix_fifo"}
METRICS = (
    "interactive_slo_goodput_rps",
    "interactive_ttft_p99_ms",
    "interactive_itl_p99_ms",
    "long_window_token_goodput_tps",
    "output_throughput_tps",
    "elapsed_seconds",
)
AGGREGATION_DEPENDENCIES = {
    "aggregate_stage12_formal_sha256": Path("experiments/aggregate_stage12_formal.py"),
    "aggregate_stage12_sha256": Path("experiments/aggregate_stage12.py"),
    "aggregate_stage11_sha256": Path("experiments/aggregate_stage11.py"),
}


def aggregation_fingerprint() -> dict[str, str]:
    """Bind the top-level aggregate and its two replay helper modules."""
    return {
        field: sha256(REPO_ROOT / path)
        for field, path in AGGREGATION_DEPENDENCIES.items()
    }


def _path(value: Any, *, field: str, directory: Path) -> Path:
    try:
        return resolve_recorded_manifest_path(
            value,
            field=field,
            allowed_directory=directory,
        )
    except ValueError as exception:
        raise ValueError(
            f"formal manifest {field} is outside its frozen directory"
        ) from exception


def _expected_run_id(policy: str, seed: int) -> str:
    return f"stage12_formal_r3_{policy}_seed{seed}"


def _trace_path(seed: int) -> Path:
    return REPO_ROOT / "experiments/data/stage12_formal_r3" / f"stage12_formal_r3_seed{seed}.json"


def _config_path(policy: str, seed: int) -> Path:
    return REPO_ROOT / "experiments/configs/stage12_formal_r3" / f"{_expected_run_id(policy, seed)}.json"


def _slack_manifest_path(seed: int) -> Path:
    return REPO_ROOT / "experiments/configs/stage12_formal_r3" / f"{_expected_run_id('slack', seed)}.manifest.json"


def load_formal_plan(manifest_path: Path = FORMAL_MANIFEST) -> dict[tuple[str, int], dict[str, Any]]:
    """Validate the immutable 2-policy x 3-seed r3 input plan."""
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        set(document) != {"schema_version", "stage", "revision", "seeds", "policies", "cells"}
        or document.get("schema_version") != 1
        or document.get("stage") != "stage12"
        or document.get("revision") != "r3"
        or document.get("seeds") != list(SEEDS)
        or document.get("policies") != list(POLICIES)
        or not isinstance(document.get("cells"), list)
    ):
        raise ValueError("Stage 12 formal manifest schema differs")
    plan: dict[tuple[str, int], dict[str, Any]] = {}
    expected_keys = {(policy, seed) for policy in POLICIES for seed in SEEDS}
    for cell in document["cells"]:
        if not isinstance(cell, dict):
            raise ValueError("formal manifest cell differs")
        policy, seed = cell.get("policy"), cell.get("seed")
        key = (policy, seed)
        required = {"policy", "seed", "config_path", "config_sha256", "trace_path", "trace_sha256"}
        if policy == "slack":
            required |= {"manifest_path", "manifest_sha256"}
        if (
            key not in expected_keys
            or key in plan
            or set(cell) != required
            or not isinstance(seed, int)
            or isinstance(seed, bool)
        ):
            raise ValueError("formal manifest cell identity differs")
        config_path = _path(cell["config_path"], field="config_path", directory=REPO_ROOT / "experiments/configs")
        trace_path = _path(cell["trace_path"], field="trace_path", directory=REPO_ROOT / "experiments/data")
        if (
            config_path.resolve() != _config_path(policy, seed).resolve()
            or trace_path.resolve() != _trace_path(seed).resolve()
            or sha256(config_path) != cell["config_sha256"]
            or sha256(trace_path) != cell["trace_sha256"]
        ):
            raise ValueError("formal manifest hash binding differs")
        frozen = {**cell, "config_path": config_path, "trace_path": trace_path}
        if policy == "slack":
            slack_manifest = _path(cell["manifest_path"], field="manifest_path", directory=REPO_ROOT / "experiments/configs")
            if slack_manifest.resolve() != _slack_manifest_path(seed).resolve() or sha256(slack_manifest) != cell["manifest_sha256"]:
                raise ValueError("formal manifest slack binding differs")
            frozen["manifest_path"] = slack_manifest
        plan[key] = frozen
    if set(plan) != expected_keys:
        raise ValueError("formal manifest required cells differ")
    return plan


def _normalized_pair(config: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(config)
    value["admission"]["policy"] = "<policy>"
    for field in (
        "eta_prefill_seconds",
        "eta_decode_seconds_per_token",
        "eta_safety_margin_seconds",
    ):
        value["admission"][field] = "<policy-eta>"
    value["output"]["run_id"] = "<run_id>"
    return value


def validate_matched_configs(plan: dict[tuple[str, int], dict[str, Any]]) -> None:
    for seed in SEEDS:
        prefix = json.loads(plan[("prefix", seed)]["config_path"].read_text(encoding="utf-8"))
        slack = json.loads(plan[("slack", seed)]["config_path"].read_text(encoding="utf-8"))
        for policy, config in (("prefix", prefix), ("slack", slack)):
            if (
                config.get("schema_version") != 1
                or config.get("output", {}).get("run_id") != _expected_run_id(policy, seed)
                or config.get("workload", {}).get("trace_path")
                != str(_trace_path(seed).relative_to(REPO_ROOT)).replace("\\", "/")
                or config.get("admission", {}).get("policy") != POLICIES[policy]
                or config.get("engine", {}).get("num_kvcache_blocks") != 6
                or config.get("engine", {}).get("kvcache_block_size") != 256
                or config.get("measurement", {}).get("max_run_seconds") != 30.0
            ):
                raise ValueError("formal config protocol differs")
        if any(prefix["admission"].get(field) != 0.0 for field in (
            "eta_prefill_seconds", "eta_decode_seconds_per_token", "eta_safety_margin_seconds"
        )):
            raise ValueError("prefix control ETA differs")
        if _normalized_pair(prefix) != _normalized_pair(slack):
            raise ValueError("matched policy-pair config differs")


def bind_historical_execution(
    plan: dict[tuple[str, int], dict[str, Any]],
) -> None:
    for seed in SEEDS:
        slack = plan[("slack", seed)]
        frozen = json.loads(
            slack["manifest_path"].read_text(encoding="utf-8")
        )
        execution_code = frozen.get("execution_code")
        if (
            not isinstance(execution_code, dict)
            or set(execution_code) != set(CODE_PATHS)
        ):
            raise ValueError("formal historical execution binding differs")
        plan[("prefix", seed)]["execution_code"] = execution_code
        slack["execution_code"] = execution_code


def _completion_gate(summary: dict[str, Any], requests: list[dict[str, Any]], config: dict[str, Any]) -> None:
    terminal = summary.get("summary", {}).get("terminal_counts")
    counts = Counter(item.get("terminal_state") for item in requests)
    states = ("Finished", "Rejected", "Failed", "Cancelled", "Unfinished")
    if (
        not isinstance(terminal, dict)
        or terminal.get("submitted") != len(requests)
        or terminal.get("reconciled") is not True
        or any(terminal.get(state) != counts[state] for state in states)
        or counts["Failed"]
        or counts["Cancelled"]
        or counts["Unfinished"]
        or summary.get("summary", {}).get("runtime", {}).get("timed_out") is not False
        or summary.get("pressure", {}).get("oom_detected") is not False
        or summary.get("runtime", {}).get("cuda_available") is not True
        or summary.get("pressure", {}).get("kv_cache", {}).get("final_used_blocks") != 0
        or summary.get("admission", {}).get("final_reserved_blocks") != 0
    ):
        raise ValueError("formal completion gate differs")
    validate_pressure(summary, config["engine"]["num_kvcache_blocks"], counts["Rejected"])


def validate_prefix_run(*, cell: dict[str, Any]) -> dict[str, Any]:
    """Replay the prefix-FIFO control with the same strictness as treatment."""
    config_path, trace_path = cell["config_path"], cell["trace_path"]
    expected_execution_code = cell.get(
        "execution_code",
        execution_code_fingerprint(),
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_id = config["output"]["run_id"]
    summary_path = RESULTS / f"{run_id}.summary.json"
    if not summary_path.is_file():
        raise ValueError(f"formal summary is missing: {run_id}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "passed" or summary.get("error") is not None or summary.get("stage12_manifest") is not None:
        raise ValueError(f"prefix formal status differs: {run_id}")
    configured_trace = Path(config.get("workload", {}).get("trace_path", ""))
    if not configured_trace.is_absolute():
        configured_trace = REPO_ROOT / configured_trace
    if (
        sha256(config_path) != cell["config_sha256"]
        or configured_trace.resolve() != trace_path.resolve()
        or sha256(configured_trace) != cell["trace_sha256"]
        or summary.get("provenance", {}).get("config_sha256") != cell["config_sha256"]
        or summary.get("provenance", {}).get("trace_sha256") != cell["trace_sha256"]
        or {key: summary.get("provenance", {}).get(key) for key in CODE_PATHS}
        != expected_execution_code
        or summary.get("upstream_commit") != config.get("upstream_commit")
    ):
        raise ValueError(f"prefix formal provenance differs: {run_id}")
    for field in ("engine", "sampling", "slo", "measurement"):
        if summary.get(field) != config.get(field):
            raise ValueError(f"prefix formal {field} differs: {run_id}")
    expected_prefix_admission = {
        "policy": config["admission"]["policy"],
        "max_queue_wait_seconds": config["admission"]["max_queue_wait_seconds"],
        "observe_prefix_cache": config["admission"]["observe_prefix_cache"],
    }
    if (
        any(summary.get("model", {}).get(field) != config["model"].get(field) for field in ("repo_id", "revision", "sha256"))
        or summary.get("model", {}).get("verified_sha256") != config["model"].get("sha256")
        or any(summary.get("admission", {}).get(key) != value for key, value in expected_prefix_admission.items())
        or any(config["admission"].get(key) != 0.0 for key in (
            "eta_prefill_seconds", "eta_decode_seconds_per_token", "eta_safety_margin_seconds"
        ))
    ):
        raise ValueError(f"prefix formal config binding differs: {run_id}")
    artifacts = summary.get("artifacts", {})
    paths = {
        "requests": RESULTS / f"{run_id}.requests.jsonl",
        "steps": RESULTS / f"{run_id}.steps.jsonl",
        "admission_events": RESULTS / f"{run_id}.admission.jsonl",
        "cache_events": RESULTS / f"{run_id}.cache.jsonl",
        "cache_states": RESULTS / f"{run_id}.cache_states.jsonl",
    }
    documents: dict[str, list[dict[str, Any]]] = {}
    for name, path in paths.items():
        if (
            not recorded_path_matches(
                artifacts.get(f"{name}_path"),
                path,
                repository_root=REPO_ROOT,
                allowed_directory=RESULTS,
                label=f"prefix {name}",
            )
            or artifacts.get(f"{name}_sha256") != sha256(path)
        ):
            raise ValueError(f"prefix raw artifact hash differs: {run_id}:{name}")
        documents[name] = read_jsonl(path)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    validate_raw(trace=trace, requests=documents["requests"], steps=documents["steps"], summary=summary)
    replay = validate_events(
        trace=trace,
        requests=documents["requests"],
        admission_events=documents["admission_events"],
        cache_events=documents["cache_events"],
        cache_states=documents["cache_states"],
        admission=summary["admission"],
        policy="prefix_aware_fifo",
        block_size=config["engine"]["kvcache_block_size"],
    )
    if any(summary["admission"].get(key) != value for key, value in replay.items()):
        raise ValueError(f"prefix admission replay differs: {run_id}")
    _completion_gate(summary, documents["requests"], config)
    return summary


def _metrics(summary: dict[str, Any]) -> dict[str, float | None]:
    body = summary["summary"]
    return {
        "interactive_slo_goodput_rps": body["interactive"]["slo_goodput_rps"],
        "interactive_ttft_p99_ms": body["interactive"]["ttft_ms"]["p99"],
        "interactive_itl_p99_ms": body["interactive"]["itl_ms"]["p99"],
        "long_window_token_goodput_tps": body["long"]["token_goodput_tps"],
        "output_throughput_tps": body["output_throughput_tps"],
        "elapsed_seconds": body["runtime"]["elapsed_seconds"],
    }


def _mean_std(values: list[float | None]) -> dict[str, float | None]:
    if all(value is None for value in values):
        return {"mean": None, "sample_std": None}
    if any(value is None for value in values):
        raise ValueError("metric nullability differs across matched repetitions")
    numeric = [float(value) for value in values]
    return {
        "mean": statistics.mean(numeric),
        "sample_std": statistics.stdev(numeric) if len(numeric) > 1 else 0.0,
    }


def aggregate(manifest_path: Path = FORMAL_MANIFEST) -> dict[str, Any]:
    if (
        manifest_path.resolve() != FORMAL_MANIFEST.resolve()
        or sha256(manifest_path) != FORMAL_MANIFEST_SHA256
    ):
        raise ValueError("formal aggregation requires the approved r3 manifest")
    plan = load_formal_plan(manifest_path)
    validate_matched_configs(plan)
    bind_historical_execution(plan)
    historical_execution = plan[("slack", SEEDS[0])]["execution_code"]
    if any(
        cell["execution_code"] != historical_execution
        for cell in plan.values()
    ):
        raise ValueError("formal historical execution differs across cells")
    records: list[dict[str, Any]] = []
    for seed in SEEDS:
        for name, policy in POLICIES.items():
            cell = plan[(name, seed)]
            if name == "slack":
                validate_run_artifacts(
                    config_path=cell["config_path"],
                    summary_path=RESULTS / f"{_expected_run_id(name, seed)}.summary.json",
                    manifest_path=cell["manifest_path"],
                )
                summary = json.loads((RESULTS / f"{_expected_run_id(name, seed)}.summary.json").read_text(encoding="utf-8"))
            else:
                summary = validate_prefix_run(cell=cell)
            requests = read_jsonl(RESULTS / f"{_expected_run_id(name, seed)}.requests.jsonl")
            records.append({
                "seed": seed,
                "policy": policy,
                "run_id": _expected_run_id(name, seed),
                "terminal_counts": summary["summary"]["terminal_counts"],
                "admission": summary["admission"],
                "metrics": _metrics(summary),
                "early_rejections": sum(
                    item.get("terminal_reason") == "predicted_deadline_miss" for item in requests
                ),
            })
    by_policy = {}
    for policy in POLICIES.values():
        selected = [item for item in records if item["policy"] == policy]
        by_policy[policy] = {
            key: _mean_std([item["metrics"][key] for item in selected]) for key in METRICS
        } | {
            "mean_rejected_requests": statistics.mean(item["terminal_counts"]["Rejected"] for item in selected),
            "mean_early_rejections": statistics.mean(item["early_rejections"] for item in selected),
        }
    pairs = []
    matched_metric_coverage = {
        key: {"paired_non_null": 0, "total_pairs": len(SEEDS), "comparable": False}
        for key in METRICS
    }
    for seed in SEEDS:
        prefix = next(item for item in records if item["seed"] == seed and item["policy"] == POLICIES["prefix"])
        slack = next(item for item in records if item["seed"] == seed and item["policy"] == POLICIES["slack"])
        deltas: dict[str, float | None] = {}
        for key in METRICS:
            if slack["metrics"][key] is not None and prefix["metrics"][key] is not None:
                matched_metric_coverage[key]["paired_non_null"] += 1
                deltas[key] = float(slack["metrics"][key]) - float(prefix["metrics"][key])
            else:
                deltas[key] = None
        pairs.append({
            "seed": seed,
            "slack_minus_prefix": deltas | {
                "rejected_requests": slack["terminal_counts"]["Rejected"] - prefix["terminal_counts"]["Rejected"],
                "early_rejections": slack["early_rejections"] - prefix["early_rejections"],
            },
        })
    for coverage in matched_metric_coverage.values():
        coverage["comparable"] = coverage["paired_non_null"] == coverage["total_pairs"]
    return {
        "schema_version": 1,
        "stage": "stage12",
        "revision": "r3",
        "required_cells": 6,
        "validated_cells": len(records),
        "formal_manifest_sha256": sha256(manifest_path),
        "execution_code": historical_execution,
        "aggregation_code": aggregation_fingerprint(),
        "runs": records,
        "by_policy": by_policy,
        "matched_pairs": pairs,
        "matched_metric_coverage": matched_metric_coverage,
        "decision": (
            "This is a matched single-GPU admission-control comparison. Early "
            "rejections are a mechanism count, not a counterfactual accuracy claim; "
            "interpret them together with terminal, SLO, latency, and rejection metrics. "
            "A matched delta is eligible for cross-policy interpretation only when "
            "matched_metric_coverage[metric].comparable is true."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate frozen Stage 12 r3 formal evidence.")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(RESULTS.resolve()):
        raise ValueError("aggregate output must be inside experiments/results")
    value = aggregate()
    value["aggregate_sha256"] = sha256(Path(__file__))
    encoded = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if output.exists() and output.read_text(encoding="utf-8") != encoded:
        raise FileExistsError("aggregate output differs")
    output.write_text(encoded, encoding="utf-8")
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
