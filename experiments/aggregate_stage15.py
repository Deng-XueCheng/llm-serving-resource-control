from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from pathlib import Path, PurePath
from typing import Any

from experiments.benchmark.calibration_evidence import (
    read_jsonl,
    recorded_path_is_absolute,
    recompute_metrics,
    resolve_recorded_artifact_path,
    sha256_file,
    validate_requests,
    validate_runtime,
    validate_steps,
    validate_terminal_counts,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CAPACITIES = (4, 6, 8)
SEEDS = (1, 2, 3)
POLICIES = {
    "pressure": "pressure_aware_decode",
    "recompute": "recompute_aware",
}
CODE_FINGERPRINT_KEYS = (
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


def _resolve_recorded_artifact_path(
    value: Any,
    *,
    label: str,
    allowed_directory: Path,
) -> Path:
    try:
        return resolve_recorded_artifact_path(
            value,
            repository_root=REPO_ROOT,
            allowed_directory=allowed_directory,
            label=f"Stage 15 {label} artifact",
        )
    except ValueError as exception:
        if isinstance(value, str):
            filename = PurePath(value.replace("\\", "/")).name
            migrated = (
                REPO_ROOT
                / "experiments/results/final/scheduler/stage15"
                / filename
            ).resolve()
            if (
                filename not in {"", ".", ".."}
                and migrated.is_relative_to(allowed_directory.resolve())
                and migrated.is_file()
            ):
                return migrated
        raise ValueError(f"Stage 15 {label} artifact is missing") from exception


def _resolve_config_trace_path(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Stage 15 config trace path is invalid")
    if recorded_path_is_absolute(value):
        raise ValueError("Stage 15 config trace path must be relative")
    path = (REPO_ROOT / Path(value)).resolve()
    if (
        not path.is_relative_to(REPO_ROOT.resolve())
        or not path.is_relative_to((REPO_ROOT / "experiments/data").resolve())
        or not path.is_file()
    ):
        raise ValueError("Stage 15 config trace path is invalid")
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exception:
                raise ValueError(
                    f"Invalid step JSON at line {line_number}"
                ) from exception
            if not isinstance(row, dict):
                raise ValueError("Step JSONL rows must be objects")
            rows.append(row)
    if not rows:
        raise ValueError("Step artifact is empty")
    return rows


def raw_recompute_metrics(
    step_path: Path,
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    steps = _read_jsonl(step_path)
    initial_prefill_tokens = 0
    actual_recompute_tokens = 0
    preemption_count = 0
    resume_count = 0
    pending_recompute: set[int] = set()

    for expected_index, step in enumerate(steps):
        if step.get("step_index") != expected_index:
            raise ValueError("Scheduler step indexes are not contiguous")
        phase = step.get("phase")
        if phase not in {"prefill", "decode"}:
            raise ValueError("Unknown scheduler step phase")
        events = step.get("events")
        scheduler = step.get("scheduler")
        if not isinstance(events, list) or not isinstance(scheduler, dict):
            raise ValueError("Stage 15 steps require events and scheduler")
        if any(event.get("phase") != phase for event in events):
            raise ValueError("Step/event phase fields differ")
        event_tokens = sum(
            event.get("num_scheduled_tokens", -1) for event in events
        )
        if event_tokens != step.get("num_scheduled_tokens"):
            raise ValueError("Step/event token totals differ")

        selected_key = (
            "selected_prefill_ids" if phase == "prefill"
            else "selected_decode_ids"
        )
        selected = scheduler.get(selected_key)
        event_ids = [event.get("seq_id") for event in events]
        if selected != event_ids:
            raise ValueError("Scheduler selected IDs differ from step events")

        preemptions = scheduler.get("preemptions")
        if not isinstance(preemptions, list):
            raise ValueError("Scheduler preemptions must be a list")
        for preemption in preemptions:
            seq_id = preemption.get("seq_id")
            if not isinstance(seq_id, int):
                raise ValueError("Preemption sequence ID must be an integer")
            if seq_id in pending_recompute:
                raise ValueError("Sequence preempted twice before resume")
            pending_recompute.add(seq_id)
            preemption_count += 1

        for event in events:
            kind = event.get("prefill_kind")
            recompute_tokens = event.get("actual_recompute_tokens")
            resumed = event.get("resumed")
            seq_id = event.get("seq_id")
            scheduled_tokens = event.get("num_scheduled_tokens")
            if phase == "decode":
                if kind is not None or recompute_tokens not in {None, 0}:
                    raise ValueError("Decode event carries Prefill accounting")
                if resumed not in {None, False}:
                    raise ValueError("Decode event cannot resume Prefill")
                continue
            if kind not in {"initial_prefill", "recompute_prefill"}:
                raise ValueError("Prefill event has invalid prefill_kind")
            if not isinstance(recompute_tokens, int) or recompute_tokens < 0:
                raise ValueError("Invalid actual recompute token count")
            if not isinstance(resumed, bool):
                raise ValueError("Prefill resumed flag must be boolean")
            if kind == "initial_prefill":
                if recompute_tokens != 0 or resumed:
                    raise ValueError("Initial Prefill cannot report recompute")
                if seq_id in pending_recompute:
                    raise ValueError("Pending sequence mislabeled as initial")
                initial_prefill_tokens += scheduled_tokens
                continue
            if seq_id not in pending_recompute:
                raise ValueError("Recompute/resume lacks prior preemption")
            if recompute_tokens != scheduled_tokens:
                raise ValueError("Raw recompute tokens differ from execution")
            actual_recompute_tokens += recompute_tokens
            if resumed:
                pending_recompute.remove(seq_id)
                resume_count += 1

    pending_recompute_seq_ids = sorted(pending_recompute)
    if require_complete and pending_recompute_seq_ids:
        raise ValueError(
            "Raw artifact ends with pending recompute sequences that did "
            "not resume"
        )
    if initial_prefill_tokens <= 0:
        raise ValueError("Initial Prefill token denominator is zero")
    total_prefill_tokens = (
        initial_prefill_tokens + actual_recompute_tokens
    )
    return {
        "initial_prefill_tokens": initial_prefill_tokens,
        "actual_recompute_tokens": actual_recompute_tokens,
        "total_prefill_tokens": total_prefill_tokens,
        "preemption_count": preemption_count,
        "resume_count": resume_count,
        "complete_recompute_lifecycle": not pending_recompute_seq_ids,
        "pending_recompute_seq_ids": pending_recompute_seq_ids,
        "recompute_amplification": (
            total_prefill_tokens / initial_prefill_tokens
        ),
        "extra_prefill_ratio": (
            actual_recompute_tokens / initial_prefill_tokens
        ),
        "recompute_tokens_per_preemption": (
            actual_recompute_tokens / preemption_count
            if preemption_count
            else 0.0
        ),
    }


def load_validated_stage15_run(
    summary_path: Path,
    *,
    expected_policy: str | None = None,
    expected_capacity: int | None = None,
) -> dict[str, Any]:
    summary_path = summary_path.resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "passed":
        raise ValueError("Stage 15 run status is not passed")

    try:
        paths = {
            "config": _resolve_recorded_artifact_path(
                summary["config_path"],
                label="config",
                allowed_directory=REPO_ROOT / "experiments/configs",
            ),
            "trace": _resolve_recorded_artifact_path(
                summary["provenance"]["trace_path"],
                label="trace",
                allowed_directory=REPO_ROOT / "experiments/data",
            ),
            "requests": _resolve_recorded_artifact_path(
                summary["artifacts"]["requests_path"],
                label="requests",
                allowed_directory=REPO_ROOT / "experiments/results",
            ),
            "steps": _resolve_recorded_artifact_path(
                summary["artifacts"]["steps_path"],
                label="steps",
                allowed_directory=REPO_ROOT / "experiments/results",
            ),
        }
        expected_hashes = {
            "config": summary["provenance"]["config_sha256"],
            "trace": summary["provenance"]["trace_sha256"],
            "requests": summary["artifacts"]["requests_sha256"],
            "steps": summary["artifacts"]["steps_sha256"],
        }
    except (KeyError, TypeError) as exception:
        raise ValueError("Stage 15 artifact binding is incomplete") from exception
    for label, path in paths.items():
        if sha256_file(path) != expected_hashes[label]:
            raise ValueError(f"Stage 15 {label} artifact hash differs")

    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    trace = json.loads(paths["trace"].read_text(encoding="utf-8"))
    requests = read_jsonl(paths["requests"])
    steps = read_jsonl(paths["steps"])
    if config.get("output", {}).get("run_id") != summary_path.name.removesuffix(
        ".summary.json"
    ):
        raise ValueError("Stage 15 run ID is not bound to summary filename")
    expected_engine = copy.deepcopy(config.get("engine"))
    if not isinstance(expected_engine, dict):
        raise ValueError("Stage 15 config engine is missing")
    expected_engine.setdefault("cuda_event_timing", False)
    if expected_engine != summary.get("engine"):
        raise ValueError("Stage 15 config and summary engine differ")
    for field in ("sampling", "slo", "measurement"):
        if config.get(field) != summary.get(field):
            raise ValueError(f"Stage 15 config and summary {field} differ")
    if config.get("upstream_commit") != summary.get("upstream_commit"):
        raise ValueError("Stage 15 config and summary upstream differ")
    for field in ("repo_id", "revision", "sha256"):
        if config.get("model", {}).get(field) != summary.get("model", {}).get(
            field
        ):
            raise ValueError(f"Stage 15 config and model {field} differ")
    if summary.get("model", {}).get("verified_sha256") != summary.get(
        "model", {}
    ).get("sha256"):
        raise ValueError("Stage 15 model hashes were not verified")
    trace_path = _resolve_config_trace_path(config["workload"]["trace_path"])
    if trace_path != paths["trace"]:
        raise ValueError("Stage 15 config and summary trace paths differ")

    policy = config["engine"].get("scheduler_policy")
    capacity = config["engine"].get("num_kvcache_blocks")
    if expected_policy is not None and policy != expected_policy:
        raise ValueError("Stage 15 scheduler policy differs")
    if expected_capacity is not None and capacity != expected_capacity:
        raise ValueError("Stage 15 KV capacity differs")

    validate_terminal_counts(summary, requests)
    indexed_requests = validate_requests(trace, requests)
    validate_steps(
        steps,
        indexed_requests,
        summary["summary"]["runtime"]["steps"],
    )
    elapsed = validate_runtime(summary, requests)
    start = float(summary["measurement"]["start_seconds"])
    end = float(summary["measurement"]["end_seconds"])
    offered_tokens = sum(
        request["max_output_tokens"]
        for request in trace["requests"]
        if start <= float(request["arrival_time"]) < end
    )
    serving_metrics = recompute_metrics(
        summary,
        trace,
        requests,
        offered_tokens / (end - start),
    )

    request_id_by_seq_id: dict[int, str] = {}
    for step in steps:
        for event in step["events"]:
            seq_id = event["seq_id"]
            request_id = event["request_id"]
            previous = request_id_by_seq_id.setdefault(seq_id, request_id)
            if previous != request_id:
                raise ValueError("Stage 15 seq/request mapping is not stable")
    for step in steps:
        scheduler = step["scheduler"]
        if scheduler["policy"] != policy:
            raise ValueError("Stage 15 raw scheduler policy differs")
        active = (
            "prefill" if step["phase"] == "prefill" else "decode"
        )
        inactive = "decode" if active == "prefill" else "prefill"
        event_seq_ids = [event["seq_id"] for event in step["events"]]
        event_request_ids = [
            event["request_id"] for event in step["events"]
        ]
        if (
            scheduler[f"selected_{active}_ids"] != event_seq_ids
            or scheduler[f"selected_{active}_request_ids"]
            != event_request_ids
            or scheduler[f"selected_{inactive}_ids"]
            or scheduler[f"selected_{inactive}_request_ids"]
        ):
            raise ValueError("Stage 15 raw scheduler selection differs")
        for item in scheduler["resident_costs"]:
            if request_id_by_seq_id.get(item["seq_id"]) != item["request_id"]:
                raise ValueError("Stage 15 resident-cost identity differs")
        for item in scheduler["preemptions"]:
            if request_id_by_seq_id.get(item["seq_id"]) != item["request_id"]:
                raise ValueError("Stage 15 preemption identity differs")
            triggering_seq_id = item["triggering_seq_id"]
            expected_triggering_request = (
                None
                if triggering_seq_id is None
                else request_id_by_seq_id.get(triggering_seq_id)
            )
            if item["triggering_request_id"] != expected_triggering_request:
                raise ValueError(
                    "Stage 15 triggering request identity differs"
                )

    pressure = summary.get("pressure")
    if (
        not isinstance(pressure, dict)
        or pressure.get("observation_complete") is not True
        or pressure.get("oom_detected") is not False
        or pressure.get("kv_cache", {}).get("final_used_blocks") != 0
    ):
        raise ValueError("Stage 15 pressure completion gate failed")
    metrics = raw_recompute_metrics(paths["steps"], require_complete=True)
    observability = summary.get("recompute_observability")
    if not isinstance(observability, dict) or observability.get(
        "schema_version"
    ) != 1:
        raise ValueError("Stage 15 recompute observability is missing")
    counter_pairs = {
        "preemption_count": pressure.get("preemption_count"),
        "actual_recompute_tokens": observability.get(
            "actual_recompute_tokens"
        ),
        "resume_count": observability.get("resume_count"),
    }
    if observability.get("preemption_count") != pressure.get(
        "preemption_count"
    ):
        raise ValueError("Stage 15 summary preemption counters differ")
    for field, reported in counter_pairs.items():
        if metrics[field] != reported:
            raise ValueError(f"Stage 15 raw and summary {field} differ")
    if policy == "recompute_aware":
        policy_summary = summary.get("recompute_aware")
        if not isinstance(policy_summary, dict):
            raise ValueError("Stage 15 recompute-aware summary is missing")
        for field in ("actual_recompute_tokens", "resume_count"):
            if policy_summary.get(field) != metrics[field]:
                raise ValueError(
                    f"Stage 15 policy and raw {field} differ"
                )

    return {
        "summary": summary,
        "config": config,
        "trace": trace,
        "requests": requests,
        "steps": steps,
        "paths": paths,
        "elapsed_seconds": elapsed,
        "serving_metrics": serving_metrics,
        "recompute_metrics": metrics,
    }


def validate_stage15_manifest(manifest: dict[str, Any]) -> None:
    revision = manifest.get("revision")
    if (
        manifest.get("schema_version") != 1
        or revision
        not in {
            "stage15_diagnostic_r1",
            "stage15_diagnostic_r2",
            "stage15_diagnostic_r3",
        }
        or tuple(manifest.get("capacities", ())) != CAPACITIES
        or tuple(manifest.get("seeds", ())) != SEEDS
        or manifest.get("policies") != list(POLICIES)
        or manifest.get("required_cells") != 18
        or manifest.get("controlled_difference")
        != ["engine.scheduler_policy", "output.run_id"]
        or len(manifest.get("cells", ())) != 18
    ):
        raise ValueError("Stage 15 manifest contract differs")
    expected_cells = {
        (capacity, seed, policy)
        for capacity in CAPACITIES
        for seed in SEEDS
        for policy in POLICIES
    }
    actual_cells = {
        (cell.get("capacity"), cell.get("seed"), cell.get("policy"))
        for cell in manifest["cells"]
    }
    if actual_cells != expected_cells:
        raise ValueError("Stage 15 manifest cells differ")
    if manifest.get("completion_gates") != {
        "status": "passed",
        "all_requests_finished": True,
        "terminal_reconciled": True,
        "timed_out": False,
        "oom_detected": False,
        "final_kv_used_blocks": 0,
        "raw_summary_counters_match": True,
    }:
        raise ValueError("Stage 15 completion gate contract differs")
    if manifest.get("acceptance") != {
        "aggregate_actual_recompute_tokens_reduction_min": 0.20,
        "improved_matched_pairs_min": 6,
        "matched_pairs": 9,
    }:
        raise ValueError("Stage 15 acceptance contract differs")
    if revision in {"stage15_diagnostic_r2", "stage15_diagnostic_r3"} and any(
        set(cell)
        != {
            "capacity",
            "seed",
            "policy",
            "run_id",
            "config_path",
            "source_config_path",
            "config_sha256",
            "source_config_sha256",
        }
        for cell in manifest["cells"]
    ):
        raise ValueError("Stage 15 r2 manifest hash binding differs")


def normalized_pair_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(config)
    normalized["engine"]["scheduler_policy"] = "<policy>"
    normalized["output"]["run_id"] = "<run-id>"
    return normalized


def _relative_change(candidate: float, baseline: float) -> float | None:
    if baseline == 0:
        return None
    return (candidate - baseline) / baseline


def _run_metrics(record: dict[str, Any]) -> dict[str, Any]:
    summary = record["summary"]
    serving = record["serving_metrics"]
    recompute = record["recompute_metrics"]
    return {
        "actual_recompute_tokens": recompute["actual_recompute_tokens"],
        "recompute_amplification": recompute["recompute_amplification"],
        "recompute_tokens_per_preemption": recompute[
            "recompute_tokens_per_preemption"
        ],
        "preemption_count": recompute["preemption_count"],
        "resume_count": recompute["resume_count"],
        "ttft_p99_ms": serving["interactive_ttft"]["p99"],
        "itl_p99_ms": serving["interactive_itl"]["p99"],
        "interactive_slo_goodput_rps": summary["summary"]["interactive"][
            "slo_goodput_rps"
        ],
        "output_throughput_tps": serving[
            "achieved_output_throughput_tps"
        ],
        "elapsed_seconds": record["elapsed_seconds"],
        "steps": summary["summary"]["runtime"]["steps"],
    }


def aggregate_stage15(
    results_directory: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_stage15_manifest(manifest)
    records: dict[tuple[int, int, str], dict[str, Any]] = {}
    code_fingerprints = set()
    repository_heads = set()
    runtime_fingerprints = set()
    model_fingerprints = set()
    for cell in manifest["cells"]:
        capacity = cell["capacity"]
        seed = cell["seed"]
        policy_name = cell["policy"]
        run_id = cell["run_id"]
        expected_config_path = (REPO_ROOT / cell["config_path"]).resolve()
        source_config_path = (
            REPO_ROOT / cell["source_config_path"]
        ).resolve()
        if "config_sha256" in cell and sha256_file(
            expected_config_path
        ) != cell["config_sha256"]:
            raise ValueError("Stage 15 frozen config hash differs")
        if "source_config_sha256" in cell and sha256_file(
            source_config_path
        ) != cell["source_config_sha256"]:
            raise ValueError("Stage 15 frozen source hash differs")
        summary_path = (results_directory / f"{run_id}.summary.json").resolve()
        record = load_validated_stage15_run(
            summary_path,
            expected_policy=POLICIES[policy_name],
            expected_capacity=capacity,
        )
        if record["paths"]["config"] != expected_config_path:
            raise ValueError("Stage 15 config path differs from manifest")
        expected_config = json.loads(
            source_config_path.read_text(encoding="utf-8")
        )
        expected_config["engine"]["scheduler_policy"] = POLICIES[
            policy_name
        ]
        expected_config["output"]["run_id"] = run_id
        if record["config"] != expected_config:
            raise ValueError("Stage 15 config differs from frozen source")
        provenance = record["summary"]["provenance"]
        fingerprint = tuple(provenance[key] for key in CODE_FINGERPRINT_KEYS)
        code_fingerprints.add(fingerprint)
        repository_heads.add(record["summary"]["repository_head"])
        runtime_fingerprints.add(
            json.dumps(record["summary"]["runtime"], sort_keys=True)
        )
        model_fingerprints.add(
            json.dumps(record["summary"]["model"], sort_keys=True)
        )
        records[(capacity, seed, policy_name)] = {
            **record,
            "metrics": _run_metrics(record),
        }
    if (
        len(records) != 18
        or len(code_fingerprints) != 1
        or len(repository_heads) != 1
        or len(runtime_fingerprints) != 1
        or len(model_fingerprints) != 1
    ):
        raise ValueError("Stage 15 matrix code/cell consistency differs")

    pairs = []
    for capacity in CAPACITIES:
        for seed in SEEDS:
            baseline = records[(capacity, seed, "pressure")]
            candidate = records[(capacity, seed, "recompute")]
            if (
                normalized_pair_config(baseline["config"])
                != normalized_pair_config(candidate["config"])
                or baseline["summary"]["provenance"]["trace_sha256"]
                != candidate["summary"]["provenance"]["trace_sha256"]
            ):
                raise ValueError("Stage 15 matched pair contract differs")
            baseline_metrics = baseline["metrics"]
            candidate_metrics = candidate["metrics"]
            pairs.append(
                {
                    "capacity": capacity,
                    "seed": seed,
                    "baseline": baseline_metrics,
                    "candidate": candidate_metrics,
                    "relative_change_candidate_vs_baseline": {
                        key: _relative_change(
                            float(candidate_metrics[key]),
                            float(baseline_metrics[key]),
                        )
                        for key in baseline_metrics
                    },
                }
            )

    totals = {
        policy: {
            metric: sum(
                float(records[(capacity, seed, policy)]["metrics"][metric])
                for capacity in CAPACITIES
                for seed in SEEDS
            )
            for metric in ("actual_recompute_tokens", "preemption_count")
        }
        for policy in POLICIES
    }
    gates = {}
    for metric in ("actual_recompute_tokens", "preemption_count"):
        baseline_total = totals["pressure"][metric]
        candidate_total = totals["recompute"][metric]
        reduction = (
            None
            if baseline_total == 0
            else (baseline_total - candidate_total) / baseline_total
        )
        improved_pairs = sum(
            pair["candidate"][metric] < pair["baseline"][metric]
            for pair in pairs
        )
        gates[metric] = {
            "baseline_total": baseline_total,
            "candidate_total": candidate_total,
            "reduction": reduction,
            "improved_pairs": improved_pairs,
            "passed": (
                reduction is not None
                and reduction >= 0.20
                and improved_pairs >= 6
            ),
        }
    return {
        "schema_version": 1,
        "revision": manifest["revision"],
        "matrix": {"runs": 18, "matched_pairs": 9},
        "completion_gates_validated": True,
        "raw_metrics_recomputed": True,
        "execution_code_fingerprint": dict(
            zip(CODE_FINGERPRINT_KEYS, next(iter(code_fingerprints)))
        ),
        "pairs": pairs,
        "aggregate_gates": gates,
        "accepted": any(gate["passed"] for gate in gates.values()),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT
        / "experiments/configs/stage15_diagnostic_r1_matrix.json",
    )
    parser.add_argument(
        "--results-directory",
        type=Path,
        default=REPO_ROOT / "experiments/results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "experiments/results/stage15_diagnostic_r1.aggregate.json",
    )
    args = parser.parse_args()
    aggregate = aggregate_stage15(
        args.results_directory.resolve(),
        args.manifest.resolve(),
    )
    write_json_atomic(args.output.resolve(), aggregate)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
