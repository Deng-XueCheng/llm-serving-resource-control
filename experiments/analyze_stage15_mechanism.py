from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments.aggregate_stage15 import (
    REPO_ROOT,
    aggregate_stage15,
    load_validated_stage15_run,
    sha256_file,
    write_json_atomic,
)


def bucket_steps(
    steps: list[dict[str, Any]],
    *,
    bucket_seconds: float,
) -> list[dict[str, Any]]:
    buckets: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "steps": 0,
            "prefill_steps": 0,
            "decode_steps": 0,
            "preemptions": 0,
            "recompute_tokens": 0,
            "finished_requests": 0,
            "kv_utilization_sum": 0.0,
            "kv_utilization_max": 0.0,
            "waiting_requests_sum": 0,
            "waiting_requests_max": 0,
            "oldest_waiting_age_max": 0,
            "modes": Counter(),
        }
    )
    for step in steps:
        index = int(float(step["started_at"]) // bucket_seconds)
        row = buckets[index]
        scheduler = step["scheduler"]
        utilization = (
            scheduler["kv_used_blocks_before"]
            / scheduler["kv_total_blocks"]
        )
        waiting = len(scheduler["waiting_ids_before"])
        row["steps"] += 1
        row[f"{step['phase']}_steps"] += 1
        row["preemptions"] += len(scheduler["preemptions"])
        row["recompute_tokens"] += sum(
            event.get("actual_recompute_tokens", 0)
            for event in step["events"]
        )
        row["finished_requests"] += sum(
            event["finished"] for event in step["events"]
        )
        row["kv_utilization_sum"] += utilization
        row["kv_utilization_max"] = max(
            row["kv_utilization_max"],
            utilization,
        )
        row["waiting_requests_sum"] += waiting
        row["waiting_requests_max"] = max(
            row["waiting_requests_max"],
            waiting,
        )
        row["oldest_waiting_age_max"] = max(
            row["oldest_waiting_age_max"],
            scheduler["oldest_waiting_age"],
        )
        row["modes"][scheduler["mode"]] += 1
    result = []
    for index in sorted(buckets):
        row = buckets[index]
        steps = row.pop("steps")
        result.append(
            {
                "bucket_start_s": index * bucket_seconds,
                "bucket_end_s": (index + 1) * bucket_seconds,
                "steps": steps,
                "kv_utilization_mean": row.pop("kv_utilization_sum")
                / steps,
                "waiting_requests_mean": row.pop("waiting_requests_sum")
                / steps,
                **row,
                "modes": dict(row["modes"]),
            }
        )
    return result


def mode_counts(steps: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(step["scheduler"]["mode"] for step in steps))


def analyze_pair(
    *,
    capacity: int,
    seed: int,
    revision: str,
    results_directory: Path,
    manifest_path: Path,
    bucket_seconds: float,
) -> dict[str, Any]:
    if not math.isfinite(bucket_seconds) or bucket_seconds <= 0:
        raise ValueError("bucket_seconds must be finite and positive")
    aggregate = aggregate_stage15(
        results_directory.resolve(),
        manifest_path.resolve(),
    )
    if aggregate["revision"] != revision:
        raise ValueError("Stage 15 analysis revision differs from manifest")
    matched_pairs = [
        pair
        for pair in aggregate["pairs"]
        if pair["capacity"] == capacity and pair["seed"] == seed
    ]
    if len(matched_pairs) != 1:
        raise ValueError("Stage 15 analysis pair is not in frozen matrix")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pair_cells = {
        cell["policy"]: cell
        for cell in manifest["cells"]
        if cell["capacity"] == capacity and cell["seed"] == seed
    }
    if set(pair_cells) != {"pressure", "recompute"}:
        raise ValueError("Stage 15 manifest pair is incomplete")

    records = {}
    for label, policy_label, policy in (
        ("baseline", "pressure", "pressure_aware_decode"),
        ("candidate", "recompute", "recompute_aware"),
    ):
        cell = pair_cells[policy_label]
        summary_path = results_directory / (
            f"{cell['run_id']}.summary.json"
        )
        records[label] = load_validated_stage15_run(
            summary_path,
            expected_policy=policy,
            expected_capacity=capacity,
        )
    summary_paths = {
        "baseline": results_directory.resolve()
        / f"{pair_cells['pressure']['run_id']}.summary.json",
        "candidate": results_directory.resolve()
        / f"{pair_cells['recompute']['run_id']}.summary.json",
    }
    return {
        "schema_version": 1,
        "revision": revision,
        "capacity": capacity,
        "seed": seed,
        "bucket_seconds": bucket_seconds,
        "validation": {
            "manifest_path": str(manifest_path.resolve()),
            "full_matrix_runs": aggregate["matrix"]["runs"],
            "full_matrix_matched_pairs": aggregate["matrix"][
                "matched_pairs"
            ],
            "execution_code_fingerprint": aggregate[
                "execution_code_fingerprint"
            ],
        },
        "policies": {
            label: {
                "summary_path": str(summary_paths[label]),
                "steps_path": str(record["paths"]["steps"]),
                "elapsed_seconds": record["elapsed_seconds"],
                "recompute_metrics": record["recompute_metrics"],
                "mode_counts": mode_counts(record["steps"]),
                "timeline": bucket_steps(
                    record["steps"],
                    bucket_seconds=bucket_seconds,
                ),
            }
            for label, record in records.items()
        },
    }


def write_timeline_csv(path: Path, analysis: dict[str, Any]) -> None:
    rows = []
    for policy, data in analysis["policies"].items():
        for row in data["timeline"]:
            rows.append(
                {
                    "policy": policy,
                    **{key: value for key, value in row.items() if key != "modes"},
                    "modes": json.dumps(row["modes"], sort_keys=True),
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_timeline(path: Path, analysis: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    all_rows = [
        row
        for data in analysis["policies"].values()
        for row in data["timeline"]
    ]
    primary_max = max(
        max(
            row["kv_utilization_mean"],
            row["preemptions"] / 20,
            row["recompute_tokens"] / 10000,
        )
        for row in all_rows
    )
    waiting_age_max = max(row["oldest_waiting_age_max"] for row in all_rows)
    for axis, (policy, data) in zip(
        axes,
        analysis["policies"].items(),
        strict=True,
    ):
        timeline = data["timeline"]
        times = [row["bucket_start_s"] for row in timeline]
        axis.plot(
            times,
            [row["kv_utilization_mean"] for row in timeline],
            color="#2563eb",
            linewidth=2,
            label="Step-weighted KV utilization (pre-schedule)",
        )
        axis.bar(
            times,
            [row["preemptions"] / 20 for row in timeline],
            width=analysis["bucket_seconds"] * 0.75,
            color="#dc2626",
            alpha=0.55,
            label="Preemptions / 20",
        )
        axis.bar(
            times,
            [row["recompute_tokens"] / 10000 for row in timeline],
            width=analysis["bucket_seconds"] * 0.40,
            color="#f59e0b",
            alpha=0.65,
            label="Recompute tokens / 10k",
        )
        axis.set_ylabel(policy)
        axis.set_ylim(0, max(1.0, primary_max * 1.05))
        axis.grid(axis="y", alpha=0.25)
        waiting_axis = axis.twinx()
        waiting_axis.plot(
            times,
            [row["oldest_waiting_age_max"] for row in timeline],
            color="#7c3aed",
            linestyle="--",
            linewidth=1.8,
            label="Oldest waiting age (steps)",
        )
        waiting_axis.set_ylabel("Waiting age (steps)", color="#7c3aed")
        waiting_axis.tick_params(axis="y", labelcolor="#7c3aed")
        waiting_axis.set_ylim(0, max(1.0, waiting_age_max * 1.05))
        handles, labels = axis.get_legend_handles_labels()
        waiting_handles, waiting_labels = (
            waiting_axis.get_legend_handles_labels()
        )
        axis.legend(
            handles + waiting_handles,
            labels + waiting_labels,
            loc="upper right",
            ncol=2,
            fontsize=8,
        )
    axes[-1].set_xlabel("Benchmark time (s)")
    figure.suptitle(
        "KV pressure → Preemption → Recompute timeline "
        f"(KV={analysis['capacity']}, seed={analysis['seed']})"
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_analysis_bundle(
    prefix: Path,
    analysis: dict[str, Any],
) -> dict[str, Any]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    target_paths = {
        "json": prefix.with_suffix(".json"),
        "csv": prefix.with_suffix(".csv"),
        "png": prefix.with_suffix(".png"),
        "manifest": prefix.with_suffix(".bundle.json"),
    }
    with tempfile.TemporaryDirectory(
        dir=prefix.parent,
        prefix=f".{prefix.name}.",
    ) as temporary_directory:
        temporary_prefix = Path(temporary_directory) / prefix.name
        temporary_paths = {
            "json": temporary_prefix.with_suffix(".json"),
            "csv": temporary_prefix.with_suffix(".csv"),
            "png": temporary_prefix.with_suffix(".png"),
            "manifest": temporary_prefix.with_suffix(".bundle.json"),
        }
        write_json_atomic(temporary_paths["json"], analysis)
        write_timeline_csv(temporary_paths["csv"], analysis)
        plot_timeline(temporary_paths["png"], analysis)
        bundle = {
            "schema_version": 1,
            "revision": analysis["revision"],
            "capacity": analysis["capacity"],
            "seed": analysis["seed"],
            "artifacts": {
                label: {
                    "path": str(target_paths[label]),
                    "sha256": sha256_file(temporary_paths[label]),
                }
                for label in ("json", "csv", "png")
            },
        }
        write_json_atomic(temporary_paths["manifest"], bundle)
        for label in ("json", "csv", "png", "manifest"):
            temporary_paths[label].replace(target_paths[label])
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default="stage15_diagnostic_r3")
    parser.add_argument("--capacity", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--bucket-seconds", type=float, default=1.0)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT
        / "experiments/configs/stage15_diagnostic_r3_matrix.json",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=REPO_ROOT / "experiments/results/stage15_diagnostic_r3_timeline",
    )
    args = parser.parse_args()
    analysis = analyze_pair(
        capacity=args.capacity,
        seed=args.seed,
        revision=args.revision,
        results_directory=REPO_ROOT / "experiments/results",
        manifest_path=args.manifest,
        bucket_seconds=args.bucket_seconds,
    )
    prefix = args.output_prefix.resolve()
    write_analysis_bundle(prefix, analysis)
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
