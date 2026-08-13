from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from statistics import mean

from experiments.run_multi_replica import run
from experiments.snapshot_preflight import model_contract


REPO_ROOT = Path(__file__).resolve().parents[1]
ROUTERS = ("round_robin", "resource_aware")
SEEDS = (1, 2, 3)
LOADS = (0.50, 0.75, 1.00, 1.25, 1.50)
REFERENCE_RPS = 4.8
MEASUREMENT_SECONDS = 20.0

SHAPES = {
    "prefill_heavy": ((640, 16), (768, 16), (896, 24), (512, 16)),
    "balanced": ((256, 48), (384, 64), (512, 48), (320, 64)),
    "decode_heavy": ((32, 128), (64, 160), (96, 128), (48, 192)),
    "prefix_heavy": ((512, 32), (640, 48), (768, 32), (576, 48)),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def pattern() -> list[dict]:
    source = json.loads(
        (REPO_ROOT / "experiments/data/stage18_imbalance_seed1.json").read_text()
    )
    return source["requests"]


def make_trace(*, offered_rps: float, seed: int, shape: str | None = None) -> dict:
    count = int(round(offered_rps * MEASUREMENT_SECONDS))
    source = pattern()
    requests = []
    shape_values = SHAPES.get(shape) if shape else None
    for index in range(count):
        template = source[index % len(source)]
        request = dict(template)
        request["request_id"] = f"{shape or 'saturation'}-s{seed}-{index:03d}"
        request["arrival_time"] = index / offered_rps
        request["seed"] = 200000 + seed * 10000 + index
        if shape_values:
            prompt, output = shape_values[index % len(shape_values)]
            request["prompt_length"] = prompt
            request["max_output_tokens"] = output
            request["request_class"] = "interactive"
            if shape == "prefix_heavy":
                request["prefix_group"] = f"prefix-{index % 2}"
                request["shared_prefix_length"] = min(384, prompt - 1)
            else:
                request["prefix_group"] = None
                request["shared_prefix_length"] = 0
        requests.append(request)
    return {
        "schema_version": 2,
        "description": "Final system characterization; generated from Stage 18 request pattern.",
        "time_unit": "seconds",
        "requests": requests,
    }


def base_config(trace_path: Path, port: int, *, model_path: Path) -> dict:
    return {
        **model_contract(model_path),
        "trace_path": str(trace_path.relative_to(REPO_ROOT)),
        "token_id_upper_bound": 10000,
        "replicas": {"gpu_ids": [0, 1], "distributed_init_port_base": port},
        "engine": {
            "enforce_eager": True, "tensor_parallel_size": 1,
            "max_model_len": 1024, "max_num_batched_tokens": 256,
            "max_num_seqs": 8, "gpu_memory_utilization": 0.35,
            "kvcache_block_size": 256, "num_kvcache_blocks": 8,
            "scheduler_policy": "mixed_token_budget", "decode_token_budget": 8,
            "incremental_kv_allocation": False,
        },
        "sampling": {"temperature": 0.6, "ignore_eos": True, "seed": 2000},
        "measurement_start": 0.0, "measurement_end": MEASUREMENT_SECONDS,
        "max_run_seconds": 90.0, "ttft_slo_ms": 5000.0,
        "itl_slo_ms": 1000.0, "require_itl": True,
    }


def distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {"mean": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "mean": mean(ordered),
        "p99": ordered[round((len(ordered) - 1) * 0.99)],
        "max": ordered[-1],
    }


def recompute_metrics(artifact: dict, offered_rps: float, trace: dict) -> dict:
    summary = artifact["summary"]
    resources = artifact["resource_events"]
    snapshots = [snapshot for event in resources for snapshot in event["snapshots"]]
    waiting_queues = [s["waiting_queue_length"] for s in snapshots]
    system_queues = [s["waiting_queue_length"] + s["running_requests"] for s in snapshots]
    kv = [s["kv_utilization"] for s in snapshots]
    ages = [s["oldest_waiting_age"] for s in snapshots]
    terminal = summary["terminal_counts"]
    interactive = summary["interactive"]
    duration = summary["measurement"]["duration"]
    finished_tokens = sum(
        len(request["token_timestamps"])
        for request in artifact["requests"]
        if request["arrival_at"] < duration
    )
    return {
        "offered_rps": offered_rps,
        "offered_output_tps": sum(
            request["max_output_tokens"]
            for request in trace["requests"]
            if request["arrival_time"] < duration
        ) / duration,
        "achieved_output_tps": summary["output_throughput_tps"],
        "slo_goodput_rps": interactive["slo_goodput_rps"],
        "ttft_p50_ms": interactive["ttft_ms"]["p50"],
        "ttft_p99_ms": interactive["ttft_ms"]["p99"],
        "itl_p50_ms": interactive["itl_ms"]["p50"],
        "itl_p99_ms": interactive["itl_ms"]["p99"],
        "submitted": terminal["submitted"], "finished": terminal["Finished"],
        "rejected": terminal["Rejected"], "failed": terminal["Failed"],
        "unfinished": terminal["Unfinished"],
        "completion_rate": terminal["Finished"] / terminal["submitted"],
        "waiting_queue_length": distribution(waiting_queues),
        "system_queue_length": distribution(system_queues),
        "queue_imbalance": artifact["routing_summary"]["queue_imbalance"],
        "kv_utilization": distribution(kv),
        "kv_imbalance": artifact["routing_summary"]["kv_imbalance"],
        "oldest_waiting_age": distribution(ages),
        "prefix_matched_blocks": artifact["routing_summary"]["matched_prefix_blocks_at_route"],
        "requests_by_replica": artifact["routing_summary"]["routed_requests_by_replica"],
        "raw_token_count_check": finished_tokens,
    }


def run_cell(config: dict, *, output: Path, metadata: dict) -> dict:
    config_path = output.with_suffix(".config.json")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    trace_path = REPO_ROOT / config["trace_path"]
    trace = json.loads(trace_path.read_text())
    result = run(config)
    offered_rps = metadata["offered_rps"]
    result["characterization"] = {
        **metadata, "code_commit": git_head(),
        "config_sha256": sha256(config_path),
        "trace_sha256": sha256(trace_path),
        "runner_sha256": sha256(REPO_ROOT / "experiments/run_multi_replica.py"),
        "characterization_sha256": sha256(Path(__file__)),
        "metrics": recompute_metrics(result, offered_rps, trace),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    if result["status"] != "ok":
        raise RuntimeError(f"Cell failed: {output.name}: {result['error']}")
    terminal = result["summary"]["terminal_counts"]
    if not terminal["reconciled"] or terminal["Failed"] or terminal["Unfinished"]:
        raise RuntimeError(f"Incomplete terminal states: {output.name}: {terminal}")
    if any(item["used_kv_blocks"] != 0 for item in result["final_replica_snapshots"]):
        raise RuntimeError(f"Final KV leak: {output.name}")
    return result


def write_trace(path: Path, trace: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace, indent=2) + "\n")


def run_p0_1(
    output: Path,
    *,
    model_path: Path,
    reference_rps: float,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    cells = []
    for load_index, multiplier in enumerate(LOADS):
        offered_rps = reference_rps * multiplier
        for seed in SEEDS:
            trace_path = REPO_ROOT / "experiments/data/final_characterization" / f"saturation_load{multiplier:g}_seed{seed}.json"
            write_trace(trace_path, make_trace(offered_rps=offered_rps, seed=seed))
            for router_index, router in enumerate(ROUTERS):
                cell_id = f"load{multiplier:g}_seed{seed}_{router}"
                config = base_config(
                    trace_path,
                    25000 + load_index * 100 + seed * 10 + router_index * 2,
                    model_path=model_path,
                )
                config.update({"router": router, "prefix_aware": False})
                result = run_cell(config, output=output / f"{cell_id}.json", metadata={
                    "phase": "P0-1", "cell_id": cell_id, "router": router,
                    "seed": seed, "load_multiplier": multiplier,
                    "offered_rps": offered_rps,
                })
                cells.append(result["characterization"])
    write_outputs(output, cells, phase="P0-1", reference_rps=reference_rps)


def run_p0_2(
    output: Path,
    *,
    model_path: Path,
    offered_rps: float,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    cells = []
    for shape_index, shape in enumerate(SHAPES):
        for seed in SEEDS:
            trace_path = REPO_ROOT / "experiments/data/final_characterization" / f"{shape}_seed{seed}.json"
            write_trace(trace_path, make_trace(offered_rps=offered_rps, seed=seed, shape=shape))
            for router_index, router in enumerate(ROUTERS):
                cell_id = f"{shape}_seed{seed}_{router}"
                config = base_config(
                    trace_path,
                    26000 + shape_index * 100 + seed * 10 + router_index * 2,
                    model_path=model_path,
                )
                config.update({"router": router, "prefix_aware": shape == "prefix_heavy" and router == "resource_aware"})
                result = run_cell(config, output=output / f"{cell_id}.json", metadata={
                    "phase": "P0-2", "cell_id": cell_id, "router": router,
                    "seed": seed, "workload_shape": shape,
                    "prefix_affinity": config["prefix_aware"],
                    "offered_rps": offered_rps,
                })
                cells.append(result["characterization"])
    write_outputs(output, cells, phase="P0-2")


def flatten(cell: dict) -> dict:
    metrics = cell["metrics"]
    row = {k: v for k, v in cell.items() if not isinstance(v, dict)}
    for key, value in metrics.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                if not isinstance(subvalue, dict):
                    row[f"{key}_{subkey}"] = subvalue
        else:
            row[key] = value
    return row


def write_outputs(
    output: Path, cells: list[dict], *, phase: str,
    reference_rps: float | None = None,
) -> None:
    rows = [flatten(cell) for cell in cells]
    fields = sorted({key for row in rows for key in row})
    with (output / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    pairs = paired_summary(cells)
    summary = {
        "schema_version": 1,
        "phase": phase,
        "cells": cells,
        "matched_pairs": pairs,
        "grouped_means": grouped_means(cells),
        "reference_rps": reference_rps,
        "saturation_knee": None,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_svg(output / "characterization.svg", cells, phase)


def grouped_means(cells: list[dict]) -> list[dict]:
    condition_key = (
        "load_multiplier" if cells[0]["phase"] == "P0-1" else "workload_shape"
    )
    grouped = {}
    for cell in cells:
        grouped.setdefault((cell[condition_key], cell["router"]), []).append(
            cell["metrics"]
        )
    output = []
    scalar_metrics = (
        "offered_rps", "offered_output_tps", "achieved_output_tps",
        "slo_goodput_rps", "ttft_p50_ms", "ttft_p99_ms",
        "itl_p50_ms", "itl_p99_ms", "completion_rate",
        "prefix_matched_blocks",
    )
    nested_metrics = (
        "waiting_queue_length", "system_queue_length", "queue_imbalance",
        "kv_utilization", "kv_imbalance", "oldest_waiting_age",
    )
    for (condition, router), metrics in sorted(
        grouped.items(), key=lambda item: (str(item[0][0]), item[0][1])
    ):
        row = {condition_key: condition, "router": router, "seeds": len(metrics)}
        for name in scalar_metrics:
            values = [item[name] for item in metrics if item[name] is not None]
            row[name] = mean(values) if values else None
        for name in nested_metrics:
            for statistic in ("mean", "p99", "max"):
                row[f"{name}_{statistic}"] = mean(
                    item[name][statistic] for item in metrics
                )
        output.append(row)
    return output


def paired_summary(cells: list[dict]) -> list[dict]:
    key_name = "load_multiplier" if cells[0]["phase"] == "P0-1" else "workload_shape"
    grouped = {}
    for cell in cells:
        grouped.setdefault((cell[key_name], cell["seed"]), {})[cell["router"]] = cell
    output = []
    for (condition, seed), pair in sorted(grouped.items(), key=lambda item: str(item[0])):
        rr = pair["round_robin"]["metrics"]; ra = pair["resource_aware"]["metrics"]
        output.append({key_name: condition, "seed": seed, **{
            f"{metric}_delta": ra[metric] - rr[metric]
            for metric in ("achieved_output_tps", "slo_goodput_rps", "ttft_p99_ms", "itl_p99_ms", "completion_rate")
        }})
    return output


def write_svg(path: Path, cells: list[dict], phase: str) -> None:
    metrics = ("achieved_output_tps", "slo_goodput_rps", "ttft_p99_ms", "itl_p99_ms")
    xkey = "load_multiplier" if phase == "P0-1" else "workload_shape"
    xs = sorted({cell[xkey] for cell in cells}, key=str)
    width, height = 1000, 760; parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>']
    colors = {"round_robin": "#0072B2", "resource_aware": "#D55E00"}
    for chart, metric in enumerate(metrics):
        left = 70 + (chart % 2) * 490; top = 45 + (chart // 2) * 360
        values = [cell["metrics"][metric] or 0 for cell in cells]; maximum = max(values) or 1
        parts.append(f'<text x="{left}" y="{top}" font-size="16">{metric}</text>')
        for router in ROUTERS:
            points = []
            for index, x in enumerate(xs):
                vals = [c["metrics"][metric] or 0 for c in cells if c[xkey] == x and c["router"] == router]
                px = left + index * (400 / max(1, len(xs) - 1)); py = top + 285 - mean(vals) / maximum * 250
                points.append(f"{px:.1f},{py:.1f}")
                parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{colors[router]}"/>')
            parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[router]}" stroke-width="2"/>')
        for index, x in enumerate(xs):
            px = left + index * (400 / max(1, len(xs) - 1)); parts.append(f'<text x="{px:.1f}" y="{top+310}" text-anchor="middle" font-size="11">{x}</text>')
    parts.append('</svg>'); path.write_text("\n".join(parts) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("p0-1", "p0-2"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--offered-rps", type=float)
    parser.add_argument("--reference-rps", type=float)
    args = parser.parse_args()
    if args.phase == "p0-1":
        if args.reference_rps is None:
            parser.error("p0-1 requires --reference-rps from post-fix calibration")
        run_p0_1(
            args.output,
            model_path=args.model_path,
            reference_rps=args.reference_rps,
        )
    elif args.offered_rps is None: parser.error("p0-2 requires --offered-rps")
    else:
        run_p0_2(
            args.output,
            model_path=args.model_path,
            offered_rps=args.offered_rps,
        )


if __name__ == "__main__":
    main()
