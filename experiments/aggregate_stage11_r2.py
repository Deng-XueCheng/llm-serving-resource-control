from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from experiments import aggregate_stage11 as r1
from experiments.generate_matrix import write_json_idempotent


OUTPUT = r1.RESULTS / "stage11_formal_r1_aggregate_r2.json"


def nullable_mean_std(values: list[float | None]) -> dict[str, float | int | None]:
    present = [float(value) for value in values if value is not None]
    if any(value is not None and not r1.finite_number(value) for value in values):
        raise ValueError("metric contains a non-finite value")
    if not present:
        return {
            "mean": None,
            "sample_std": None,
            "valid_runs": 0,
            "missing_runs": len(values),
        }
    return {
        "mean": statistics.mean(present),
        "sample_std": (
            statistics.stdev(present) if len(present) > 1 else 0.0
        ),
        "valid_runs": len(present),
        "missing_runs": len(values) - len(present),
    }


def attach_metric_coverage(value: dict) -> None:
    for record in value["runs"]:
        cell = (
            f"reuse{record['reuse_ratio']}_kv{record['capacity']}_"
            f"{record['policy']}"
        )
        for metric, raw_value in record["metrics"].items():
            coverage = value["by_cell"][cell][metric]
            key = "valid_seeds" if raw_value is not None else "missing_seeds"
            coverage.setdefault(key, []).append(record["seed"])
    for cell in value["by_cell"].values():
        for metric in cell.values():
            if not isinstance(metric, dict) or "valid_runs" not in metric:
                continue
            metric["valid_seeds"] = sorted(metric.get("valid_seeds", []))
            metric["missing_seeds"] = sorted(metric.get("missing_seeds", []))


def aggregate() -> dict:
    # r1 remains the frozen evidence validator. r2 changes only post-validation
    # descriptive statistics for valid cells whose P99 is undefined because a
    # repeat had no completed requests of that class.
    original = r1.nullable_mean_std
    try:
        r1.nullable_mean_std = nullable_mean_std
        value = r1.aggregate()
    finally:
        r1.nullable_mean_std = original
    attach_metric_coverage(value)
    value["aggregation_revision"] = "stage11_aggregate_r2"
    value["aggregation_delta"] = (
        "Retains r1 validation unchanged; nullable per-run latency metrics "
        "now report valid_runs and missing_runs instead of rejecting a formal "
        "cell solely because no eligible request completed in one repeat."
    )
    value["metric_semantics"] = (
        "Latency P99 with missing_runs > 0 is a conditional descriptive "
        "statistic over valid_seeds only. It must not be used for matched "
        "cross-policy latency deltas, rankings, or significance claims."
    )
    value["base_aggregator_sha256"] = r1.sha256(Path(r1.__file__))
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(r1.RESULTS.resolve()):
        raise ValueError("aggregate output must be inside experiments/results")
    value = aggregate()
    value["aggregate_sha256"] = r1.sha256(Path(__file__))
    write_json_idempotent(args.output, value)


if __name__ == "__main__":
    main()
