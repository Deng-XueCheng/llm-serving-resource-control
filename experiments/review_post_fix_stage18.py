from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    artifacts = sorted(args.directory.glob("load*.json"))
    if len(artifacts) != 12:
        raise RuntimeError(f"Expected 12 raw cells, found {len(artifacts)}")
    cells = []
    pairs = defaultdict(dict)
    for path in artifacts:
        item = json.loads(path.read_text())
        formal = item["formal"]
        terminal = item["summary"]["terminal_counts"]
        preflight = item["gpu_preflight"]
        runtime = item["gpu_provenance"]
        assertions = {
            "post_fix": item["evidence_generation"] == "POST_GPU_BINDING_FIX",
            "preflight": preflight["status"] == "PREFLIGHT_PROVENANCE_PASS",
            "runtime": runtime["cell_status"] == "VALID_GPU_PROVENANCE",
            "distinct_gpu_uuids": len(set(preflight["physical_gpu_uuid_list"])) == 2,
            "overlap": runtime["dual_gpu_overlap_observed"] is True,
            "both_cuda": {x["replica_id"] for x in runtime["step_execution_intervals"]
                          if x["cuda_elapsed_ms"] > 0} == {0, 1},
            "terminal": terminal["reconciled"] and terminal["Finished"] == terminal["submitted"],
            "kv_released": runtime["final_free_kv_blocks"] == [8, 8],
            "both_routed": set(runtime["requests_per_replica"]) == {"0", "1"},
        }
        if not all(assertions.values()):
            raise RuntimeError(f"Evidence review failed for {path}: {assertions}")
        summary = item["summary"]
        row = {
            **formal,
            "artifact": str(path),
            "code_sha": item["code_sha"],
            "config_hash": item["benchmark_config_hash"],
            "trace_hash": item["trace_hash"],
            "ttft_p99_ms": summary["interactive"]["ttft_ms"]["p99"],
            "itl_p99_ms": summary["interactive"]["itl_ms"]["p99"],
            "slo_goodput_rps": summary["interactive"]["slo_goodput_rps"],
            "throughput_tps": summary["output_throughput_tps"],
            "queue_imbalance_mean": item["routing_summary"]["queue_imbalance"]["mean"],
            "kv_imbalance_mean": item["routing_summary"]["kv_imbalance"]["mean"],
            "gpu_uuids": preflight["physical_gpu_uuid_list"],
            "assertions": assertions,
        }
        cells.append(row)
        pairs[(formal["offered_rps"], formal["seed"])][formal["router"]] = row
    pair_rows = []
    for key, pair in sorted(pairs.items()):
        if set(pair) != {"round_robin", "resource_aware"}:
            raise RuntimeError(f"Incomplete matched pair: {key}")
        if pair["round_robin"]["trace_hash"] != pair["resource_aware"]["trace_hash"]:
            raise RuntimeError(f"Trace mismatch: {key}")
        pair_rows.append({
            "offered_rps": key[0], "seed": key[1],
            **{
                f"delta_{metric}": pair["resource_aware"][metric] - pair["round_robin"][metric]
                for metric in (
                    "ttft_p99_ms", "itl_p99_ms", "slo_goodput_rps",
                    "throughput_tps", "queue_imbalance_mean", "kv_imbalance_mean",
                )
            },
        })
    grouped = []
    for offered_rps in sorted({row["offered_rps"] for row in pair_rows}):
        selected = [row for row in pair_rows if row["offered_rps"] == offered_rps]
        grouped.append({
            "offered_rps": offered_rps,
            **{key: mean(row[key] for row in selected)
               for key in selected[0] if key.startswith("delta_")},
        })
    output = {
        "schema_version": 1,
        "review_status": "PASS",
        "review_source": "independent raw artifact recomputation",
        "cells": cells,
        "matched_pairs": pair_rows,
        "grouped_mean_deltas_candidate_minus_baseline": grouped,
    }
    (args.directory / "evidence_review.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
