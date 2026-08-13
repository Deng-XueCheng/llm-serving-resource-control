from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--expected-cells", type=int, required=True)
    parser.add_argument("--knee-rps", type=float)
    args = parser.parse_args()
    raw = sorted(
        path for path in args.directory.glob("*.json")
        if not path.name.endswith(".config.json") and path.name not in {
            "summary.json", "evidence_review.json"
        }
    )
    if len(raw) != args.expected_cells:
        raise RuntimeError(f"Expected {args.expected_cells} cells, found {len(raw)}")
    pairs = defaultdict(dict)
    cells = []
    for path in raw:
        item = json.loads(path.read_text())
        meta = item["characterization"]
        provenance = item["gpu_provenance"]
        preflight = item["gpu_preflight"]
        terminal = item["summary"]["terminal_counts"]
        checks = {
            "post_fix": item["evidence_generation"] == "POST_GPU_BINDING_FIX",
            "preflight": preflight["status"] == "PREFLIGHT_PROVENANCE_PASS",
            "runtime": provenance["cell_status"] == "VALID_GPU_PROVENANCE",
            "different_gpus": len(set(preflight["physical_gpu_uuid_list"])) == 2,
            "overlap": provenance["dual_gpu_overlap_observed"],
            "both_cuda": {x["replica_id"] for x in provenance["step_execution_intervals"]
                          if x["cuda_elapsed_ms"] > 0} == {0, 1},
            "terminal": terminal["reconciled"] and terminal["Finished"] == terminal["submitted"],
            "kv_released": provenance["final_free_kv_blocks"] == [8, 8],
        }
        if not all(checks.values()):
            raise RuntimeError(f"Review failure {path}: {checks}")
        condition = meta.get("load_multiplier", meta.get("workload_shape"))
        pairs[(condition, meta["seed"])][meta["router"]] = meta
        cells.append({"artifact": str(path), "checks": checks, **meta})
    for key, pair in pairs.items():
        if set(pair) != {"round_robin", "resource_aware"}:
            raise RuntimeError(f"Incomplete pair {key}")
        if pair["round_robin"]["trace_sha256"] != pair["resource_aware"]["trace_sha256"]:
            raise RuntimeError(f"Trace mismatch {key}")
    output = {
        "schema_version": 1, "review_status": "PASS",
        "expected_cells": args.expected_cells, "matched_pairs": len(pairs),
        "saturation_knee_rps": args.knee_rps, "cells": cells,
    }
    (args.directory / "evidence_review.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
