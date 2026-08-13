from __future__ import annotations

import unittest

from experiments.final_characterization import (
    MEASUREMENT_SECONDS,
    REFERENCE_RPS,
    make_trace,
    grouped_means,
    paired_summary,
    recompute_metrics,
)


class FinalCharacterizationTests(unittest.TestCase):
    def test_saturation_trace_has_requested_monotonic_offered_load(self):
        low = make_trace(offered_rps=REFERENCE_RPS * 0.5, seed=1)
        high = make_trace(offered_rps=REFERENCE_RPS * 1.5, seed=1)
        self.assertEqual(len(low["requests"]), round(REFERENCE_RPS * 0.5 * MEASUREMENT_SECONDS))
        self.assertEqual(len(high["requests"]), round(REFERENCE_RPS * 1.5 * MEASUREMENT_SECONDS))
        self.assertGreater(len(high["requests"]), len(low["requests"]))
        self.assertTrue(all(
            left["arrival_time"] < right["arrival_time"]
            for left, right in zip(high["requests"], high["requests"][1:])
        ))

    def test_shape_trace_preserves_prefix_only_for_prefix_heavy(self):
        prefix = make_trace(offered_rps=1.0, seed=2, shape="prefix_heavy")
        decode = make_trace(offered_rps=1.0, seed=2, shape="decode_heavy")
        self.assertTrue(all(item["shared_prefix_length"] > 0 for item in prefix["requests"]))
        self.assertTrue(all(item["shared_prefix_length"] == 0 for item in decode["requests"]))
        self.assertTrue(all(item["max_output_tokens"] >= 128 for item in decode["requests"]))

    def test_raw_metric_recompute_distinguishes_offered_and_achieved(self):
        trace = {"requests": [
            {"arrival_time": 0.0, "max_output_tokens": 10},
            {"arrival_time": 1.0, "max_output_tokens": 20},
        ]}
        artifact = {
            "summary": {
                "measurement": {"duration": 2.0},
                "terminal_counts": {"submitted": 2, "Finished": 2, "Rejected": 0, "Failed": 0, "Unfinished": 0},
                "interactive": {"slo_goodput_rps": 0.5, "ttft_ms": {"p50": 1, "p99": 2}, "itl_ms": {"p50": 3, "p99": 4}},
                "output_throughput_tps": 7.0,
            },
            "requests": [{"arrival_at": 0.0, "token_timestamps": [1]}, {"arrival_at": 1.0, "token_timestamps": [1, 2]}],
            "resource_events": [{"snapshots": [
                {"waiting_queue_length": 2, "running_requests": 1, "kv_utilization": 0.5, "oldest_waiting_age": 3},
                {"waiting_queue_length": 0, "running_requests": 1, "kv_utilization": 0.25, "oldest_waiting_age": 0},
            ]}],
            "routing_summary": {"queue_imbalance": {"mean": 2}, "kv_imbalance": {"mean": 0.25}, "matched_prefix_blocks_at_route": 0, "routed_requests_by_replica": {"0": 1, "1": 1}},
        }
        metrics = recompute_metrics(artifact, 1.0, trace)
        self.assertEqual(metrics["offered_output_tps"], 15.0)
        self.assertEqual(metrics["achieved_output_tps"], 7.0)
        self.assertEqual(metrics["waiting_queue_length"]["max"], 2)
        self.assertEqual(metrics["system_queue_length"]["max"], 3)

    def test_pairs_compare_only_same_condition_and_seed(self):
        cells = []
        for router, value in (("round_robin", 2.0), ("resource_aware", 1.0)):
            cells.append({"phase": "P0-1", "load_multiplier": 0.5, "seed": 1, "router": router, "metrics": {
                "achieved_output_tps": value, "slo_goodput_rps": value,
                "ttft_p99_ms": value, "itl_p99_ms": value, "completion_rate": 1.0,
            }})
        pairs = paired_summary(cells)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["itl_p99_ms_delta"], -1.0)

    def test_grouped_means_preserve_condition_and_router(self):
        metrics = {
            "offered_rps": 1.0, "offered_output_tps": 2.0,
            "achieved_output_tps": 2.0, "slo_goodput_rps": 1.0,
            "ttft_p50_ms": 1.0, "ttft_p99_ms": 2.0,
            "itl_p50_ms": 3.0, "itl_p99_ms": 4.0,
            "completion_rate": 1.0, "prefix_matched_blocks": 0,
            **{name: {"mean": 1.0, "p99": 2.0, "max": 3.0} for name in (
                "waiting_queue_length", "system_queue_length", "queue_imbalance",
                "kv_utilization", "kv_imbalance", "oldest_waiting_age",
            )},
        }
        rows = grouped_means([
            {"phase": "P0-1", "load_multiplier": 0.5, "router": "round_robin", "metrics": metrics},
            {"phase": "P0-1", "load_multiplier": 0.5, "router": "round_robin", "metrics": metrics},
        ])
        self.assertEqual(rows[0]["seeds"], 2)
        self.assertEqual(rows[0]["waiting_queue_length_max"], 3.0)


if __name__ == "__main__":
    unittest.main()
