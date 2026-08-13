from __future__ import annotations

import unittest

from experiments.benchmark.lifecycle import (
    RequestRecord,
    TerminalState,
    summarize_requests,
)


class LifecycleMetricsTests(unittest.TestCase):
    def test_finished_request_metrics_and_terminal_reconciliation(self) -> None:
        finished = RequestRecord(
            request_id="interactive-000",
            request_class="interactive",
            arrival_at=0.0,
        )
        finished.mark_admitted(0.01)
        finished.record_schedule(0.02)
        finished.record_token(0.05)
        finished.record_schedule(0.07)
        finished.record_token(0.07)
        finished.record_schedule(0.09)
        finished.record_token(0.09)
        finished.mark_terminal(TerminalState.FINISHED, 0.09)

        unfinished = RequestRecord(
            request_id="interactive-001",
            request_class="interactive",
            arrival_at=0.2,
        )
        unfinished.mark_admitted(0.21)
        unfinished.mark_terminal(TerminalState.UNFINISHED, 1.0)

        summary = summarize_requests(
            [finished, unfinished],
            measurement_start=0.0,
            measurement_end=1.0,
            ttft_slo_ms=60.0,
            itl_slo_ms=30.0,
        )

        self.assertEqual(summary["terminal_counts"]["submitted"], 2)
        self.assertEqual(summary["terminal_counts"]["Finished"], 1)
        self.assertEqual(summary["terminal_counts"]["Unfinished"], 1)
        self.assertTrue(summary["terminal_counts"]["reconciled"])
        self.assertAlmostEqual(summary["interactive"]["ttft_ms"]["p50"], 50.0)
        self.assertAlmostEqual(summary["interactive"]["itl_ms"]["p99"], 20.0)
        self.assertEqual(summary["interactive"]["slo_successes"], 1)
        self.assertAlmostEqual(summary["interactive"]["slo_goodput_rps"], 1.0)

    def test_terminal_state_is_exclusive(self) -> None:
        record = RequestRecord(
            request_id="request-000",
            request_class="long",
            arrival_at=0.0,
        )
        record.mark_terminal(TerminalState.FAILED, 0.1, reason="synthetic")

        with self.assertRaises(RuntimeError):
            record.mark_terminal(TerminalState.CANCELLED, 0.2)

    def test_single_token_request_does_not_pass_required_itl_slo(self) -> None:
        record = RequestRecord(
            request_id="interactive-000",
            request_class="interactive",
            arrival_at=0.0,
        )
        record.mark_admitted(0.0)
        record.record_schedule(0.0)
        record.record_token(0.01)
        record.mark_terminal(TerminalState.FINISHED, 0.01)

        summary = summarize_requests(
            [record],
            measurement_start=0.0,
            measurement_end=1.0,
            ttft_slo_ms=20.0,
            itl_slo_ms=20.0,
            require_itl=True,
        )

        self.assertEqual(summary["interactive"]["slo_successes"], 0)

    def test_measurement_window_excludes_outside_arrivals(self) -> None:
        record = RequestRecord(
            request_id="interactive-late",
            request_class="interactive",
            arrival_at=2.0,
        )
        record.mark_admitted(2.0)
        record.record_schedule(2.0)
        record.record_token(2.1)
        record.mark_terminal(TerminalState.FINISHED, 2.1)

        summary = summarize_requests(
            [record],
            measurement_start=0.0,
            measurement_end=1.0,
            ttft_slo_ms=200.0,
            itl_slo_ms=200.0,
            require_itl=False,
        )

        self.assertEqual(summary["measurement"]["eligible_requests"], 0)
        self.assertEqual(summary["interactive"]["submitted"], 0)


if __name__ == "__main__":
    unittest.main()
