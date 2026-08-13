from __future__ import annotations

import unittest
from types import SimpleNamespace

from nanovllm import SamplingParams
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence


class FakeModelRunner:
    def __init__(self, token_ids: list[int]) -> None:
        self.token_ids = iter(token_ids)

    def call(self, method_name: str, seqs, is_prefill: bool) -> list[int]:
        if method_name != "run":
            raise AssertionError(f"Unexpected method: {method_name}")
        return [next(self.token_ids) for _ in seqs]


class FakeMixedModelRunner:
    def __init__(self) -> None:
        self.sampled_phases = []

    def call(self, method_name: str, batch):
        if method_name != "run_mixed":
            raise AssertionError(f"Unexpected method: {method_name}")
        self.sampled_phases.append(
            [item.phase for item in batch.items if item.requires_sampling]
        )
        return {
            item.sequence.seq_id: 1000 + item.sequence.seq_id
            for item in batch.items
            if item.requires_sampling
        }


class FakeTokenizer:
    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(token_id) for token_id in token_ids)


def scheduler_config() -> SimpleNamespace:
    return SimpleNamespace(
        max_num_batched_tokens=4,
        max_num_seqs=4,
        eos=-1,
        kvcache_block_size=4,
        num_kvcache_blocks=32,
    )


class EngineObservabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        Sequence.block_size = 4
        self.engine = LLMEngine.__new__(LLMEngine)
        self.engine.scheduler = Scheduler(scheduler_config())
        self.engine.model_runner = FakeModelRunner([101, 102])
        self.engine.tokenizer = FakeTokenizer()

    def test_add_request_returns_internal_sequence_id(self) -> None:
        seq_id = self.engine.add_request(
            [1, 2],
            SamplingParams(temperature=1.0, max_tokens=2, ignore_eos=True),
        )

        self.assertIsInstance(seq_id, int)
        self.assertEqual(self.engine.scheduler.waiting[0].seq_id, seq_id)

    def test_step_with_events_reports_prefill_and_decode_tokens(self) -> None:
        seq_id = self.engine.add_request(
            [1, 2],
            SamplingParams(temperature=1.0, max_tokens=2, ignore_eos=True),
        )

        prefill = self.engine.step_with_events()
        self.assertEqual(prefill.phase, "prefill")
        self.assertEqual(prefill.num_scheduled_tokens, 2)
        self.assertEqual(len(prefill.events), 1)
        self.assertEqual(prefill.events[0].seq_id, seq_id)
        self.assertEqual(prefill.events[0].emitted_token_id, 101)
        self.assertFalse(prefill.events[0].finished)
        self.assertEqual(prefill.events[0].prefill_kind, "initial_prefill")
        self.assertEqual(prefill.events[0].actual_recompute_tokens, 0)
        self.assertFalse(prefill.events[0].resumed)

        decode = self.engine.step_with_events()
        self.assertEqual(decode.phase, "decode")
        self.assertEqual(decode.num_scheduled_tokens, 1)
        self.assertEqual(decode.events[0].emitted_token_id, 102)
        self.assertTrue(decode.events[0].finished)
        self.assertEqual(decode.outputs, [(seq_id, [101, 102])])

    def test_preempted_sequence_reports_executed_recompute_and_resume(
        self,
    ) -> None:
        seq_id = self.engine.add_request(
            [1, 2],
            SamplingParams(temperature=1.0, max_tokens=3, ignore_eos=True),
        )
        self.engine.step_with_events()
        seq = self.engine.scheduler.running[0]
        self.engine.scheduler.running.remove(seq)
        self.engine.scheduler.preempt(
            seq,
            reason="unit_test",
            prepend=True,
        )

        recovery = self.engine.step_with_events()

        self.assertEqual(recovery.phase, "prefill")
        self.assertEqual(recovery.events[0].seq_id, seq_id)
        self.assertEqual(
            recovery.events[0].prefill_kind,
            "recompute_prefill",
        )
        self.assertEqual(
            recovery.events[0].actual_recompute_tokens,
            recovery.events[0].num_scheduled_tokens,
        )
        self.assertTrue(recovery.events[0].resumed)
        snapshot = self.engine.observability_snapshot()
        self.assertEqual(
            snapshot["actual_recompute_tokens"],
            recovery.events[0].actual_recompute_tokens,
        )
        self.assertEqual(snapshot["resume_count"], 1)

    def test_legacy_step_return_shape_is_preserved(self) -> None:
        seq_id = self.engine.add_request(
            [1, 2],
            SamplingParams(temperature=1.0, max_tokens=2, ignore_eos=True),
        )

        outputs, signed_token_count = self.engine.step()

        self.assertEqual(outputs, [])
        self.assertEqual(signed_token_count, 2)

        outputs, signed_token_count = self.engine.step()
        self.assertEqual(outputs, [(seq_id, [101, 102])])
        self.assertEqual(signed_token_count, -1)

    def test_chunked_prefill_event_does_not_report_discarded_sample(self) -> None:
        seq_id = self.engine.add_request(
            list(range(10)),
            SamplingParams(temperature=1.0, max_tokens=2, ignore_eos=True),
        )

        result = self.engine.step_with_events()

        self.assertEqual(result.phase, "prefill")
        self.assertEqual(result.num_scheduled_tokens, 4)
        self.assertEqual(result.events[0].seq_id, seq_id)
        self.assertIsNone(result.events[0].emitted_token_id)
        self.assertFalse(result.events[0].finished)

    def test_generate_happy_path_remains_compatible(self) -> None:
        outputs = self.engine.generate(
            [[1, 2]],
            SamplingParams(temperature=1.0, max_tokens=2, ignore_eos=True),
            use_tqdm=False,
        )

        self.assertEqual(
            outputs,
            [{"text": "101 102", "token_ids": [101, 102]}],
        )

    def test_engine_exposes_resettable_scheduler_observability(self) -> None:
        self.engine.add_request(
            [1, 2],
            SamplingParams(temperature=1.0, max_tokens=2, ignore_eos=True),
        )
        self.engine.step_with_events()

        snapshot = self.engine.observability_snapshot()
        self.assertEqual(snapshot["kv_cache"]["peak_used_blocks"], 1)
        self.assertEqual(snapshot["preemption_count"], 0)

        self.engine.reset_observability()
        reset = self.engine.observability_snapshot()
        self.assertEqual(reset["kv_cache"]["peak_used_blocks"], 1)
        self.assertEqual(reset["preemption_count"], 0)

    def test_mixed_step_samples_decode_but_not_partial_prefill(self) -> None:
        mixed = scheduler_config()
        mixed.scheduler_policy = "mixed_token_budget"
        mixed.decode_token_budget = 4
        self.engine.scheduler = Scheduler(mixed)
        self.engine.model_runner = FakeMixedModelRunner()
        running_id = self.engine.add_request(
            [1, 2],
            SamplingParams(temperature=1.0, max_tokens=3, ignore_eos=True),
        )
        first = self.engine.step_with_events()
        self.assertEqual(first.phase, "prefill")
        waiting_id = self.engine.add_request(
            list(range(10)),
            SamplingParams(temperature=1.0, max_tokens=2, ignore_eos=True),
        )

        result = self.engine.step_with_events()

        self.assertEqual(result.phase, "mixed")
        self.assertEqual(
            [(event.seq_id, event.phase) for event in result.events],
            [(running_id, "decode"), (waiting_id, "prefill")],
        )
        self.assertIsNotNone(result.events[0].emitted_token_id)
        self.assertIsNone(result.events[1].emitted_token_id)
        self.assertEqual(self.engine.model_runner.sampled_phases[-1], ["decode"])


if __name__ == "__main__":
    unittest.main()
