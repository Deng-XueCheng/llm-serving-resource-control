from __future__ import annotations

import unittest
from pathlib import Path

from experiments.snapshot_preflight import (
    verify_model_identity,
    verify_source_snapshot,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class CleanRepositoryContractTests(unittest.TestCase):
    def test_required_final_evidence_roots_exist(self) -> None:
        required = (
            "experiments/results/final/admission/stage11",
            "experiments/results/final/scheduler/stage16",
            "experiments/results/final/multi_replica/stage18",
            "experiments/results/final/characterization/p0_1",
            "experiments/results/final/characterization/p0_2",
            "experiments/results/final/pd/stage19",
        )
        self.assertEqual(
            [],
            [relative for relative in required if not (REPO_ROOT / relative).is_dir()],
        )

    def test_invalid_multi_gpu_results_are_not_migrated(self) -> None:
        forbidden = (
            "experiments/results/pre_gpu_binding_fix",
            "experiments/results/historical_invalid",
            "experiments/results/final/pre_gpu_binding_fix",
        )
        self.assertEqual(
            [],
            [relative for relative in forbidden if (REPO_ROOT / relative).exists()],
        )

    def test_clean_snapshot_does_not_embed_legacy_git_history(self) -> None:
        nested_git = [
            path
            for path in REPO_ROOT.rglob(".git")
            if path.resolve() != (REPO_ROOT / ".git").resolve()
        ]
        self.assertEqual([], nested_git)

    def test_source_manifest_covers_the_clean_snapshot(self) -> None:
        result = verify_source_snapshot()
        self.assertGreater(result["verified_files"], 0)

    def test_multi_gpu_model_contract_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "model_repo_id"):
            verify_model_identity({"model_path": str(REPO_ROOT)})


if __name__ == "__main__":
    unittest.main()
