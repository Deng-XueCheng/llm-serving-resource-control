# Final Evidence Layout

本目录只保存从 legacy release `v1.0-serving-final` 迁移的合法最终 evidence。所有文件保持原始内容；目录重组不改变 artifact 内记录的 legacy path 或 SHA。

```text
admission/stage11/          Prefix-aware Admission formal matrix
admission/stage12/          Slack-aware Admission boundary
scheduler/stage15/          Recompute-aware mechanism / serving failure
scheduler/stage16/          Bounded Drain formal matrix and analysis
multi_replica/stage18/      corrected post-GPU-binding-fix formal evidence
characterization/p0_1/      saturation characterization
characterization/p0_2/      workload-shape boundary
pd/stage19/                 corrected 1P1D negative diagnostic
MANIFEST.sha256             evidence integrity manifest
```

旧 pre-GPU-binding-fix performance、calibration、temporary smoke 和 timeout artifacts 没有迁移。量化结论见 [`docs/FINAL_RESULTS.md`](../../../docs/FINAL_RESULTS.md)，历史 lineage 见 [`docs/PROVENANCE.md`](../../../docs/PROVENANCE.md)。
