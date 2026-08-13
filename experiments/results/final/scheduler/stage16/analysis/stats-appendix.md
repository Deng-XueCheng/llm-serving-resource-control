# Stage16A 统计附录

> 本文档为项目历史阶段统计附录；最终 baseline 分离口径请参阅 `docs/FINAL_RESULTS.md`。

## 方法

分析单位为冻结的 matched configuration points，不把各 policy cell 当作独立样本。每项比较使用配对差值（lower-is-better 指标定义为 baseline - bounded），报告均值、样本标准差、matched-point bootstrap percentile 95% 区间、精确 sign test、精确 Wilcoxon signed-rank p 值及 Holm 校正后的 p 值。capacity 之间可能共享同一 seed/trace，因此这些区间和 p 值只作为描述性敏感性分析，不视为来自独立生产总体的推断；正式 gate 仍以预注册 ratio-of-totals reduction 与 direction criterion 为准。

## 配对比较

| 指标 | baseline → candidate | baseline mean±SD | candidate mean±SD | mean reduction | improved | Sign-test p | Wilcoxon p | Holm p | difference 95% interval |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `post_token_progress_gap_p99_steps` | recompute → bounded | 245.450 ± 63.606 | 45.212 ± 7.214 | 80.58% | 9/9 | 0.0039 | 0.0039 | 0.0234 | [162.209, 236.646] |
| `max_waiting_age_steps` | recompute → bounded | 475.889 ± 94.643 | 35.778 ± 0.833 | 92.12% | 9/9 | 0.0039 | 0.0039 | 0.0234 | [377.111, 493.111] |
| `itl_p99_ms` | recompute → bounded | 5,364.825 ± 1,351.223 | 1,035.439 ± 179.777 | 79.77% | 9/9 | 0.0039 | 0.0039 | 0.0234 | [3,518.310, 5,070.850] |
| `actual_recompute_tokens` | pressure → bounded | 186,762.556 ± 20,707.194 | 45,515.111 ± 30,438.026 | 76.74% | 9/9 | 0.0039 | 0.0039 | 0.0234 | [130,931.092, 152,112.444] |
| `preemption_count` | pressure → bounded | 605.333 ± 120.052 | 276.222 ± 95.668 | 55.89% | 9/9 | 0.0039 | 0.0039 | 0.0234 | [305.667, 352.000] |
| `ttft_p99_ms` | pressure → bounded | 457.478 ± 87.520 | 487.574 ± 103.809 | -7.11% | 3/9 | 0.5078 | 0.3008 | 0.3008 | [-77.065, 22.873] |

## 解释边界

Wilcoxon、sign test 与 bootstrap 均把冻结配置点作为 matched observations；由于相同 seed 跨 capacity 共享 trace，这里不声称观测独立或外推总体显著性。多重校正覆盖本附录 6 项主要对比。即使某个 p 值未达到传统阈值，也不推翻预注册 gate；相反，若 aggregate gate 与统计检验方向冲突，应优先保留完整 pair-level 数据并降级结论。
