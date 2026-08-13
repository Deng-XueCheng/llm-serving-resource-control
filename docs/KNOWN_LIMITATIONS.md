# 已知边界与停止条件

## 系统范围

- 这是研究型 resource-control system，不是生产级 vLLM 替代品。
- 没有自研 FlashAttention、PagedAttention、CUDA/Triton kernel；这些能力属于 Nano-vLLM upstream。
- 没有实现跨 Replica 共享 KV Cache 或 distributed KV manager。
- Prefix-aware Router score 存在于源码，但 corrected final Routing 收益主要来自 load balance，不能声称 prefix affinity 提升。

## Evidence 外推

- 正式结果绑定固定模型 revision、runtime、hardware、config、trace 和 seed。
- 不证明跨模型、跨 GPU、跨集群或生产稳态流量泛化。
- 三个 seed 和冻结 diagnostic matrix 支持 matched comparison，不构成生产总体统计抽样。
- Absolute TTFT/ITL 不应脱离实验环境引用。

## Admission

- Prefix-aware Admission 在 90% reuse、8 blocks 有收益，但在 4 blocks 观察到退化。
- Slack-aware Admission 的 baseline 没有完整 interactive token sample，不能跨策略比较 TTFT/ITL。

## Scheduler

- Stage 15 的 Recompute-aware 策略是资源机制成功、在线 ITL 失败。
- Bounded Drain 相对不同 baseline 分别报告 latency 与 resource；不能拼成一个“全面提升”结论。
- Stage 16 多数 cells 的 SLO Goodput 为 0，不声称 Goodput 提升。
- Mixed SLO allocator 与 incremental KV allocation 缺少同等强度的 formal performance evidence。
- Mixed Scheduler 不是 fused mixed Attention 或 single-forward mixed batch。

## Routing

- Resource-aware Routing 的收益主要出现在 saturation 附近；低负载没有稳定优势。
- Prefill-heavy 和 Decode-heavy 存在 ITL/Goodput trade-off。
- Balanced workload 的 TTFT、ITL 和 Goodput 均退化，是明确 no-benefit boundary。
- Corrected Stage 18 改善 tail/Goodput/balance，raw throughput 不变。

## P/D

- 1P1D transport 使用 pinned-host staging，不是 direct GPU P2P。
- Corrected performance 只有一个 matched diagnostic。
- Prefill queue + transfer overhead 超过 isolation benefit，故停止正式矩阵扩展。

## Repository Reconstruction

- Clean repository 不包含 legacy Git objects，因此 formal artifact 中的历史 code SHA 必须在 legacy repository 查询。
- Frozen execution configs 可能记录历史机器的绝对 model path；它们作为不可改写 provenance 保留。可运行命令通过 `--model-path` 覆盖，正式文档不依赖这些路径。
- Legacy 全量 Stage reports、invalid artifacts 和 profiling dumps 未迁移，需通过 legacy repository 审计。
