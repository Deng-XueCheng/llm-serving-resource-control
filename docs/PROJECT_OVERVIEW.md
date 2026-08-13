# 项目总览

## 项目目标

本项目研究 LLM Serving 中请求、KV Cache、Prefill/Decode execution 与 SLO 之间的资源控制问题。系统基于 Nano-vLLM 的轻量推理引擎扩展，从 Open-loop arrival 开始，覆盖 Global Routing、Replica-local Admission、Local Scheduling、KV allocation、Model Execution、Lifecycle Observability 和 Evidence Review。

目标不是为单一 benchmark 添加若干 heuristic，而是建立一个可以回答以下问题的完整系统：

- 新请求需要多少**增量** KV，而不是只看完整 prompt/output footprint？
- KV 满时，应该继续 Prefill、推进 Decode、抢占谁，还是拒绝请求？
- 减少 Preemption/Recompute 后，如何防止 Decode drain 造成 token starvation？
- 多个独立 Replica 中，如何用实时资源状态降低 queue/KV skew？
- 为什么 throughput 不变时，TTFT、ITL 与 SLO Goodput 仍可能显著变化？
- 如何证明 Multi-GPU benchmark 实际使用了不同 physical GPU？

## 系统主线

```text
Request Arrival
→ Global Routing
→ Replica-local Admission
→ Local Scheduling
→ KV Cache Resource Management
→ Model Execution
→ Lifecycle Metrics
→ Evidence Aggregation / Review
```

该链路中的每层都有独立职责：Router 选择 Replica，Admission 决定是否接纳，Scheduler 分配 token progress，BlockManager 管理 local physical KV，ModelRunner 执行模型，evidence pipeline 证明结果可重算且实验前提成立。

## 设计闭环

项目的演进由观测和 stop condition 推动：

```text
Observation
→ Root Cause
→ Design
→ Implementation
→ Matched Measurement
→ Result / Trade-off
→ Continue, Redesign or Stop
```

例如，Recompute-aware Scheduling 成功减少了 Preemption 和真实恢复 Prefill，却让 ITL 严重退化。该结果没有被包装成综合优化，而是通过 raw timeline 定位到 phase-exclusive unbounded drain，随后引出 Bounded Drain。类似地，P/D isolation 的功能链路通过，但 corrected performance 为负，因此停止扩展 formal matrix。

## 最终交付范围

- final Nano-vLLM runtime 与 resource-control extensions；
- Open-loop、Multi-Replica 与 1P1D benchmark runners；
- deterministic config/trace、matched multi-seed comparison 与 raw-event reaggregation；
- Stage 11/12/15/16 single Engine final evidence；
- corrected Stage 18/19 和 P0-1/P0-2 Multi-GPU evidence；
- CPU/non-GPU contract tests、GPU provenance gates 与 repository validation；
- 中文系统文档、技术演进、性能分析和 provenance bridge。

## 证据边界

本仓库只承载 final valid evidence。旧 pre-GPU-binding-fix Stage 18/P0/Stage 19、失败协议轮次、temporary smoke、profiling dumps 和完整历史 Stage 文档由 legacy repository 保管。正式数字的唯一摘要入口是 [`FINAL_RESULTS.md`](FINAL_RESULTS.md)。

## 上游归属

Nano-vLLM upstream 提供 Continuous Batching、Paged KV Cache、Prefix Cache、FlashAttention、CUDA Graph、Triton KV write、Tensor Parallel 以及基础 Scheduler/BlockManager/ModelRunner。本项目是在这些组件上构建 LLM Serving resource control 与 evidence infrastructure，不把 upstream 能力描述为本项目原创。详见 [`../NOTICE.md`](../NOTICE.md)。
