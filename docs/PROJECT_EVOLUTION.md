# 技术演进

本文按问题演进组织，而不是把 Stage 编号当作功能清单。完整 Stage history 保存在 legacy repository；本仓库保留能解释最终系统决策的最小时间线。

## 1. 从 Offline Engine 到 Open-loop Serving Measurement

### Problem

Nano-vLLM upstream 的 offline generation 无法直接回答 arrival queue、TTFT、ITL、Goodput 与 KV pressure。

### Observation / Root Cause

按完成顺序发请求会隐藏排队；只记录最终 throughput 会丢失 first-token、token gap、preemption 和 terminal semantics。

### Design / Implementation

构建 frozen arrival trace、Open-loop injection、request/step JSONL、TTFT/ITL/E2E、SLO Goodput、config/trace/code fingerprint 与失败 artifact 保存。

### Result / Decision

形成所有后续 matched benchmark 的共同分母。观测表明 Prefill/Decode 竞争需要显式 progress 与 KV 语义。

## 2. 从 Phase Priority 到 KV-aware Admission

### Problem

Prefill-first 保护新请求但延迟 Decode；Decode-first 保护 ITL却可能阻塞 waiting request。静态 token budget 无法说明 KV full 时为何反复抢占。

### Observation / Root Cause

极限 KV 容量下出现 `KV full → preempt → WAITING → repeated Prefill`。仅调整 Decode budget 或 Progress Guard 产生 TTFT/ITL trade-off，没有 Pareto win。

### Design / Implementation

显式记录 KV peak/final、preemption、OOM 和 pressure；引入 full-request KV reservation、FIFO Admission 与 terminal release。

### Result / Trade-off / Decision

Admission 能阻止不可服务 offered load 进入 Scheduler，但保守 full footprint 会拒绝可复用 Prefix 的请求，下一步转向 Prefix-aware semantics。

## 3. Prefix-aware 与 Slack-aware Admission

### Problem / Root Cause

有 Prefix Cache 时，active shared blocks 不应重复 reservation；inactive cached blocks 又不能被当作永久共享容量。极低容量下，不可行 FIFO head 还会阻塞后续请求。

### Design

`preview_prefix_cache()` 无副作用地计算 matched/active prefix；Admission 只预留 incremental KV。Slack-aware variant 再使用 queue waiting、service ETA、TTFT/ITL deadline 做 early infeasibility rejection。

### Result

在 90% reuse、8 KV blocks、3 seeds 下，Prefix-aware Admission 的 SLO Goodput `+83.3%`、mean reject `-55.6%`；但 4-block 条件下出现退化。Slack-aware 在冻结 structured-pressure workload 中恢复 FIFO head，但不被外推为通用性能提升。

### Decision

Admission 控制入口，不能解决已接纳 request 的 preemption/recompute，因此进入 Scheduler 内部资源治理。

## 4. Pressure-aware Scheduling：机制暴露而非最终答案

### Hypothesis

根据 KV utilization、recent preemption 与 waiting age 动态切换 Prefill/Decode，可能降低 pressure。

### Result / Trade-off

策略揭示了 pressure 信号和 progress conflict，但仍在 Waiting allocation 与 Decode expansion 两条路径产生不一致 victim choice。减少局部 pressure 不等价于减少真实 Recompute。

### Decision

停止继续叠加阈值，转向 request-level victim cost 和 Actual Recompute observability。

## 5. Chunked Prefill：达到 Stop Condition

### Hypothesis

更小 Prefill chunk 可以降低对 Decode 的单次阻塞。

### Implementation / Result

在 upstream chunked Prefill 基础上增加预算与 contention-aware variant。两轮 matched smoke 中 TTFT、ITL、elapsed、steps 和 preemption 同时退化。

### Decision

瓶颈不是固定 chunk 的单步阻塞，而是 request-level preemption/recompute feedback。未扩大 formal matrix，转向 victim selection。

## 6. Recompute-aware Scheduling：资源机制成功，Serving 失败

### Problem / Root Cause

deque-tail victim 没有表达 resident KV、真正可释放 blocks、remaining decode 与 recompute harm。

### Design / Implementation

构建 `ResidentSequenceCost`，统一 Waiting allocation 和 Decode expansion 两条 victim path；允许高压力时推进更可能自然释放 KV 的 resident Decode，并从 raw step events 计算 Actual Recompute。

### Result

正式 18-cell / 9-pair matrix 中，Actual Recompute `-93.65%`、Preemption `-88.77%`，均 9/9 改善；但 ITL P99 cell mean `+834.56%`，Goodput 下降。Timeline 显示无界 `drain_decode` 将资源收益换成多秒 token starvation。

### Decision

这是 mechanism success / serving failure。停止 victim/pressure 参数 sweep，下一步把 progress/SLO bound 作为一等约束。

## 7. SLO-aware Bounded Drain

### Problem / Design

无界 drain 的终点是 resident exhaustion，缺少 waiting/post-token progress 上界。引入 episode budget、waiting-age bound、entry SLO watch、有限 Prefill window 和明确 exit reason。

### Result

相对 unbounded recompute，ITL P99 `-80.70%`、progress-gap P99 `-81.58%`、max waiting age `-92.48%`，均 9/9 改善；相对 pressure baseline，Actual Recompute `-75.63%`、Preemption `-54.37%`，但 TTFT P99 aggregate `+6.58%`。Latency 与 resource 使用不同 baseline，保持分开报告。

### Decision

Bounded Drain 建立资源效率与 token fairness 的可验证折中，但不声称所有 SLO 指标全面提升。

## 8. Unified Mixed Prefill/Decode Token Budget

### Problem

Phase-exclusive high-level policy 难以同时表达 Decode progress、Prefill deadline 与 global token budget。

### Design / Implementation

Scheduler 在一个 step 内分配 `max_num_batched_tokens`，输出 per-request phase metadata，支持 partial Prefill、Decode accounting、SLO/age-aware allocation 与 incremental KV allocation。ModelRunner 仍按 phase 拆分 eager sub-batch。

### Result / Trade-off

功能与 invariants 通过 tests；Mixed SLO allocator 和 incremental allocation 的 diagnostics 显示 workload-dependent trade-off，没有被包装成稳定 formal performance win。该阶段为 Multi-Replica local engine 提供统一调度接口。

## 9. Multi-Replica Resource-aware Routing

### Problem / Root Cause

多个独立 Replica 中，Round Robin 忽略 queue、KV footprint 和 service progress，接近 saturation 时形成 queue/KV skew。

### Design

Global Router 在 Admission 前读取 capacity feasibility、free/used KV、waiting/running、oldest waiting age 与 deterministic tie-break；目标 Replica 仍维护独立 Admission、Scheduler 和 KV/Prefix Cache。

### Correctness Incident

旧 launcher 的两个 worker 实际落在同一 physical GPU。配置中的 logical GPU index 被误当成 runtime truth，旧 Multi-GPU performance 因此作废。修复后加入 Torch/NVML UUID、model residency、CUDA work/overlap、request distribution 与 final KV gate。

### Corrected Result

12-cell / 6-pair corrected formal matrix review `PASS`。4.8/6.0 req/s 下 TTFT、ITL、Goodput 和 queue/KV balance 改善，raw throughput 不变。完成的是 corrected Stage 18；作废的只是 pre-fix artifacts。

## 10. Saturation 与 Workload Boundary

### Problem

两个 overload operating point 不能说明 Router 从何时有效，也不能说明 workload shape 的边界。

### Result

P0-1 的 30-cell matrix 定位 saturation onset 为 `3.6 req/s`：低负载差异有限，接近 saturation 后 Resource-aware 延缓 tail/SLO collapse。P0-2 的 24-cell matrix 显示 Prefix-heavy 最大收益，Prefill/Decode-heavy 存在 trade-off，Balanced 明确退化。

### Decision

最终主张限定为 saturation/workload-dependent SLO serving efficiency，而不是普遍 throughput improvement。

## 11. P/D Disaggregation：功能完成，性能边界为负

### Hypothesis / Implementation

以 1P1D 隔离 Prefill 与 Decode，完成 destination allocation、logical remapping、pinned-host staging、sequence/RNG restore、output consistency 和资源释放。

### Corrected Result

不同 physical GPU 上的 matched diagnostic 显示 mean KV transfer 约 `389 ms/request`、Prefill queue mean 约 `5.58 s`；TTFT、ITL 与 Goodput 均未形成正收益。

### Decision

`queue + transfer overhead > isolation benefit`，停止扩展 formal matrix。保留为架构边界和 transport/queue measurement 案例。

## Legacy Stage 映射

| 问题阶段 | Legacy 编号 | 最终状态 |
|---|---|---|
| Benchmark / observability | Stage 1–5 | 基础设施 |
| Decode budget / progress | Stage 6–7 | historical trade-off |
| KV pressure / early Admission | Stage 8–10 | 机制演进 |
| Prefix / Slack Admission | Stage 11–12 | valid formal / boundary |
| remote KV 计划 | Stage 13 | 未实施 |
| Chunked Prefill | Stage 14 | stopped diagnostic |
| Recompute-aware | Stage 15 | mechanism success / serving failure |
| Bounded Drain | Stage 16A | valid formal |
| Mixed scheduling | Stage 17A/17B/17C | functional / diagnostic |
| Multi-Replica Routing | Stage 18 corrected | valid formal |
| 1P1D | Stage 19 corrected | valid negative diagnostic |
| Saturation / workload boundary | P0-1 / P0-2 | release characterization，非 Stage 编号 |
