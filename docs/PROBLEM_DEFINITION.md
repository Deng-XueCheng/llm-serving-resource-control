# 问题定义

## 1. LLM Serving workload 的特征

在线生成请求具有异质 prompt/output 长度、不同到达时间和不同 SLO。Open-loop arrival 下，请求按外部 trace 到达，系统不能通过“上一批完成后再发下一批”隐藏排队。一个有效 serving policy 必须同时面对 arrival burst、queue growth、KV residency、token progress 和 terminal completion。

## 2. KV Cache 为什么是关键资源

Autoregressive Decode 需要保留历史 token 的 Key/Value state。请求越长、并发越高，resident KV 越多。KV 不足时，系统只能等待、evict cache、preempt active request、recompute context 或拒绝新请求。因而 KV Cache 不只是 execution optimization，而是 Admission 和 Scheduling 的硬资源约束。

## 3. Prefill 与 Decode 的冲突

Prefill 一次处理多 token，能够快速给新请求产生 first token，但可能长时间占用计算并集中申请 KV；Decode 每步通常只推进一个 token，却决定持续 ITL。Prefill-first 容易保护 TTFT、伤害 resident Decode；Decode-first 容易保护 ITL、让 waiting request starvation。静态优先级不能表达当前 KV pressure、request slack 和 progress history。

## 4. Prefix reuse 改变资源语义

已有 Prefix Cache 时，完整 request footprint 不等于新增占用。需要区分：

- `full_required_blocks`：请求完整生命周期的上界；
- `active_shared_blocks`：仍被活跃 request 引用、可以安全共享的 blocks；
- `inactive_cached_blocks`：可能被 eviction 的 cache；
- `incremental_required_blocks = full_required_blocks - active_shared_blocks`。

若 Admission 对所有请求都预留 full footprint，会重复计算 shared prefix；若把 inactive cache 也视为保证共享容量，又会 overcommit。

## 5. FIFO 与静态 phase priority 的局限

FIFO 简单可解释，但不可行的 queue head 会阻塞后续可服务请求。静态 Prefill-first/Decode-first 只表达 phase preference，不表达 request feasibility、KV footprint、waiting age 或 TTFT/ITL slack。资源和 SLO 状态需要成为显式决策输入。

## 6. Preemption 与 Recompute amplification

在 KV 高压下，Waiting Prefill allocation 与 Decode expansion 都可能触发 preemption。被抢占 request 释放 KV 后回到 WAITING；恢复时除可命中的 Prefix blocks 外，其余上下文会重新 Prefill。错误 victim selection 会形成：

```text
KV pressure
→ preempt resident request
→ release KV
→ request returns WAITING
→ recompute historical context
→ consume scheduler steps and KV again
→ more pressure
```

因此只统计 preemption count 不够，还需从 step events 计算 Actual Recompute、recompute tokens/preemption、progress gap 与 waiting age。

## 7. Throughput、Tail Latency 与 Goodput 不等价

完成相同总 token 数，并不代表请求在 SLO 内完成。最终 drain 后的 achieved throughput 还会掩盖 measurement window 内的 queue growth。SLO Goodput 只统计满足 TTFT/ITL contract 的完成请求；Router 即使不改变 raw throughput，也可能通过降低 queue skew 和 tail latency 提高 Goodput。反之，资源 work 减少也可能因 starvation 造成更差 ITL。

## 8. Multi-Replica 为什么需要 Global Routing

多个 Replica 各自维护独立 queue、Scheduler、KV/Prefix Cache 和 ModelRunner。Round Robin 只保证请求数轮转，不保证 work、KV footprint 或 queue service time 均衡。异质请求或动态 KV pressure 下，Replica 会出现 queue/KV skew，需要在 Admission 前读取 local snapshot 做全局选路。

Global Router 不能直接修改 local KV，也不能绕过 local Admission；它解决“发给谁”，Admission 解决“能否接纳”，Scheduler 解决“如何推进”。

## 9. P/D isolation 的成本

Prefill/Decode Disaggregation 可以隔离两种 phase，但需要：

- Prefill worker 排队与执行；
- destination block allocation；
- KV tensor transport；
- logical-to-physical block remapping；
- sequence/RNG state restore；
- source/destination release。

若 transport 经过 pinned-host staging，且 Prefill queue 已很长，isolation benefit 可能小于 queue + transfer overhead。P/D 不能只凭架构直觉认定更快。

## 10. 最终 Research / Engineering Questions

- **RQ1**：如何利用 active Prefix reuse 建模真实 incremental KV demand，并建立可释放的 request-level reservation？
- **RQ2**：如何降低 KV pressure 下的 Preemption/Recompute amplification，同时限制 waiting 与 post-token starvation？
- **RQ3**：如何在一个 Scheduler step 内共享 Prefill/Decode token budget，而不错误声称 fused mixed execution？
- **RQ4**：如何根据独立 Replica 的实时 queue/KV state 改善 saturation 区间的 SLO serving efficiency？
- **RQ5**：如何证明 benchmark 的功能、物理拓扑、统计分母和 raw-to-summary lineage 都正确？
