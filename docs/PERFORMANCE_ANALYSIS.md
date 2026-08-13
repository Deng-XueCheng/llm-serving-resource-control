# 性能分析

本文解释最终结果背后的机制，不增加任何新指标。精确数字和 evidence path 以 [`FINAL_RESULTS.md`](FINAL_RESULTS.md) 为准。

## 1. Admission：为什么 Prefix-aware 有效，也会失效

Full-footprint KV Admission 假设每个 request 都要独占完整生命周期 blocks。在高 prefix reuse 下，活跃请求已经持有共享 Prefix，重复 reservation 会制造虚假容量不足。`prefix_aware_fifo` 扣除 active shared blocks，因而能接纳更多真实可行请求，减少 reject 并提高 Goodput。

但该收益要求“共享后剩余增量需求”仍能被物理 KV 容纳。4-block 极低容量下，系统没有足够缓冲应对并发增量 allocation，candidate 反而退化。结论不是 Prefix-aware 永远更优，而是 Admission 必须区分 active sharing、inactive cache 和 capacity headroom。

Slack-aware Admission 解决的是另一个问题：queue head 已经不可满足 deadline 时，继续等待会阻塞后续请求。Early rejection 可以恢复 FIFO progress，但 baseline=0 的场景只能报告 absolute delta，不能制造相对提升百分比。

## 2. Recompute：资源减少为何不自动改善延迟

Recompute-aware victim model 同时减少抢占频率和单次 victim harm，因此 Actual Recompute、Preemption、Scheduler Steps 与 elapsed 都下降。然而 unbounded drain 把执行机会集中给少量 resident Decode：它们更快自然完成并释放 KV，pending requests 却出现长 token gap。

这解释了 Stage 15 的表面矛盾：资源 work 明显减少，但 ITL P99 恶化。Elapsed 是完成整个冻结 workload 的代价；ITL 是单请求连续 token 的服务质量，两者不是同一目标。

Bounded Drain 的作用不是推翻 victim model，而是给 drain episode 增加 progress boundary。Episode budget、waiting-age bound 与 SLO watch 把长 drain 切成有限窗口，因此相对 unbounded baseline，ITL/progress gap/waiting age 改善；同时仍保留相对 pressure baseline 的 Recompute/Preemption 收益。TTFT 小幅退化说明该折中仍不是所有指标的统一最优。

## 3. Multi-Replica：throughput 不变为何 Goodput 上升

Round Robin 按请求数轮转，但请求的 prompt/output、KV footprint 和到达时刻并不等价。一个 Replica 可能积累更长 queue 和更高 KV utilization，而另一个 Replica 尚有可服务空间。最终 drain 时，两者仍能完成相同 token 总量，所以 achieved throughput 相同；但 skewed Replica 上的请求更容易错过 TTFT/ITL SLO。

Resource-aware Routing 在 Admission 前读取 capacity、queue 和 KV snapshot，把请求导向当前更可行的 Replica。它减少 queue/KV imbalance，降低 tail latency，使更多 completion 落入 SLO，从而提高 Goodput。该机制改善的是 **SLO serving efficiency**，不是模型计算吞吐。

## 4. Saturation：收益为何在 knee 附近出现

低 load 下两个 Replica 都有充分 headroom，任意合理 Router 都能及时服务，额外 state-aware decision 没有稳定收益。P0-1 在 3.6 req/s 观察到 Round Robin queue/latency 转折；此后小的分配偏差会累积成持续 queue skew，Resource-aware 才显著延缓 collapse。

P0 的 throughput 随 offered load 和最终 drain 线性变化，不能用 throughput plateau 定义 capacity。这里的 knee 来自 measurement window 内 queue growth、TTFT/ITL 变陡和 Goodput 转折。

## 5. Workload Shape：Prefix locality 与 balance 的冲突

- **Prefix-heavy**：Round Robin 容易把高工作量请求不均匀地堆到 Replica；Resource-aware 显著改善 balance。Corrected artifacts 中 matched prefix blocks 没增加，因此收益主要不是 prefix affinity。
- **Prefill-heavy**：更均匀的 queue 降低 TTFT，但 Prefill 分配变化会轻微干扰 Decode cadence，出现 ITL trade-off。
- **Decode-heavy**：TTFT/queue 改善，但更长 resident Decode 集合改变 token interleaving，ITL 和 Goodput 退化。
- **Balanced**：原始 imbalance 已接近 1，额外 resource score 没有足够可纠正的 skew，反而扰动原本有效的轮转，形成明确 regression。

因此 Router 需要 workload-aware policy selection 或 guardrail，而不是默认在所有 workload 开启同一策略。

## 6. Mixed Scheduling：设计复杂度与 evidence 强度不同

Unified token budget 解决了调度接口和 accounting：一个 step 可以表达 Decode 与 partial Prefill progress，incremental KV allocation 也能按实际 scheduled tokens 申请 blocks。但执行仍拆成 phase-aware sub-batch，kernel 层没有 fused mixed Attention。

Mixed SLO allocator 和 incremental allocation 的 diagnostics 没有形成比 Stage 16/18 更强的 formal performance evidence，因此它们属于系统设计能力和边界分析，不应占用核心量化主张。

## 7. P/D：isolation benefit 为何被抵消

1P1D 把 Prefill 与 Decode 放到不同 physical GPU，理论上减少 phase contention；实际链路却增加了 Prefill worker queue、destination allocation、KV materialization、pinned-host transport、block remapping 和 state restore。

Corrected diagnostic 中，Prefill queue 已达到秒级，KV transfer 还增加每请求数百毫秒。Isolation benefit 小于 queue + transfer cost，TTFT 和 Goodput 退化。该结果说明 P/D 的可行条件至少包括更低 Prefill queue、更高效 transport（例如真正 P2P）和足够 workload concurrency；当前 evidence 不支持继续扩大矩阵。

## 8. 结果解释原则

1. 不跨 baseline 拼接 Stage 16 指标。
2. 不把 final-drain throughput 当 steady-state capacity。
3. 不把 single matched P/D diagnostic 写成稳定统计结论。
4. 不隐藏 Balanced regression、Stage 15 ITL failure 或 Prefix-aware 低容量反例。
5. 不用 pre-GPU-binding-fix 数据解释 Multi-GPU performance。
