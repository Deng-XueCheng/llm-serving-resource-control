# Multi-Replica Routing、Saturation 与 Goodput

## Round Robin 的盲区

Round Robin 均衡 request count，不均衡实际 work。不同 prompt/output、Prefix 命中和 KV footprint 会让两个 Replica 的 queue service time 与 KV utilization 分叉。

## Resource-aware Snapshot

Global Router 可以读取：

- capacity feasibility；
- total/free/used KV 和 utilization；
- waiting/running request 数；
- oldest waiting age；
- optional prefix preview；
- deterministic tie-break state。

Router 只选择目标；每个 Replica 的 Admission/Scheduler/KV state 仍独立。

## 为什么 Throughput 不变

在最终 drain 的固定 token workload 中，两种 Router 都可能完成相同 token 总量，achieved throughput 因而相同。差异发生在 completion timing：queue-skewed Replica 的请求错过 TTFT/ITL deadline，降低 SLO Goodput。

Resource-aware 通过降低 skew，让更多请求在 SLO 内完成。正确表述是提高 saturation 区间的 SLO serving efficiency，而不是提高模型吞吐。

## Saturation 依赖

低 load 下所有 Replica 都有 headroom，Router 差异很小。接近 knee 后，微小分配误差会积累成持续 queue growth，资源状态才成为高价值信号。Saturation 应结合 queue/latency/Goodput 转折判断，不能只看 final-drain throughput plateau。

## Workload Boundary

Prefix-heavy 可能同时涉及 locality 与 load balance，但必须从 matched prefix blocks 证明 locality 贡献。Balanced workload 原本 skew 很小，额外 score 可能扰动有效轮转并退化。最终 Router 需要 activation guard，而不是默认适用于所有 workload。
