# 最终合法结果

本文是 clean repository 的量化结果唯一摘要入口。数字来自 legacy release `v1.0-serving-final` 的合法 artifacts；新仓库没有重新运行或改写 benchmark。历史 `code_sha` 仍指向 legacy repository，详见 [`PROVENANCE.md`](PROVENANCE.md)。

## 口径

- matched pair 要求 workload、trace、seed、model、SLO 与非候选配置一致；
- relative change 只在 baseline 非零且 denominator 一致时计算；
- Stage 16 latency 与 resource 使用不同 baseline，分表报告；
- P0 final-drain throughput 不等于 steady-state capacity；saturation knee 由 queue/latency growth 定义；
- Multi-GPU 只使用 `experiments/results/final/` 下迁移的 post-GPU-binding-fix artifacts。

## 1. Prefix-aware Admission

**Problem**：full footprint reservation 重复计算 active shared Prefix。
**Baseline**：`kv_aware_fifo`。
**Candidate**：`prefix_aware_fifo`。
**Workload**：90% prefix reuse、8 KV blocks；3 seeds，完整正式矩阵 54 cells。

| Metric | Baseline | Candidate | Absolute delta | Relative change |
|---|---:|---:|---:|---:|
| Interactive SLO Goodput | 0.1333 rps | 0.2444 rps | +0.1111 rps | +83.3% |
| mean reject count | 6.00 | 2.67 | -3.33 | -55.6% |

Evidence：[aggregate](../experiments/results/final/admission/stage11/stage11_formal_r1_aggregate_r2.json) 与同目录 raw events。
**Limitation**：90% reuse、4 KV blocks 下 candidate 退化；收益不能外推到所有容量。

## 2. Slack-aware Admission Boundary

**Problem**：不可行 FIFO head 阻塞后续请求。
**Baseline → Candidate**：Prefix-aware FIFO → Slack-aware Prefix FIFO。
**Workload**：6-block structured-pressure、3 seeds。

Goodput `0 → 0.25 rps`，mean reject `2 → 1`。Baseline 没有完成 interactive token sample，因此不计算 TTFT/ITL relative comparison。

Evidence：[aggregate/raw](../experiments/results/final/admission/stage12/)。
**Limitation**：用于证明 early infeasibility rejection 与 FIFO head recovery，不作为通用性能主张。

## 3. Recompute-aware Mechanism Boundary

**Baseline**：`pressure_aware_decode`。
**Candidate**：unbounded `recompute_aware`。
**Workload**：KV 4/6/8 × seed 1/2/3；18 cells / 9 matched pairs。

Actual Recompute `1,621,467 → 103,012 tokens`（`-93.65%`），Preemption `5,459 → 613`（`-88.77%`），均 9/9 改善；但 ITL P99 cell mean `568.44 → 5312.38 ms`（`+834.56%`），Goodput 下降。
Evidence：[aggregate/raw](../experiments/results/final/scheduler/stage15/) 与同目录 timeline artifact。
**Limitation**：mechanism success / serving failure，不得写成在线延迟优化。

## 4. SLO-aware Bounded Drain

**Workload**：KV 4/6/8 × seed 1/2/3；27-cell frozen matrix，9 matched triples。

### 相对 unbounded recompute baseline

| Metric | `recompute_aware` | bounded | Absolute delta | Relative change | Pair direction |
|---|---:|---:|---:|---:|---:|
| ITL P99 aggregate | 48283.42 ms | 9318.95 ms | -38964.47 ms | -80.70% | 9/9 |
| progress-gap P99 | 2209.05 | 406.91 | -1802.14 | -81.58% | 9/9 |
| max waiting age | 4283 | 322 | -3961 | -92.48% | 9/9 |

### 相对 pressure baseline

| Metric | `pressure_aware_decode` | bounded | Absolute delta | Relative change | Pair direction |
|---|---:|---:|---:|---:|---:|
| Actual Recompute | 1680863 | 409636 | -1271227 | -75.63% | 9/9 |
| Preemption | 5448 | 2486 | -2962 | -54.37% | 9/9 |
| TTFT P99 aggregate | 4117.30 ms | 4388.17 ms | +270.87 ms | +6.58% | 3/9 改善 |

Evidence：[aggregate](../experiments/results/final/scheduler/stage16/stage16_diagnostic_r2.aggregate.json)、[analysis](../experiments/results/final/scheduler/stage16/analysis/analysis.json) 与 raw events。
**Limitation**：多数 cells 的 Goodput 为 0，不声称 Goodput 提升；TTFT 相对 pressure baseline 有退化。

## 5. Corrected Multi-Replica Routing

**Baseline**：Round Robin。
**Candidate**：Resource-aware。
**Matrix**：2 loads × 2 routers × 3 seeds = 12 cells / 6 matched pairs；independent review `PASS`；legacy code SHA `3a35458eb513f9e9dba20229a377942fc2cccc25`。

### 4.8 req/s

| Metric | Round Robin | Resource-aware | Absolute delta | Relative change | Improved pairs |
|---|---:|---:|---:|---:|---:|
| TTFT P99 | 9330.83 ms | 5980.93 ms | -3349.90 ms | -35.90% | 3/3 |
| ITL P99 | 4060.71 ms | 1648.39 ms | -2412.32 ms | -59.41% | 3/3 |
| SLO Goodput | 1.20 rps | 1.8833 rps | +0.6833 rps | +56.94% | 2/3 |
| achieved throughput | 115.2 tok/s | 115.2 tok/s | 0 | 0% | equal |
| queue imbalance | 6.846 | 1.666 | -5.180 | -75.67% | 3/3 |
| KV imbalance | 0.336 | 0.212 | -0.124 | -36.91% | 3/3 |

### 6.0 req/s

| Metric | Round Robin | Resource-aware | Absolute delta | Relative change | Improved pairs |
|---|---:|---:|---:|---:|---:|
| TTFT P99 | 18444.97 ms | 7050.58 ms | -11394.39 ms | -61.77% | 3/3 |
| ITL P99 | 5606.62 ms | 3405.57 ms | -2201.05 ms | -39.26% | 3/3 |
| SLO Goodput | 0.90 rps | 2.2333 rps | +1.3333 rps | +148.15% | 3/3 |
| achieved throughput | 144 tok/s | 144 tok/s | 0 | 0% | equal |
| queue imbalance | 9.634 | 2.118 | -7.517 | -78.02% | 3/3 |
| KV imbalance | 0.323 | 0.175 | -0.148 | -45.94% | 3/3 |

Evidence：[review](../experiments/results/final/multi_replica/stage18/evidence_review.json) 与同目录 12 个 raw cells。
**Limitation**：这是 saturation/overload 下的 tail、SLO 与 balance 收益，不是 throughput 提升。

## 6. Saturation Characterization

P0-1：5 loads × 2 routers × 3 seeds = 30 cells / 15 matched pairs，review `PASS`。Saturation onset 为 **3.6 req/s**。

| Load | Router | TTFT P99 | ITL P99 | Goodput | Queue imbalance | Throughput |
|---:|---|---:|---:|---:|---:|---:|
| 2.4 | Round Robin | 2574.36 ms | 296.39 ms | 1.3333 rps | 1.708 | 57.6 tok/s |
| 2.4 | Resource-aware | 2843.36 ms | 343.55 ms | 1.3333 rps | 0.686 | 57.6 tok/s |
| 3.6 | Round Robin | 5503.22 ms | 1807.55 ms | 1.7000 rps | 4.889 | 86.4 tok/s |
| 3.6 | Resource-aware | 3376.30 ms | 400.55 ms | 2.0167 rps | 1.303 | 86.4 tok/s |
| 6.0 | Round Robin | 14214.59 ms | 4136.46 ms | 1.3667 rps | 9.328 | 144 tok/s |
| 6.0 | Resource-aware | 7799.76 ms | 3988.56 ms | 1.6167 rps | 3.092 | 144 tok/s |
| 7.2 | Round Robin | 19234.10 ms | 7511.77 ms | 0.8000 rps | 11.197 | 172.8 tok/s |
| 7.2 | Resource-aware | 13446.92 ms | 5287.46 ms | 1.0500 rps | 7.450 | 172.8 tok/s |

Evidence：[summary](../experiments/results/final/characterization/p0_1/summary.json)、[review](../experiments/results/final/characterization/p0_1/evidence_review.json) 与 raw cells。
**Limitation**：低负载没有稳定优势；final-drain throughput 不能解释为 steady-state plateau。

## 7. Workload-shape Boundary

P0-2：4 shapes × 2 routers × 3 seeds = 24 cells / 12 matched pairs，固定 4.8 req/s；所有 cells completion=1.0，same-workload throughput 不变。

| Workload | Round Robin → Resource-aware | Conclusion |
|---|---|---|
| Prefix-heavy | TTFT `27336.08 → 20347.30 ms`；ITL `105.47 → 98.29 ms`；Goodput `0.05 → 0.1333 rps` | 最大收益，主要来自 load balance |
| Prefill-heavy | TTFT `19329.44 → 17306.40 ms`；ITL `104.65 → 109.20 ms`；Goodput `0 → 0.0167 rps` | TTFT/balance 改善，ITL 退化 |
| Decode-heavy | TTFT `13616.37 → 10908.99 ms`；ITL `60.94 → 70.19 ms`；Goodput `2.6667 → 2.4833 rps` | TTFT/balance 改善，ITL/Goodput 退化 |
| Balanced | TTFT `18749.54 → 20625.41 ms`；ITL `86.22 → 98.57 ms`；Goodput `0.5667 → 0.3333 rps` | 明确 no-benefit boundary |

Evidence：[summary](../experiments/results/final/characterization/p0_2/summary.json)、[review](../experiments/results/final/characterization/p0_2/evidence_review.json) 与 raw cells。

## 8. P/D Boundary

**Baseline**：colocated execution。
**Candidate**：corrected cross-GPU 1P1D。
**Evidence**：一个 matched diagnostic。

Mean KV transfer 约 `389 ms/request`、total 约 `392.7 MB`；TTFT `4942 → 7798 ms`，ITL `894 → 910 ms`，Goodput `1.2 → 0 rps`；throughput/completion 相同。Prefill queue mean 约 `5.58 s`。

Evidence：[colocated](../experiments/results/final/pd/stage19/colocated.json)、[1P1D](../experiments/results/final/pd/stage19/pd.json)。
**Limitation**：功能链有效，但只有 single matched diagnostic 且性能为负；不作为性能优化结论。
