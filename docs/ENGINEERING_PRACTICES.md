# 工程与实验实践

本文记录仓库中真实存在的工程约束，以及这些约束解决的失败模式。它不是抽象 checklist。

## 1. Deterministic Config 与 Frozen Trace

Formal run 使用预生成 arrival trace、固定 seed、固定 model revision/hash、显式 SLO 和完整 engine config。Matched pair 只能改变候选 policy 与 run identity；trace、seed、model、SLO 和非候选参数必须相同。

原因：若 arrival 在运行时随机生成，Router/Scheduler 的差异会与 workload sampling 混在一起，无法判断 tail 变化来自策略还是输入。

## 2. Request Lifecycle 与 Step Events

Request 级事件覆盖 arrival、route、admit/reject、first token、每个 output token 和 terminal。Scheduler step 事件覆盖 policy/mode、waiting/running、KV before/after、scheduled phase/tokens、preemption victim、recompute、waiting age 和 progress gap。

原因：summary 只能报告“发生了什么”，raw timeline 才能回答“为什么”。Stage 15 正是通过 step events 将 ITL failure 定位到 unbounded drain，而不是继续调 victim weight。

## 3. Raw-event Reaggregation

Aggregator 不直接信任 run summary。它重新读取 requests、steps、admission/cache events，计算 TTFT、ITL、Goodput、Preemption、Actual Recompute、terminal 和 final KV，并与 summary 交叉校验。

缺文件、截断 JSONL、request count 不守恒、pending terminal、final KV 非零或重算不一致都会使 cell invalid。

## 4. Fingerprint 与 Artifact Immutability

Formal artifact 记录或绑定：

- legacy `code_sha` / `code_commit`；
- runner 与关键模块 SHA；
- `nanovllm` package tree hash；
- config SHA；
- trace SHA；
- model revision/file hash；
- runtime/package/hardware metadata。

Protocol、fingerprint 或执行代码改变时必须使用新 run ID，不能覆盖旧 artifact。Clean repository 保留历史 SHA 原值；这些值属于 legacy repository namespace。

## 5. Multi-seed Matched Benchmark

主要 formal comparison 使用 3 seeds，并报告 matched pair 方向，而不是挑选最好 seed。不同 offered load 是不同 operating point，不把百分比简单平均成单一数字。Baseline=0 时只报告 absolute delta。

冻结 diagnostic matrix 不等价于生产总体抽样，因此不制造无依据的 p-value 或跨硬件置信区间。

## 6. Terminal 与 Final KV Reconciliation

一次 run 结束必须同时满足：

- offered = finished + rejected + failed/cancelled；
- 无未解释 pending request；
- reservation 全部释放；
- physical KV final used blocks 为 0；
- P/D source/destination block state 均已释放；
- output consistency contract 成立。

这避免“只看完成请求”而忽略泄漏、超时或被静默丢弃的 denominator。

## 7. Physical GPU Provenance

Multi-GPU correctness 不以配置中的 `gpu_id` 或 `CUDA_VISIBLE_DEVICES` 为证据。Runner 记录并验证：

```text
role + worker PID
→ Torch logical device
→ Torch physical UUID == NVML physical UUID
→ model residency on expected UUID
→ positive CUDA event time
→ cross-worker execution overlap
→ required request distribution
```

配置只表达 intent；Torch/NVML/CUDA event 才表达 runtime truth。该 gate 用于隔离旧 launcher 的 same-physical-GPU contamination。

## 8. Independent Review 与 Fail-closed

Corrected Stage 18/P0 的 reviewer 从 raw cells 重算 matched pairs，不只检查已有 summary。Review 检查 cell 数、seed/load/router coverage、trace/config identity、physical UUID、CUDA overlap、terminal/KV reconciliation 和 aggregate 数字。

任何 gate 失败，整个 formal result 不进入 [`FINAL_RESULTS.md`](FINAL_RESULTS.md)。旧 pre-fix artifacts 没有迁入本仓库，不存在“仍可被 README 误引用”的路径。

## 9. Test Hierarchy

| 层级 | 作用 | 代表测试 |
|---|---|---|
| Pure policy unit | cost、ordering、budget、guard | `test_recompute_aware_scheduler.py`、`test_mixed_token_budget_scheduler.py` |
| Resource invariant | block/reservation/incremental allocation | `test_prefix_aware_admission.py`、`test_incremental_kv_allocation.py` |
| Lifecycle/runner | open-loop、terminal、metrics | `test_lifecycle_metrics.py`、`test_open_loop_runner.py` |
| Multi-Replica/P&D | routing、endpoint、remapping | `test_multi_replica_router.py`、`test_pd_disaggregation.py` |
| Evidence pipeline | raw aggregation、fingerprint、fail-closed | Stage 11/12/15/16 aggregation tests、`test_final_characterization.py` |

CPU/non-GPU tests 验证逻辑和 evidence contract；它们不能替代真实 GPU performance benchmark。

## 10. Invalid Artifact Quarantine

Legacy repository 保存完整 invalid history 和 contamination audit。Clean repository 不复制 pre-GPU-binding-fix Stage 18/P0/Stage 19 performance，也不复制 calibration/timeout/local smoke。Provenance 文档只说明它们存在和为何无效，不提供容易被误用的本地 performance 文件。
