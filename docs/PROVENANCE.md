# Repository 与 Evidence Provenance

## Repository Lineage

| 层 | 标识 | 用途 |
|---|---|---|
| Legacy repository | `Deng-XueCheng/nano-vllm-serving-lab-private` | 完整 Git history、Stage 演进、失败实验、invalid evidence、contamination audit |
| Upstream repository | `GeeeekExplorer/nano-vllm` | Nano-vLLM engine 基础 |
| Core code / formal evidence release | tag `v1.0-serving-final` | final runtime 与合法 evidence Source of Truth |
| Legacy release commit | `e8b5bf1cd2b982632d763b93448250c1157efba1` | tag 对应 commit |
| Documentation refresh branch | `docs/final-documentation-refresh` | final 中文文档来源 |
| Documentation refresh commit | `4aef0fdc76bb62f1351b3ca97f87197a057e6776` | clean repository 文档事实基础 |
| Clean repository | `llm-serving-resource-control` | 无 legacy Git history 的最终系统 snapshot |

## Source Distinction

Documentation refresh commit 相对 `v1.0-serving-final` 只修改 Markdown 文档和历史文档归档；`nanovllm/`、benchmark code、tests、configs、traces 与 formal evidence 仍来自 release commit。Clean reconstruction 没有重新运行 benchmark，也没有更改 raw/aggregate evidence。

为适应全新 Git history 和可移植路径，clean repository 只对 packaging metadata、复现 CLI/path handling、repository validation 和中文文档做维护性调整；serving core 保持 legacy release 内容。

## Historical SHA Namespace

Formal artifacts 中的字段，例如：

- `code_sha` / `code_commit`；
- `runner_sha`；
- `trace_sha`；
- `config_sha`；
- package tree/module fingerprint；

全部保留原值。这些 SHA 指向 **legacy repository 的实验执行版本或 artifact 内容**，不应能在 clean repository 的单一初始 Git history 中用 `git show <sha>` 解析。

Clean repository commit 只证明 snapshot 组装与文档状态，不替代历史实验 code SHA。

## Evidence Migration Map

| Legacy evidence | Clean path | 状态 |
|---|---|---|
| `experiments/results/stage11_formal_r1_*` | `experiments/results/final/admission/stage11/` | VALID single Engine formal |
| `stage12_formal_r3*` | `experiments/results/final/admission/stage12/` | VALID boundary formal |
| `stage15_diagnostic_r3*` | `experiments/results/final/scheduler/stage15/` | VALID mechanism / negative serving boundary |
| `stage16_diagnostic_r2*` | `experiments/results/final/scheduler/stage16/` | VALID formal |
| `post_gpu_binding_fix/stage18/formal/` | `experiments/results/final/multi_replica/stage18/` | POST_FIX VALID formal |
| `post_gpu_binding_fix/p0_1/` | `experiments/results/final/characterization/p0_1/` | POST_FIX VALID characterization |
| `post_gpu_binding_fix/p0_2/` | `experiments/results/final/characterization/p0_2/` | POST_FIX VALID characterization |
| `post_gpu_binding_fix/stage19/` | `experiments/results/final/pd/stage19/` | POST_FIX VALID negative diagnostic |

## 未迁移的历史证据

以下内容只在 legacy repository 存在：

- pre-GPU-binding-fix Stage 18 performance；
- 由旧 Stage 18 选择的旧 P0 operating points；
- pre-fix Stage 19 performance；
- Stage 6–10 全量 matrices；
- Stage 14 smoke、timeout/local diagnostics；
- profiling `.nsys-rep`；
- handoff、task plan、progress 和历史 Stage protocol。

其中 pre-fix Multi-GPU 数据的 legacy 状态是：

```text
INVALID_PRE_GPU_BINDING_FIX
HISTORICAL_ONLY
DO_NOT_USE_FOR_FINAL_METRICS
```

不复制 invalid artifacts 是为了防止误用，不是删除历史；完整 contamination audit 仍由 legacy repository 保存。

## Integrity Manifests

- `docs/SOURCE_SNAPSHOT.sha256`：clean repository 中 runtime、benchmark、tests 和 reproduction source 的 SHA-256；
- `experiments/results/final/MANIFEST.sha256`：迁移 final evidence 的 SHA-256。

这些 manifest 验证当前文件完整性，不把 clean Git commit 冒充历史 experiment commit。
