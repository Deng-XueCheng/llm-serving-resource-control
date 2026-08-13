# 复现指南

## 1. 复现边界

本仓库包含可执行代码、冻结 configs/traces 和 final evidence，但不会自动重新生成已有正式结果。运行命令时必须把输出写到 `experiments/results/local/` 或默认的非 final 路径；**禁止覆盖 `experiments/results/final/`**。

GPU benchmark 成本较高。优先执行 repository validation 和 CPU contract tests，再按需要执行 single Engine smoke、Multi-GPU provenance smoke 或 formal replay。

## 2. Source Lineage

- Legacy release tag：`v1.0-serving-final`
- Legacy release commit：`e8b5bf1cd2b982632d763b93448250c1157efba1`
- Documentation source commit：`4aef0fdc76bb62f1351b3ca97f87197a057e6776`
- Model：`Qwen/Qwen3-0.6B`
- Model revision：`c1899de289a04d12100db370d81485cdf75e47ca`

Formal evidence 中的历史 SHA 对应 legacy repository，不对应 clean repository 的 Git history。

## 3. Reference Environment

Single Engine final evidence 与 corrected Multi-GPU evidence来自不同固定执行环境；精确 package/hardware metadata保存在 raw artifacts。主要软件栈包括：

- Python 3.12 系列；
- PyTorch 2.8.0 + CUDA 12.8；
- Triton 3.4.0；
- FlashAttention 2.8.3；
- Transformers 与 model revision/hash 绑定。

结果不声称跨硬件/模型泛化。README 不突出硬件型号，但复核 absolute latency 时必须读取对应 raw artifact 的 runtime/hardware 字段。

## 4. 安装

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --no-build-isolation -e ".[test]"
```

Windows PowerShell 激活方式：

```powershell
.venv\Scripts\Activate.ps1
```

依赖快照见 `reproduction/requirements.lock.txt` 与 `reproduction/pylock.toml`。FlashAttention/CUDA 版本必须与本地 PyTorch/CUDA 兼容。

## 5. Repository 与 Evidence Validation

```bash
python scripts/validate_repository.py
```

该命令检查 Markdown links、final evidence path、SHA-256 manifest、invalid directory、secret-like files 和正式文档中的开发机绝对路径，不运行 GPU。

## 6. CPU / Non-GPU Contract Tests

```bash
python -m pytest -q \
  tests/test_scheduler_baseline.py \
  tests/test_prefix_aware_admission.py \
  tests/test_slack_aware_admission.py \
  tests/test_recompute_aware_scheduler.py \
  tests/test_bounded_recompute_scheduler.py \
  tests/test_mixed_token_budget_scheduler.py \
  tests/test_incremental_kv_allocation.py \
  tests/test_multi_replica_router.py \
  tests/test_pd_disaggregation.py \
  tests/test_lifecycle_metrics.py
```

这些测试验证 policy、resource invariant、lifecycle、routing 与 P/D metadata，不产生性能结论。

## 7. Single Engine Smoke

```bash
python reproduction/run_smoke.py \
  --config reproduction/configs/smoke_eager.json \
  --model-path /absolute/path/to/Qwen3-0.6B
```

CUDA Graph smoke：

```bash
python reproduction/run_smoke.py \
  --config reproduction/configs/smoke_cudagraph.json \
  --model-path /absolute/path/to/Qwen3-0.6B
```

输出位于 `reproduction/results/`，默认被 Git 忽略。

## 8. Stage 11 / 16 Replay

冻结 configs/traces 保存在：

- `experiments/configs/stage11_formal_r1*/`
- `experiments/data/stage11_formal_r1/`
- `experiments/configs/stage16_diagnostic_r2*/`

运行单个 cell：

```bash
python -m experiments.run_open_loop \
  --config experiments/configs/stage11_formal_r1/stage11_formal_r1_prefix_reuse90_kv8_seed1.json \
  --model-path /absolute/path/to/Qwen3-0.6B
```

Stage 16 将 config 替换为对应 `stage16_diagnostic_r2/*.json`。Frozen config 的 run ID 会把输出写到 `experiments/results/` 非 final 根目录，该目录被 Git 忽略；不要手工把新输出复制进 `final/`。

聚合器默认仍使用 legacy-style frozen manifest 路径；复核已迁移 evidence 时优先运行 repository validator，而不是原地写 aggregate。

## 9. Multi-Replica Corrected Smoke

要求至少两张可见 physical GPU：

```bash
python -m experiments.run_multi_replica \
  --config experiments/configs/stage18_smoke.json \
  --model-path /absolute/path/to/Qwen3-0.6B \
  --output experiments/results/local/stage18_smoke.json
```

结果必须包含不同 Torch/NVML physical UUID、model residency、positive CUDA work、必要 overlap、request distribution 与 final KV reconciliation。任一 gate 失败，runner fail closed。

Multi-Replica 与 P/D runner 会在创建任何 GPU worker 前验证 `docs/SOURCE_SNAPSHOT.sha256` 的完整覆盖，并核对冻结的 model repo/revision/file SHA-256；目录、源码或模型身份不匹配时直接退出。

## 10. Corrected Stage 18 Formal Replay

```bash
python -m experiments.run_post_fix_stage18_formal \
  --model-path /absolute/path/to/Qwen3-0.6B \
  --output-directory experiments/results/local/stage18_formal
```

独立 review：

```bash
python -m experiments.review_post_fix_stage18 \
  --directory experiments/results/local/stage18_formal
```

不要对 tracked `experiments/results/final/multi_replica/stage18/` 原地运行 reviewer，以免覆盖迁移的正式 `evidence_review.json`。

## 11. P0-1 / P0-2

```bash
python -m experiments.final_characterization p0-1 \
  --model-path /absolute/path/to/Qwen3-0.6B \
  --output experiments/results/local/p0_1

python -m experiments.final_characterization p0-2 \
  --model-path /absolute/path/to/Qwen3-0.6B \
  --reference-rps 4.8 \
  --output experiments/results/local/p0_2
```

Review：

```bash
python -m experiments.review_final_characterization \
  --directory experiments/results/local/p0_1 \
  --expected-cells 30 \
  --knee-rps 3.6

python -m experiments.review_final_characterization \
  --directory experiments/results/local/p0_2 \
  --expected-cells 24
```

## 12. Corrected Stage 19 P/D Diagnostic

```bash
python -m experiments.run_pd_disaggregated \
  --config experiments/configs/stage19_diagnostic.json \
  --model-path /absolute/path/to/Qwen3-0.6B \
  --output experiments/results/local/stage19_pd.json
```

该命令用于验证 1P1D 功能、physical GPU provenance、KV transfer/remapping 和资源释放；已有 evidence 显示性能为负，不建议自动扩展 formal matrix。

## 13. Evidence 复核

最终 evidence index 见 [`FINAL_RESULTS.md`](FINAL_RESULTS.md) 和 [`../experiments/results/final/README.md`](../experiments/results/final/README.md)。迁移后的 `MANIFEST.sha256` 验证文件字节完整性；其值不是新实验结果。

复核时至少检查：

1. baseline/candidate 和 matched denominator；
2. seed/load/workload coverage；
3. trace/config/code/model/runtime fingerprint；
4. terminal 与 final KV；
5. Multi-GPU physical provenance；
6. raw-to-summary 重算；
7. limitation 是否与 `FINAL_RESULTS.md` 一致。
