# LLM Serving Benchmark Correctness

性能测试不只是打印 latency。一个可引用结果至少要通过三层正确性。

## Functional Correctness

- output token 与 state contract；
- request lifecycle 和 terminal denominator；
- block mapping/ref count；
- reservation 与 final KV release；
- P/D source/destination reconciliation。

## Experimental Correctness

- frozen config、trace、seed、model；
- baseline/candidate 只改变预定变量；
- runtime package/hardware identity；
- Multi-GPU physical UUID、model residency、CUDA work/overlap；
- artifact 不覆盖。

## Statistical / Evidence Correctness

- matched-pair 是比较单位；
- multi-seed 报告 pair direction；
- baseline=0 不计算相对百分比；
- 不平均不同 operating point 的百分比；
- summary 从 raw events 重算；
- negative result 和 regression 不隐藏。

## Fingerprint 不等于 Git HEAD

Dirty worktree 中 Git HEAD 可能不变，但执行代码已改变。因此还要绑定 runner、关键模块与 package tree hash。Clean repository 又引入新的 Git history，所以 formal artifact 的 legacy code SHA 必须与 snapshot commit 分开解释。

## Fail-closed

完整性、topology 或 denominator 任一不满足，就拒绝产生 final claim。Fail-closed 会减少“可用数字”，但能防止实验系统把配置错误、截断运行和不匹配 baseline 转化为漂亮但错误的结论。
