# Decode Drain 与 Token Starvation

## Hypothesis

KV pressure 下，让高 release-efficiency 的 resident request 继续 Decode，可以更快自然完成、释放 KV，并避免 preemption/recompute。

## 为什么无界 Drain 会失败

若 drain 的唯一终点是 resident exhaustion，少量 resident request 会连续获得 execution；已产生 token 的 pending request 和 initial Prefill waiter 在数秒内没有 progress。总 work 和 elapsed 可能下降，但 ITL P99 与 waiting age 显著恶化。

这是一类典型多目标冲突：

```text
resource efficiency ↑
preemption/recompute ↓
system elapsed ↓

but

per-request token fairness ↓
ITL tail ↑
```

## Bounded Drain

可将 drain 从“直到完成”改为有限 episode：

- token/step budget；
- oldest waiting age bound；
- episode-entry TTFT/ITL slack watch；
- post-token progress-gap bound；
- bounded Prefill recovery window；
- exit reason observability。

这些 guard 使 Scheduler 在释放 KV 和恢复公平性之间周期切换。

## 测量原则

比较 Bounded Drain 时必须分 baseline：相对 unbounded baseline 看 starvation 是否修复；相对 pressure baseline 看 Recompute/Preemption 是否仍有收益。把两个 baseline 的最佳数字拼成一个比较会破坏因果口径。
