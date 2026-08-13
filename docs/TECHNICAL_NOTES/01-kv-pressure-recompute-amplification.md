# KV Pressure 与 Recompute Amplification

## 问题

KV Cache 满时，新 Prefill 和 resident Decode 都可能需要新 block。若系统通过 preempt active request 释放 KV，被抢占 request 会返回 WAITING；恢复时未被 Prefix Cache 命中的上下文必须重新 Prefill。

## 放大环

```text
KV full
→ allocate/append fails
→ preempt resident sequence
→ free physical blocks
→ sequence becomes pending recompute
→ re-Prefill logical context
→ consume KV and scheduler steps again
→ future allocation pressure increases
```

该环的成本不是 preemption count 能完全表达的。一个长 context victim 可能释放很少独占 blocks，却需要重算大量 token；一个接近完成的 request 若被允许继续 Decode，可能很快自然释放更多 KV。

## 需要观测的量

- resident KV blocks 与真正 `ref_count == 1` 的 releasable blocks；
- remaining decode tokens / estimated steps to release；
- Actual Recompute tokens，而不是仅使用估算；
- recompute tokens per preemption；
- Scheduler Steps、waiting age 和 post-token progress gap；
- terminal/final KV reconciliation。

## 控制方法

Recompute-aware victim selection 应统一所有 preemption path：Waiting Prefill allocation failure 与 Decode KV expansion failure不能分别使用不同 selector。Cost model 可以优先保留 recompute harm 大、即将自然完成的 request，并在必须抢占时选择单位释放收益下 harm 更低的 victim。

## 关键边界

减少 Recompute 不等于降低在线延迟。若策略通过无界推进少量 resident Decode 来释放 KV，pending request 仍会出现多秒 token gap。资源效率和 token fairness 必须分别测量。
