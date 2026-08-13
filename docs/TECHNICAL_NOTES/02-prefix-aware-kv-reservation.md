# Prefix-aware KV Reservation

## 为什么 Full Footprint 不够

Request-level Admission 常用 prompt + output 上界估算完整 KV footprint。这对没有 Prefix reuse 的保守控制合理；当活跃请求共享 Prefix 时，同一批 physical blocks 已由 `BlockManager` 持有，后续 request 不应再次 reservation。

## 三种 block 语义

1. **Active shared blocks**：仍被活跃 sequence 引用，生命周期内可安全共享；
2. **Inactive cached blocks**：ref count 为 0、可被 eviction，命中只是机会而非容量承诺；
3. **New private blocks**：当前 request 必须新增的物理资源。

因此可使用：

```text
incremental_required_blocks
= full_required_blocks
- active_shared_blocks
```

不能简单减去所有 cached prefix，否则 Admission 会把可 eviction cache 当成永久预留资源。

## 无副作用 Preview

Admission/Router 需要在接纳前读取 Prefix 状态，但不能提前改变 LRU、ref count 或 physical allocation。`preview_prefix_cache()` 应返回 matched/active/incremental 信息，并保证重复调用不改变 `BlockManager`。

## Reservation 生命周期

Reservation 在 admitted 后绑定 request，并在 finished/rejected/failed/cancelled terminal 统一释放。Formal run 最后同时对账 reservation 与 physical used blocks，避免“请求完成但 bookkeeping 泄漏”。

## 容量边界

Prefix-aware 只能消除重复 reservation，不能创造物理 KV。极低容量下，即使共享 Prefix，多个 request 的增量 blocks 仍可能超过 headroom；此时更激进 Admission 可能增加后续 contention。策略必须报告 capacity boundary。
