# GPU Binding 与 Runtime Provenance

## 配置意图不等于运行事实

`gpu_id=1` 或 `CUDA_VISIBLE_DEVICES=1` 只说明 launcher 想如何绑定。Process spawn 模式、CUDA initialization timing、child environment 继承和 logical index remapping 都可能让实际设备与配置不一致。

一个典型失败是：parent/child 在环境完成隔离前初始化 CUDA，两个 worker 都看到或使用同一 physical GPU；日志仍显示不同 replica id，功能请求也能完成，但 Multi-GPU performance premise 已失效。

## 可信链路

```text
configured role/gpu
→ child PID
→ Torch logical device
→ Torch device UUID
→ NVML compute-app PID / physical UUID
→ model residency
→ positive CUDA event duration
→ cross-worker wall-clock overlap
→ request distribution
→ terminal and final KV reconciliation
```

Torch 与 NVML UUID 应交叉一致。Model residency 证明 worker 的显存实际落在目标设备；CUDA event 证明发生了计算；overlap 证明多个 worker 在要求并行的窗口实际重叠，而不是串行伪并发。

## Fail-closed 的必要性

缺少 UUID、CUDA work 为零、两个 worker UUID 相同、没有 required overlap 或 request 全落单 Replica，都应使 formal cell invalid。不能用“结果看起来合理”补偿 topology gate 失败。

## 一般化经验

Runtime provenance 应优先于 configuration intent。对于 CPU affinity、NUMA、NIC、RDMA、GPU/MIG 等系统实验，同样需要从进程身份映射到物理资源和实际 work，而不是只保存 launcher 参数。
