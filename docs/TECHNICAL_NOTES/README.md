# 技术专题

这些文档从项目中的真实问题提炼可独立阅读的 LLM Serving 技术总结：

1. [KV Pressure 与 Recompute Amplification](01-kv-pressure-recompute-amplification.md)
2. [Prefix-aware KV Reservation](02-prefix-aware-kv-reservation.md)
3. [Decode Drain 与 Token Starvation](03-drain-starvation.md)
4. [Multi-Replica Routing、Saturation 与 Goodput](04-multi-replica-routing-saturation.md)
5. [GPU Binding 与 Runtime Provenance](05-gpu-binding-runtime-provenance.md)
6. [P/D KV Transfer 的性能边界](06-pd-transfer-boundary.md)
7. [LLM Serving Benchmark Correctness](07-benchmark-correctness.md)

专题中的数字只用于解释已审计机制；正式引用入口仍是 [`../FINAL_RESULTS.md`](../FINAL_RESULTS.md)。
