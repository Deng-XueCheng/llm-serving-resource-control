# P/D KV Transfer 的性能边界

## 架构动机

Prefill 计算密集且 bursty，Decode 单步轻但 latency-sensitive。将二者放在不同 worker/GPU 可以隔离 phase contention，但会引入跨 worker state transfer。

## 完整功能链

1. Prefill worker materialize source KV；
2. destination 预分配 physical blocks；
3. 打包 logical mapping、sequence 和 RNG metadata；
4. KV 经 pinned-host staging 传输；
5. destination 安装 KV 并重映射 block table；
6. 恢复 Decode state 并继续生成；
7. source/destination 完成 terminal release。

只验证输出一致还不够；还需检查传输 bytes/latency、Prefill/Decode queue、physical GPU provenance 和两侧 final KV。

## 成本模型

```text
P/D net benefit
= saved phase interference
- Prefill queue delay
- serialization/materialization
- host staging copies
- destination allocation/install
- synchronization and state restore
```

当前实现不是 direct GPU P2P。Corrected diagnostic 中 Prefill queue 为秒级，KV transfer 还增加每请求数百毫秒，最终 `queue + transfer > isolation benefit`。

## Stop Condition

功能正确不等于性能方向成立。Single matched diagnostic 已清楚显示 TTFT/Goodput 退化时，继续扩大同类 formal matrix不会改变主要瓶颈；应先改变 transport 或 queue architecture，再启动新实验，而不是继续调 SLO threshold。
