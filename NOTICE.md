# Attribution

本项目基于 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 扩展。

Nano-vLLM upstream 提供轻量推理引擎基础，包括 Continuous Batching、Paged KV Cache、Prefix Cache、FlashAttention、CUDA Graph、Triton KV write、Tensor Parallel，以及基础 `Scheduler`、`BlockManager` 和 `ModelRunner`。原始代码采用 MIT License，版权声明见 [`LICENSE`](LICENSE)。

本仓库新增的工作集中在 LLM Serving resource control：Open-loop benchmark、request lifecycle、Admission Control、KV/Prefix-aware reservation、Recompute/Bounded Drain scheduling、Mixed Prefill/Decode token budget、Multi-Replica routing、GPU runtime provenance、1P1D 实验链路以及 fail-closed evidence pipeline。

本仓库不声称自研 Nano-vLLM upstream 已提供的 FlashAttention、PagedAttention、CUDA/Triton kernel 或完整推理引擎。
