# FastAIModel Development Roadmap

## Q3 2026

- [x] Native Vulkan GPU Layer Offloading via FastGPU
- [x] FlashAttention & Q4_0 KV-Cache Quantization
- [x] Direct Ollama GGUF Blob Resolver
- [x] Apple Silicon Metal Support
- [x] Zero-Copy Layer-wise Streaming Engine (`fastaimodel-streaming`)
- [x] Native AVX2 & F16C Kernels for `Q4_0`, `Q8_0`, `Q4_K` GEMV/GEMM
- [x] Fused Native AVX2 + F16C Multi-Head Attention (`compute_attention_avx2`)
- [x] Dynamic Prompt Formatting (`[INST]` vs ChatML) and Safe 1.5GB Chunk Limits

## Q4 2026

- [ ] FastGPU Multi-GPU Layer-Sharded Offload (Multi-Adapter Pipeline)
- [ ] Vulkan External Host Memory Zero-Copy GEMV on integrated Iris Xe GPUs
- [ ] Batched Speculative Decoding with Small Draft Models
- [ ] GGUF Model Quantization Utility inside JVM
