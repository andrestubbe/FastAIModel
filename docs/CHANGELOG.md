# FastAIModel Version Changelog

## [0.1.3] — 2026-08-14

### Added
- **Vulkan GPU Acceleration**: Offload transformer layers (`n_gpu_layers`) to Intel Iris Xe, AMD Radeon, and NVIDIA RTX GPUs via FastGPU.
- **FlashAttention & Q4_0 KV-Cache**: Reduced memory bandwidth bottlenecks and boosted generation throughput to 51.9+ Tokens/s.
- **Apple Silicon Metal Support**: Added macOS native JNI loader and Metal GPU offloading for M1–M4 chips.
- **Direct Ollama Resolver**: Automatically resolves Ollama model names (e.g. `qwen2.5:1.5b`) directly to GGUF blobs in local cache.
