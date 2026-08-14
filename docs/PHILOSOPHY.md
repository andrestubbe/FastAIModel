# FastAIModel Engineering Philosophy

## Core Principles

1. **Zero-IPC Local Inference**  
   Eliminates local HTTP network overhead (Ollama HTTP/REST) by calling native C++ `llama.cpp` directly in-process via Java JNI.

2. **Vulkan & Metal GPU Offloading**  
   Leverages **FastGPU** to offload transformer layers directly to Intel Iris Xe, AMD Radeon, NVIDIA GeForce, and Apple Silicon Metal GPUs.

3. **Optimized Context & KV-Cache**  
   Uses FlashAttention fused kernels and Q4_0 KV-cache quantization to double memory bandwidth efficiency.
