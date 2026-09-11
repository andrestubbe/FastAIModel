# FastAIModel Version Changelog

## [0.1.7] — 2026-09-11

### Added
- **Zero-Copy Layer-wise Streaming Engine (`fastaimodel-streaming`)**:
  - Direct execution of 7B, 14B, and 70B parameter models within an ultra-low **1 GB – 2 GB RAM budget** (or any custom cap) (`-Xmx2g`).
  - Zero-Copy Win32 memory-mapping (`NativeChunkMmap`) with safe explicit `sun.misc.Unsafe` unmapping to prevent Windows file-lock errors.
  - Zero-Copy GGUF v2/v3 metadata & tensor offset indexer (`GgufTensorIndexer`).
  - Off-heap double-buffering ring (`DoubleBufferRing`) with 32-byte SIMD alignment and physical memory page locking (`Memory.lockPages()` / `VirtualLock`).
  - Overlapped I/O compute pipeline (`ChunkPipelineScheduler`) using Java 21 Virtual Threads and AVX2 SIMD memory streaming.
  - Hardware GPU layer execution via **FastGPU** (Vulkan Compute) with graceful fallback to **FastSIMD** CPU vector processing.
  - Automatic Ollama local blob model discovery (`ModelResolver`) across standard user directories.
  - Turnkey interactive live demo (`run-streaming.bat` / `StreamingDemo`).

### Fixed & Improved
- Added explicit `FastCore:0.1.0` dependency resolution in `fastaimodel-llama` and `fastaimodel-streaming` to ensure smooth JitPack multi-module compilation.
- Fixed UTF-8 console codepage (`chcp 65001`) in Windows launcher to cleanly render box-drawing characters and status checkmarks.

---

## [0.1.3] — 2026-08-14

### Added
- **Zero-Copy Shared Memory IPC**: Added `predictFromMemoryAddress` for direct memory-mapped prompt reading from FastSharedMemory pointers (< 800 ns transfer latency).
- **Vulkan GPU Acceleration**: Offload transformer layers (`n_gpu_layers`) to Intel Iris Xe, AMD Radeon, and NVIDIA RTX GPUs via FastGPU.
- **FlashAttention & Q4_0 KV-Cache**: Reduced memory bandwidth bottlenecks and boosted generation throughput to 51.9+ Tokens/s.
- **Apple Silicon Metal Support**: Added macOS native JNI loader and Metal GPU offloading for M1–M4 chips.
- **Direct Ollama Resolver**: Automatically resolves Ollama model names (e.g. `qwen2.5:1.5b`) directly to GGUF blobs in local cache.
