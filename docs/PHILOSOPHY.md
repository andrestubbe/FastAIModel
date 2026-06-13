# FastAIModel Design Philosophy

The core engineering principles behind `FastAIModel`.

---

## 1. Native-First, Zero Bloat
FastAIModel does not rely on local servers or REST endpoints. Everything runs directly inside the host process. This eliminates network layers, HTTP sockets, JSON serializations, and multithreading communication overhead, maximizing overall performance.

## 2. Zero-Copy Token Flow
Tokens flow directly from the native C++ generator into Java String buffers without temporary array copies. This keeps execution latency minimal and prevents high GC allocation rates during long generation phases.

## 3. Explicit Memory Lifecycles
Java’s garbage collector is excellent for object lifecycles but poor for multi-gigabyte native allocations. FastAIModel enforces explicit lifecycles through `AutoCloseable`, giving the developer complete control over model resource allocation and memory release.
