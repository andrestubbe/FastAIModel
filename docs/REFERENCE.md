# FastAIModel API Reference

Detailed technical specification and contracts for `FastAIModel` 0.1.2.

---

## Modules

### 1. `fastaimodel-onnx`

Lightweight module for ONNX model execution. Does **not** require or load any C++ Llama DLLs.

#### Class: `fastaimodel.FastAIOnnxModel`

```java
try (FastAIOnnxModel onnx = new FastAIOnnxModel("models/model.onnx")) {
    OrtSession.Result result = onnx.run(inputs);
}
```

* `public FastAIOnnxModel(String modelPath)`
* `public OrtSession getSession()`
* `public OrtEnvironment getEnv()`
* `public OrtSession.Result run(Map<String, OnnxTensor> inputs)`
* `public void close()`

---

### 2. `fastaimodel-llama`

In-process C++ inference engine wrapping `llama.cpp` for GGUF models.

#### Class: `fastaimodel.FastAIModel`

```java
try (FastAIModel model = new FastAIModel("models/qwen.gguf")) {
    model.predict("Prompt", 128, token -> System.out.print(token));
}
```

* `public FastAIModel(String modelPath)`
* `public FastAIModel(String modelPath, int ctxSize, int gpuLayers)`
* `public void predict(String prompt, int maxTokens, TokenCallback cb)`
* `public void close()`
