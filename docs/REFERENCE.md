# FastAIModel API Reference

Detailed technical specification and native contracts for `FastAIModel`.

---

## Classes & API Quick Reference

### `FastAIModel`

Core native model execution class. Implements `AutoCloseable` to safely dispose of native handles and prevent C++ memory leaks.

#### Constructors
* `public FastAIModel(String modelPath, int contextSize, float temperature)`
  * **modelPath**: Path to the local `.gguf` model file.
  * **contextSize**: Max context token limit (e.g. `2048`, `4096`).
  * **temperature**: Sampling temperature (e.g. `0.7f`).

#### Methods
* `public void generate(String prompt, Consumer<String> tokenCallback)`
  * Generates tokens for the given prompt, invoking the callback in real-time as each token is generated.
* `public void close()`
  * Disposes of the native C++ model context and frees loaded weights from memory.

---

## JNI Architecture

The C++ JNI implementation interfaces with `llama.cpp` using the following signature signatures:

```cpp
JNIEXPORT jlong JNICALL Java_fastaimodel_FastAIModel_nativeLoadModel
  (JNIEnv *, jclass, jstring modelPath, jint contextSize, jfloat temperature);

JNIEXPORT void JNICALL Java_fastaimodel_FastAIModel_nativeGenerate
  (JNIEnv *, jclass, jlong modelHandle, jstring prompt, jobject tokenCallback);

JNIEXPORT void JNICALL Java_fastaimodel_FastAIModel_nativeFreeModel
  (JNIEnv *, jclass, jlong modelHandle);
```

---

## Native Memory Management

Because models are loaded into C++ heap memory (outside of the JVM Garbage Collector), you must always use `try-with-resources` or manually invoke `.close()` to free model weights and release physical RAM.
