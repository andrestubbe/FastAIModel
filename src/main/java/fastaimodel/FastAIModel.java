package fastaimodel;

import java.util.function.Consumer;

public class FastAIModel implements AutoCloseable {

    static {
        try {
            System.loadLibrary("fastaimodel");
        } catch (UnsatisfiedLinkError e1) {
            try {
                String dllPath = System.getProperty("user.dir") + "\\build\\fastaimodel.dll";
                System.load(dllPath);
            } catch (UnsatisfiedLinkError e2) {
                // Fallback for compiler phase
            }
        }
    }

    private final long modelHandle;

    public FastAIModel(String modelPath, int contextSize, float temperature) {
        this.modelHandle = nativeLoadModel(modelPath, contextSize, temperature);
        if (this.modelHandle == 0) {
            throw new RuntimeException("Failed to load model: " + modelPath);
        }
    }

    public void generate(String prompt, Consumer<String> tokenCallback) {
        if (modelHandle == 0) throw new IllegalStateException("Model not loaded");
        nativeGenerate(modelHandle, prompt, tokenCallback);
    }

    @Override
    public void close() {
        if (modelHandle != 0) {
            nativeFreeModel(modelHandle);
        }
    }

    // === JNI Native Methods ===
    private static native long nativeLoadModel(String path, int contextSize, float temperature);
    private static native void nativeGenerate(long handle, String prompt, Consumer<String> callback);
    private static native void nativeFreeModel(long handle);
}
