package fastaimodel;

import java.io.File;

public class FastAIModel implements AutoCloseable {

    static {
        try {
            fastcore.LibraryLoader.load("ggml", FastAIModel.class);
            fastcore.LibraryLoader.load("llama-common", FastAIModel.class);
            fastcore.LibraryLoader.load("llama", FastAIModel.class);
            fastcore.LibraryLoader.load("fastaimodel", FastAIModel.class);
        } catch (Throwable e) {
            // Fallback to local file load for development environment
            String libDir = new File("lib").getAbsolutePath() + File.separator;
            if (new File(libDir + "fastaimodel.dll").exists()) {
                try { System.load(libDir + "libomp140.x86_64.dll"); } catch (Throwable ignored) {}
                try { System.load(libDir + "ggml-base.dll"); } catch (Throwable ignored) {}
                try { System.load(libDir + "ggml.dll"); } catch (Throwable ignored) {}
                try { System.load(libDir + "ggml-cpu-x64.dll"); } catch (Throwable ignored) {}
                try { System.load(libDir + "llama.dll"); } catch (Throwable ignored) {}
                System.load(libDir + "fastaimodel.dll");
            } else {
                System.loadLibrary("fastaimodel");
            }
        }
        try {
            nativeSetVerbose(false);
        } catch (Throwable ignored) {}
    }

    private long handle;

    public interface TokenCallback {
        void onToken(String token);
    }

    public FastAIModel(String modelPath) {
        this(modelPath, 4096, 0);
    }

    public FastAIModel(String modelPath, int ctxSize, int gpuLayers) {
        String resolvedPath = OllamaModelResolver.resolve(modelPath);
        this.handle = nativeInit(resolvedPath, ctxSize, gpuLayers);
        if (handle == 0) {
            throw new RuntimeException("Failed to load model: " + resolvedPath);
        }
    }

    public void predict(String prompt, int maxTokens, TokenCallback cb) {
        if (handle == 0) {
            throw new IllegalStateException("Model is not initialized or has been closed");
        }
        nativePredict(handle, prompt, maxTokens, 0.7f, 0.9f, cb);
    }

    @Override
    public void close() {
        if (handle != 0) {
            nativeFree(handle);
            handle = 0;
        }
    }

    public static void setVerbose(boolean verbose) {
        try {
            nativeSetVerbose(verbose);
        } catch (Throwable ignored) {}
    }

    private static native void nativeSetVerbose(boolean verbose);

    private static native long nativeInit(String modelPath, int ctxSize, int gpuLayers);

    private static native void nativePredict(
            long handle,
            String prompt,
            int maxTokens,
            float temperature,
            float topP,
            TokenCallback callback
    );

    private static native void nativeFree(long handle);
}
