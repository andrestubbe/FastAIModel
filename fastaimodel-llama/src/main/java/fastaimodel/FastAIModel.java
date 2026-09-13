package fastaimodel;

import java.io.File;

public class FastAIModel implements AutoCloseable {

    private static boolean isNativeLoaded = false;

    static {
        try {
            fastcore.LibraryLoader.load("ggml", FastAIModel.class);
            fastcore.LibraryLoader.load("llama-common", FastAIModel.class);
            fastcore.LibraryLoader.load("llama", FastAIModel.class);
            fastcore.LibraryLoader.load("fastaimodel", FastAIModel.class);
            isNativeLoaded = true;
        } catch (Throwable e) {
            String os = System.getProperty("os.name").toLowerCase();
            String ext = os.contains("win") ? ".dll" : (os.contains("mac") ? ".dylib" : ".so");
            String libDir = new File("lib").getAbsolutePath() + File.separator;
            
            if (new File(libDir + "fastaimodel" + ext).exists()) {
                try { System.load(libDir + "libomp140.x86_64" + ext); } catch (Throwable ignored) {}
                try { System.load(libDir + "ggml-base" + ext); } catch (Throwable ignored) {}
                try { System.load(libDir + "ggml" + ext); } catch (Throwable ignored) {}
                try { System.load(libDir + "ggml-cpu" + ext); } catch (Throwable ignored) {}
                try { System.load(libDir + "llama" + ext); } catch (Throwable ignored) {}
                try {
                    System.load(libDir + "fastaimodel" + ext);
                    isNativeLoaded = true;
                } catch (Throwable ignored) {}
            }
            if (!isNativeLoaded) {
                try {
                    System.loadLibrary("fastaimodel");
                    isNativeLoaded = true;
                } catch (Throwable ignored) {}
            }
        }
        if (isNativeLoaded) {
            try {
                nativeSetVerbose(false);
            } catch (Throwable ignored) {}
        }
    }

    private long handle;

    public interface TokenCallback {
        void onToken(String token);
    }

    public static FastAIModel open(String modelPath) {
        return new FastAIModel(modelPath);
    }

    public static Builder builder() {
        return new Builder();
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

    public static class Builder {
        private String modelPath;
        private int contextLength = 4096;
        private int gpuLayers = 0;

        public Builder model(String modelPath) {
            this.modelPath = modelPath;
            return this;
        }

        public Builder contextLength(int contextLength) {
            this.contextLength = contextLength;
            return this;
        }

        public Builder gpuLayers(int gpuLayers) {
            this.gpuLayers = gpuLayers;
            return this;
        }

        public FastAIModel build() {
            if (modelPath == null || modelPath.isBlank()) {
                throw new IllegalArgumentException("modelPath must not be null or blank");
            }
            return new FastAIModel(modelPath, contextLength, gpuLayers);
        }
    }

    public void predict(String prompt, int maxTokens, TokenCallback cb) {
        if (handle == 0) {
            throw new IllegalStateException("Model is not initialized or has been closed");
        }
        nativePredict(handle, prompt, maxTokens, 0.7f, 0.9f, cb);
    }

    public void predictFromMemoryAddress(long memoryAddress, int maxTokens, TokenCallback cb) {
        if (handle == 0) {
            throw new IllegalStateException("Model is not initialized or has been closed");
        }
        nativePredictFromMemoryAddress(handle, memoryAddress, maxTokens, 0.7f, 0.9f, cb);
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

    private static native void nativePredictFromMemoryAddress(
            long handle,
            long memoryAddress,
            int maxTokens,
            float temperature,
            float topP,
            TokenCallback callback
    );

    private static native void nativeFree(long handle);
}
