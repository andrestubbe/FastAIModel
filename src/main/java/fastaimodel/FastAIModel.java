package fastaimodel;

public class FastAIModel implements AutoCloseable {

    static {
        try {
            String userDir = System.getProperty("user.dir");
            String libDir = userDir + "\\lib\\";
            if (!new java.io.File(libDir + "llama.dll").exists()) {
                libDir = userDir + "\\build\\";
            }
            if (!new java.io.File(libDir + "llama.dll").exists()) {
                libDir = userDir + "\\";
            }
            
            String llamaPath = libDir + "llama.dll";
            String dllPath = libDir + "fastaimodel.dll";
            
            System.out.println("Loading: " + llamaPath);
            System.load(llamaPath);
            System.out.println("Loading: " + dllPath);
            System.load(dllPath);
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Warning: JNI loading failed: " + e.getMessage());
        }
    }

    private long handle;

    public interface TokenCallback {
        void onToken(String token);
    }

    public FastAIModel(String modelPath) {
        this(modelPath, 4096, 0);
    }

    public FastAIModel(String modelPath, int ctxSize, int gpuLayers) {
        this.handle = nativeInit(modelPath, ctxSize, gpuLayers);
        if (handle == 0) {
            throw new RuntimeException("Failed to load model: " + modelPath);
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
