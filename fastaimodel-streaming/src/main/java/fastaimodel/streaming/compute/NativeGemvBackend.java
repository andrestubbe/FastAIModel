package fastaimodel.streaming.compute;

import fastpointer.Pointer;

import java.io.File;
import java.io.InputStream;
import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * High-performance Foreign Function & Memory (FFM) backend for native AVX2/FMA matrix multiplications.
 * Dispatches directly to fastai_streaming_kernels.dll without JNI overhead.
 */
public final class NativeGemvBackend {

    private static final boolean AVAILABLE;
    private static MethodHandle MH_GEMV_Q4_0;
    private static MethodHandle MH_GEMV_Q8_0;
    private static MethodHandle MH_GEMM_Q4_0;
    private static MethodHandle MH_GEMM_Q8_0;

    static {
        boolean loaded = false;
        try {
            Linker linker = Linker.nativeLinker();
            Path dllPath = resolveAndExtractDll();
            if (dllPath != null && Files.exists(dllPath)) {
                Arena arena = Arena.ofAuto();
                SymbolLookup lookup = SymbolLookup.libraryLookup(dllPath, arena);

                // void gemv_q4_0_avx2(int outRows, int inCols, const void* weights, const float* vecIn, float* vecOut, int rowBytes)
                FunctionDescriptor descGemv = FunctionDescriptor.ofVoid(
                        ValueLayout.JAVA_INT,      // outRows
                        ValueLayout.JAVA_INT,      // inCols
                        ValueLayout.ADDRESS,       // weights pointer
                        ValueLayout.ADDRESS,       // vecIn
                        ValueLayout.ADDRESS,       // vecOut
                        ValueLayout.JAVA_INT       // rowBytes
                );

                // void gemm_q4_0_avx2(int outRows, int inCols, const void* weights, const float* inBatchFlat, float* outBatchFlat, int batchSize, int rowBytes)
                FunctionDescriptor descGemm = FunctionDescriptor.ofVoid(
                        ValueLayout.JAVA_INT,      // outRows
                        ValueLayout.JAVA_INT,      // inCols
                        ValueLayout.ADDRESS,       // weights pointer
                        ValueLayout.ADDRESS,       // inBatchFlat
                        ValueLayout.ADDRESS,       // outBatchFlat
                        ValueLayout.JAVA_INT,      // batchSize
                        ValueLayout.JAVA_INT       // rowBytes
                );

                var symQ4 = lookup.find("gemv_q4_0_avx2");
                var symQ8 = lookup.find("gemv_q8_0_avx2");
                var symGemmQ4 = lookup.find("gemm_q4_0_avx2");
                var symGemmQ8 = lookup.find("gemm_q8_0_avx2");

                if (symQ4.isPresent() && symQ8.isPresent() && symGemmQ4.isPresent() && symGemmQ8.isPresent()) {
                    MH_GEMV_Q4_0 = linker.downcallHandle(symQ4.get(), descGemv);
                    MH_GEMV_Q8_0 = linker.downcallHandle(symQ8.get(), descGemv);
                    MH_GEMM_Q4_0 = linker.downcallHandle(symGemmQ4.get(), descGemm);
                    MH_GEMM_Q8_0 = linker.downcallHandle(symGemmQ8.get(), descGemm);
                    loaded = true;
                    System.out.println("[NativeGemvBackend] Successfully loaded native AVX2 streaming kernels from " + dllPath);
                }
            }
        } catch (Throwable t) {
            System.err.println("[NativeGemvBackend] Warning: Failed to load native AVX2 streaming kernels: " + t.getMessage());
            loaded = false;
        }
        AVAILABLE = loaded;
    }

    private NativeGemvBackend() {}

    public static boolean isAvailable() {
        return AVAILABLE;
    }

    private static Path resolveAndExtractDll() {
        String os = System.getProperty("os.name", "").toLowerCase();
        if (!os.contains("win")) {
            return null; // Windows x64 AVX2 initially
        }
        String libName = "fastai_streaming_kernels.dll";

        // 1. Check current build target directories
        String[] candidatePaths = {
                "fastaimodel-streaming/src/main/resources/win32-x64/" + libName,
                "src/main/resources/win32-x64/" + libName,
                libName
        };
        for (String c : candidatePaths) {
            File f = new File(c);
            if (f.exists() && f.isFile()) {
                return f.toPath().toAbsolutePath();
            }
        }

        // 2. Extract from JAR resource if bundled
        try (InputStream in = NativeGemvBackend.class.getResourceAsStream("/win32-x64/" + libName)) {
            if (in != null) {
                Path tempDll = Files.createTempFile("fastai_streaming_kernels_", ".dll");
                tempDll.toFile().deleteOnExit();
                Files.copy(in, tempDll, StandardCopyOption.REPLACE_EXISTING);
                return tempDll;
            }
        } catch (Throwable ignored) {}

        return null;
    }

    /**
     * Executes native AVX2 GEMV for Q4_0 weights.
     */
    public static void gemvQ4_0(Pointer weightsPtr, long offset,
                                float[] vecIn, float[] vecOut,
                                int outRows, int inCols, int rowBytes) throws Throwable {
        long rawAddress = weightsPtr.address() + offset;
        MemorySegment weightSegment = MemorySegment.ofAddress(rawAddress);

        MemorySegment inSegment = MemorySegment.ofArray(vecIn);
        MemorySegment outSegment = MemorySegment.ofArray(vecOut);

        MH_GEMV_Q4_0.invokeExact(outRows, inCols, weightSegment, inSegment, outSegment, rowBytes);
    }

    /**
     * Executes native AVX2 GEMV for Q8_0 weights.
     */
    public static void gemvQ8_0(Pointer weightsPtr, long offset,
                                float[] vecIn, float[] vecOut,
                                int outRows, int inCols, int rowBytes) throws Throwable {
        long rawAddress = weightsPtr.address() + offset;
        MemorySegment weightSegment = MemorySegment.ofAddress(rawAddress);

        MemorySegment inSegment = MemorySegment.ofArray(vecIn);
        MemorySegment outSegment = MemorySegment.ofArray(vecOut);

        MH_GEMV_Q8_0.invokeExact(outRows, inCols, weightSegment, inSegment, outSegment, rowBytes);
    }

    /**
     * Executes native AVX2 GEMM for Q4_0 weights across multiple batch tokens.
     */
    public static void gemmQ4_0(Pointer weightsPtr, long offset,
                                float[][] inBatch, float[][] outBatch,
                                int outRows, int inCols, int batchSize, int rowBytes) throws Throwable {
        long rawAddress = weightsPtr.address() + offset;
        MemorySegment weightSegment = MemorySegment.ofAddress(rawAddress);

        // Flatten inBatch to contiguous float[]
        float[] inFlat = new float[batchSize * inCols];
        for (int b = 0; b < batchSize; b++) {
            System.arraycopy(inBatch[b], 0, inFlat, b * inCols, inCols);
        }
        float[] outFlat = new float[batchSize * outRows];

        MemorySegment inSegment = MemorySegment.ofArray(inFlat);
        MemorySegment outSegment = MemorySegment.ofArray(outFlat);

        MH_GEMM_Q4_0.invokeExact(outRows, inCols, weightSegment, inSegment, outSegment, batchSize, rowBytes);

        // Unflatten outFlat back into outBatch
        for (int b = 0; b < batchSize; b++) {
            System.arraycopy(outFlat, b * outRows, outBatch[b], 0, outRows);
        }
    }

    /**
     * Executes native AVX2 GEMM for Q8_0 weights across multiple batch tokens.
     */
    public static void gemmQ8_0(Pointer weightsPtr, long offset,
                                float[][] inBatch, float[][] outBatch,
                                int outRows, int inCols, int batchSize, int rowBytes) throws Throwable {
        long rawAddress = weightsPtr.address() + offset;
        MemorySegment weightSegment = MemorySegment.ofAddress(rawAddress);

        float[] inFlat = new float[batchSize * inCols];
        for (int b = 0; b < batchSize; b++) {
            System.arraycopy(inBatch[b], 0, inFlat, b * inCols, inCols);
        }
        float[] outFlat = new float[batchSize * outRows];

        MemorySegment inSegment = MemorySegment.ofArray(inFlat);
        MemorySegment outSegment = MemorySegment.ofArray(outFlat);

        MH_GEMM_Q8_0.invokeExact(outRows, inCols, weightSegment, inSegment, outSegment, batchSize, rowBytes);

        for (int b = 0; b < batchSize; b++) {
            System.arraycopy(outFlat, b * outRows, outBatch[b], 0, outRows);
        }
    }
}
