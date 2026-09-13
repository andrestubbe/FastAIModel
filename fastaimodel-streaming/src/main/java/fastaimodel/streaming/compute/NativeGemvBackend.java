package fastaimodel.streaming.compute;

import fastcore.FastCore;
import fastpointer.Pointer;

import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;

/**
 * <h1>NativeGemvBackend — AVX2/FMA Native Tensor Acceleration</h1>
 *
 * <p>Dispatches quantized GEMV and GEMM kernels directly to {@code fastai_streaming_kernels.dll}
 * via {@link FastCore#lookupFunction} without JNI overhead (~2–5 ns direct downcalls).</p>
 *
 * @author Andre Stubbe
 * @version 0.1.7
 * @since 0.1.7
 */
public final class NativeGemvBackend {

    private static final String LIB_NAME = "fastai_streaming_kernels";
    private static final boolean AVAILABLE;

    private static MethodHandle MH_GEMV_Q4_0;
    private static MethodHandle MH_GEMV_Q8_0;
    private static MethodHandle MH_GEMM_Q4_0;
    private static MethodHandle MH_GEMM_Q8_0;

    static {
        boolean loaded = false;
        try {
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

            MH_GEMV_Q4_0 = FastCore.lookupFunction(LIB_NAME, "gemv_q4_0_avx2", descGemv, NativeGemvBackend.class);
            MH_GEMV_Q8_0 = FastCore.lookupFunction(LIB_NAME, "gemv_q8_0_avx2", descGemv, NativeGemvBackend.class);
            MH_GEMM_Q4_0 = FastCore.lookupFunction(LIB_NAME, "gemm_q4_0_avx2", descGemm, NativeGemvBackend.class);
            MH_GEMM_Q8_0 = FastCore.lookupFunction(LIB_NAME, "gemm_q8_0_avx2", descGemm, NativeGemvBackend.class);

            loaded = true;
            System.out.println("[NativeGemvBackend] Successfully loaded native AVX2 streaming kernels via FastCore FFM");
        } catch (Throwable t) {
            System.err.println("[NativeGemvBackend] Warning: Failed to load native AVX2 streaming kernels via FastCore: " + t.getMessage());
            loaded = false;
        }
        AVAILABLE = loaded;
    }

    private NativeGemvBackend() {
        // Utility class
    }

    public static boolean isAvailable() {
        return AVAILABLE;
    }

    /**
     * Executes native AVX2 GEMV for Q4_0 weights from MemorySegment.
     */
    public static void gemvQ4_0(MemorySegment weightSegment,
                                float[] vecIn, float[] vecOut,
                                int outRows, int inCols, int rowBytes) throws Throwable {
        MemorySegment inSegment = MemorySegment.ofArray(vecIn);
        MemorySegment outSegment = MemorySegment.ofArray(vecOut);
        MH_GEMV_Q4_0.invokeExact(outRows, inCols, weightSegment, inSegment, outSegment, rowBytes);
    }

    /**
     * Executes native AVX2 GEMV for Q8_0 weights from MemorySegment.
     */
    public static void gemvQ8_0(MemorySegment weightSegment,
                                float[] vecIn, float[] vecOut,
                                int outRows, int inCols, int rowBytes) throws Throwable {
        MemorySegment inSegment = MemorySegment.ofArray(vecIn);
        MemorySegment outSegment = MemorySegment.ofArray(vecOut);
        MH_GEMV_Q8_0.invokeExact(outRows, inCols, weightSegment, inSegment, outSegment, rowBytes);
    }

    /**
     * Executes native AVX2 GEMV for Q4_0 weights.
     */
    public static void gemvQ4_0(Pointer weightsPtr, long offset,
                                float[] vecIn, float[] vecOut,
                                int outRows, int inCols, int rowBytes) throws Throwable {
        long rawAddress = weightsPtr.address() + offset;
        MemorySegment weightSegment = FastCore.asMemorySegment(rawAddress);
        gemvQ4_0(weightSegment, vecIn, vecOut, outRows, inCols, rowBytes);
    }

    /**
     * Executes native AVX2 GEMV for Q8_0 weights.
     */
    public static void gemvQ8_0(Pointer weightsPtr, long offset,
                                float[] vecIn, float[] vecOut,
                                int outRows, int inCols, int rowBytes) throws Throwable {
        long rawAddress = weightsPtr.address() + offset;
        MemorySegment weightSegment = FastCore.asMemorySegment(rawAddress);
        gemvQ8_0(weightSegment, vecIn, vecOut, outRows, inCols, rowBytes);
    }

    /**
     * Executes native AVX2 GEMM for Q4_0 weights from MemorySegment.
     */
    public static void gemmQ4_0(MemorySegment weightSegment,
                                float[][] inBatch, float[][] outBatch,
                                int outRows, int inCols, int batchSize, int rowBytes) throws Throwable {
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
     * Executes native AVX2 GEMM for Q8_0 weights from MemorySegment.
     */
    public static void gemmQ8_0(MemorySegment weightSegment,
                                float[][] inBatch, float[][] outBatch,
                                int outRows, int inCols, int batchSize, int rowBytes) throws Throwable {
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

    /**
     * Executes native AVX2 GEMM for Q4_0 weights across multiple batch tokens.
     */
    public static void gemmQ4_0(Pointer weightsPtr, long offset,
                                float[][] inBatch, float[][] outBatch,
                                int outRows, int inCols, int batchSize, int rowBytes) throws Throwable {
        long rawAddress = weightsPtr.address() + offset;
        MemorySegment weightSegment = FastCore.asMemorySegment(rawAddress);
        gemmQ4_0(weightSegment, inBatch, outBatch, outRows, inCols, batchSize, rowBytes);
    }

    /**
     * Executes native AVX2 GEMM for Q8_0 weights across multiple batch tokens.
     */
    public static void gemmQ8_0(Pointer weightsPtr, long offset,
                                float[][] inBatch, float[][] outBatch,
                                int outRows, int inCols, int batchSize, int rowBytes) throws Throwable {
        long rawAddress = weightsPtr.address() + offset;
        MemorySegment weightSegment = FastCore.asMemorySegment(rawAddress);
        gemmQ8_0(weightSegment, inBatch, outBatch, outRows, inCols, batchSize, rowBytes);
    }
}
