package fastaimodel.streaming.compute;

import fastcore.FastCore;
import fastpointer.Pointer;

import java.lang.foreign.Arena;
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

    // Reusable thread-local off-heap native memory buffers for GEMV/GEMM downcalls
    private static final Arena GLOBAL_ARENA = Arena.ofAuto();
    private static final ThreadLocal<NativeBuffers> THREAD_BUFFERS = ThreadLocal.withInitial(NativeBuffers::new);

    private static final class NativeBuffers {
        private MemorySegment inBuf = GLOBAL_ARENA.allocate((long) 8192 * 4);
        private MemorySegment outBuf = GLOBAL_ARENA.allocate((long) 131072 * 4);

        MemorySegment ensureIn(int count) {
            long reqBytes = (long) count * 4;
            if (inBuf.byteSize() < reqBytes) {
                inBuf = GLOBAL_ARENA.allocate(Math.max(reqBytes, inBuf.byteSize() * 2));
            }
            return inBuf;
        }

        MemorySegment ensureOut(int count) {
            long reqBytes = (long) count * 4;
            if (outBuf.byteSize() < reqBytes) {
                outBuf = GLOBAL_ARENA.allocate(Math.max(reqBytes, outBuf.byteSize() * 2));
            }
            return outBuf;
        }
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
        NativeBuffers nb = THREAD_BUFFERS.get();
        MemorySegment inSegment = nb.ensureIn(inCols);
        MemorySegment outSegment = nb.ensureOut(outRows);

        MemorySegment.copy(vecIn, 0, inSegment, ValueLayout.JAVA_FLOAT, 0, inCols);
        MH_GEMV_Q4_0.invokeExact(outRows, inCols, weightSegment, inSegment, outSegment, rowBytes);
        MemorySegment.copy(outSegment, ValueLayout.JAVA_FLOAT, 0, vecOut, 0, outRows);
    }

    /**
     * Executes native AVX2 GEMV for Q8_0 weights from MemorySegment.
     */
    public static void gemvQ8_0(MemorySegment weightSegment,
                                float[] vecIn, float[] vecOut,
                                int outRows, int inCols, int rowBytes) throws Throwable {
        NativeBuffers nb = THREAD_BUFFERS.get();
        MemorySegment inSegment = nb.ensureIn(inCols);
        MemorySegment outSegment = nb.ensureOut(outRows);

        MemorySegment.copy(vecIn, 0, inSegment, ValueLayout.JAVA_FLOAT, 0, inCols);
        MH_GEMV_Q8_0.invokeExact(outRows, inCols, weightSegment, inSegment, outSegment, rowBytes);
        MemorySegment.copy(outSegment, ValueLayout.JAVA_FLOAT, 0, vecOut, 0, outRows);
    }

    /**
     * Executes native AVX2 GEMV for Q4_0 weights.
     */
    public static void gemvQ4_0(Pointer weightsPtr, long offset,
                                float[] vecIn, float[] vecOut,
                                int outRows, int inCols, int rowBytes) throws Throwable {
        long rawAddress = weightsPtr.address() + offset;
        long totalBytes = (long) outRows * rowBytes;
        MemorySegment weightSegment = FastCore.asMemorySegment(rawAddress, totalBytes);
        gemvQ4_0(weightSegment, vecIn, vecOut, outRows, inCols, rowBytes);
    }

    /**
     * Executes native AVX2 GEMV for Q8_0 weights.
     */
    public static void gemvQ8_0(Pointer weightsPtr, long offset,
                                float[] vecIn, float[] vecOut,
                                int outRows, int inCols, int rowBytes) throws Throwable {
        long rawAddress = weightsPtr.address() + offset;
        long totalBytes = (long) outRows * rowBytes;
        MemorySegment weightSegment = FastCore.asMemorySegment(rawAddress, totalBytes);
        gemvQ8_0(weightSegment, vecIn, vecOut, outRows, inCols, rowBytes);
    }

    /**
     * Executes native AVX2 GEMM for Q4_0 weights from MemorySegment.
     */
    public static void gemmQ4_0(MemorySegment weightSegment,
                                float[][] inBatch, float[][] outBatch,
                                int outRows, int inCols, int batchSize, int rowBytes) throws Throwable {
        int totalIn = batchSize * inCols;
        int totalOut = batchSize * outRows;

        NativeBuffers nb = THREAD_BUFFERS.get();
        MemorySegment inSegment = nb.ensureIn(totalIn);
        MemorySegment outSegment = nb.ensureOut(totalOut);

        // Copy batch into native contiguous buffer
        for (int b = 0; b < batchSize; b++) {
            MemorySegment.copy(inBatch[b], 0, inSegment, ValueLayout.JAVA_FLOAT, (long) b * inCols * 4, inCols);
        }

        MH_GEMM_Q4_0.invokeExact(outRows, inCols, weightSegment, inSegment, outSegment, batchSize, rowBytes);

        // Copy back to batch
        for (int b = 0; b < batchSize; b++) {
            MemorySegment.copy(outSegment, ValueLayout.JAVA_FLOAT, (long) b * outRows * 4, outBatch[b], 0, outRows);
        }
    }

    /**
     * Executes native AVX2 GEMM for Q8_0 weights from MemorySegment.
     */
    public static void gemmQ8_0(MemorySegment weightSegment,
                                float[][] inBatch, float[][] outBatch,
                                int outRows, int inCols, int batchSize, int rowBytes) throws Throwable {
        int totalIn = batchSize * inCols;
        int totalOut = batchSize * outRows;

        NativeBuffers nb = THREAD_BUFFERS.get();
        MemorySegment inSegment = nb.ensureIn(totalIn);
        MemorySegment outSegment = nb.ensureOut(totalOut);

        for (int b = 0; b < batchSize; b++) {
            MemorySegment.copy(inBatch[b], 0, inSegment, ValueLayout.JAVA_FLOAT, (long) b * inCols * 4, inCols);
        }

        MH_GEMM_Q8_0.invokeExact(outRows, inCols, weightSegment, inSegment, outSegment, batchSize, rowBytes);

        for (int b = 0; b < batchSize; b++) {
            MemorySegment.copy(outSegment, ValueLayout.JAVA_FLOAT, (long) b * outRows * 4, outBatch[b], 0, outRows);
        }
    }

    /**
     * Executes native AVX2 GEMM for Q4_0 weights across multiple batch tokens.
     */
    public static void gemmQ4_0(Pointer weightsPtr, long offset,
                                float[][] inBatch, float[][] outBatch,
                                int outRows, int inCols, int batchSize, int rowBytes) throws Throwable {
        long rawAddress = weightsPtr.address() + offset;
        long totalBytes = (long) outRows * rowBytes;
        MemorySegment weightSegment = FastCore.asMemorySegment(rawAddress, totalBytes);
        gemmQ4_0(weightSegment, inBatch, outBatch, outRows, inCols, batchSize, rowBytes);
    }

    /**
     * Executes native AVX2 GEMM for Q8_0 weights across multiple batch tokens.
     */
    public static void gemmQ8_0(Pointer weightsPtr, long offset,
                                float[][] inBatch, float[][] outBatch,
                                int outRows, int inCols, int batchSize, int rowBytes) throws Throwable {
        long rawAddress = weightsPtr.address() + offset;
        long totalBytes = (long) outRows * rowBytes;
        MemorySegment weightSegment = FastCore.asMemorySegment(rawAddress, totalBytes);
        gemmQ8_0(weightSegment, inBatch, outBatch, outRows, inCols, batchSize, rowBytes);
    }
}
