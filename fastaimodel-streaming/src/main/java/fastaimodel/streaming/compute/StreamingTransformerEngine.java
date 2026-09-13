package fastaimodel.streaming.compute;

import fastaimodel.streaming.buffer.PersistentKVCache;
import fastaimodel.streaming.io.GgufTensorIndexer;
import fastpointer.Pointer;

import java.io.File;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

import jdk.incubator.vector.*;

/**
 * High-performance streaming Transformer forward-pass engine.
 * Fused dequant-dot GEMV (Q4_0 / Q8_0 / F16 / F32) with ThreadLocal scratch,
 * direct FastPointer access and ForkJoin row dispatch.
 * Zero-GC hot path.
 */
public class StreamingTransformerEngine implements AutoCloseable {

    private static final int TOP_K = 40;
    private static final int MIN_ROWS_PER_TASK = 24;
    private static final int Q4_0_BLOCK = 32;
    private static final int Q8_0_BLOCK = 32;

    private static final VectorSpecies<Byte> BYTE_SPECIES = ByteVector.SPECIES_256;   // 32 bytes
    private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_256; // 8 floats
    private static final VectorSpecies<Integer> INT_SPECIES = IntVector.SPECIES_256;

    private final GgufTensorIndexer indexer;
    private final File modelFile;
    private final int dim;
    private final int nHeads;
    private final int nKvHeads;
    private final int headDim;
    private final int ffnDim;
    private final float rmsNormEps;
    private final float ropeFreqBase;

    private final RandomAccessFile raf;
    private final FileChannel channel;

    // Persistent workspace (allocated once)
    private final float[] normX;
    private final float[] residual;
    private final float[] q;
    private final float[] k;
    private final float[] v;
    private final float[] attnOut;
    private final float[] projAttn;
    private final float[] gate;
    private final float[] up;
    private final float[] ffnDown;
    private final float[] normWeightsScratch;

    private static final class HeadScratch {
        final float[] kPos;
        final float[] vPos;
        final float[] scores;
        HeadScratch(int headDim, int maxCtx) {
            this.kPos = new float[headDim];
            this.vPos = new float[headDim];
            this.scores = new float[maxCtx];
        }
    }
    private final ThreadLocal<HeadScratch> headScratch;
    private static final ThreadLocal<int[]> QS_SCRATCH = ThreadLocal.withInitial(() -> new int[16]);

    private final byte[] embdRowRaw;
    private final int[] topIndices;
    private final float[] topScores;
    private final float[] probsBuf;

    private final byte[] cachedOutputRaw;
    private final float[] cachedNormWeights;
    private final float[] logitsBuf;
    private final int outVocabSize;
    private final int outRowBytes;
    private final int outType;

    private final ForkJoinPool computePool;
    private final ThreadLocal<float[]> rowScratch; // only used as fallback

    public static final class BatchWorkspace {
        public final int capacity;
        public final float[][] resBatch;
        public final float[][] normXBatch;
        public final float[][] qBatch;
        public final float[][] kBatch;
        public final float[][] vBatch;
        public final float[][] attnOutBatch;
        public final float[][] projAttnBatch;
        public final float[][] gateBatch;
        public final float[][] upBatch;
        public final float[][] ffnDownBatch;

        public BatchWorkspace(int capacity, int dim, int nKvHeads, int headDim, int ffnDim) {
            this.capacity = capacity;
            this.resBatch = new float[capacity][dim];
            this.normXBatch = new float[capacity][dim];
            this.qBatch = new float[capacity][dim];
            this.kBatch = new float[capacity][nKvHeads * headDim];
            this.vBatch = new float[capacity][nKvHeads * headDim];
            this.attnOutBatch = new float[capacity][dim];
            this.projAttnBatch = new float[capacity][dim];
            this.gateBatch = new float[capacity][ffnDim];
            this.upBatch = new float[capacity][ffnDim];
            this.ffnDownBatch = new float[capacity][dim];
        }
    }

    private BatchWorkspace cachedWorkspace;

    public synchronized BatchWorkspace getOrCreateBatchWorkspace(int batchSize) {
        if (cachedWorkspace == null || cachedWorkspace.capacity < batchSize) {
            int cap = Math.max(batchSize, 64);
            cachedWorkspace = new BatchWorkspace(cap, dim, nKvHeads, headDim, ffnDim);
        }
        return cachedWorkspace;
    }

    public StreamingTransformerEngine(GgufTensorIndexer indexer) throws Exception {
        this.indexer = indexer;
        this.modelFile = indexer.getFile();
        this.dim = indexer.getEmbeddingLength();
        this.nHeads = indexer.getHeadCount();
        this.nKvHeads = indexer.getHeadCountKv();
        this.headDim = dim / Math.max(1, nHeads);
        this.ffnDim = indexer.getFeedForwardLength();
        this.rmsNormEps = indexer.getRmsNormEps();
        this.ropeFreqBase = indexer.getRopeFreqBase();

        this.raf = new RandomAccessFile(modelFile, "r");
        this.channel = raf.getChannel();

        this.normX = new float[dim];
        this.residual = new float[dim];
        this.q = new float[dim];
        this.k = new float[nKvHeads * headDim];
        this.v = new float[nKvHeads * headDim];
        this.attnOut = new float[dim];
        this.projAttn = new float[dim];
        this.gate = new float[ffnDim];
        this.up = new float[ffnDim];
        this.ffnDown = new float[dim];
        this.normWeightsScratch = new float[dim];
        final int maxContext = Math.max(2048, indexer.getContextLength());
        final int hDim = this.headDim;
        this.headScratch = ThreadLocal.withInitial(() -> new HeadScratch(hDim, maxContext));

        int maxRow = Math.max(dim, ffnDim);
        this.embdRowRaw = new byte[maxRow * 4];
        this.topIndices = new int[TOP_K];
        this.topScores = new float[TOP_K];
        this.probsBuf = new float[TOP_K];

        int parallelism = Math.max(1, Runtime.getRuntime().availableProcessors());
        this.computePool = new ForkJoinPool(parallelism);
        this.rowScratch = ThreadLocal.withInitial(() -> new float[maxRow]);

        // Cache output_norm
        GgufTensorIndexer.TensorEntry outNorm = indexer.getTensor("output_norm.weight");
        if (outNorm != null) {
            byte[] nb = readTensorBytesFull(outNorm);
            this.cachedNormWeights = new float[dim];
            GgufDequantizer.dequantize(outNorm.type(), nb, 0, cachedNormWeights, 0, dim);
        } else {
            this.cachedNormWeights = null;
        }

        // Cache output.weight (or token_embd as fallback)
        GgufTensorIndexer.TensorEntry outTensor = indexer.getTensor("output.weight");
        if (outTensor == null) {
            outTensor = indexer.getTensor("token_embd.weight");
        }
        if (outTensor != null && outTensor.shape().length >= 2) {
            this.outVocabSize = (int) outTensor.shape()[1];
            this.outRowBytes = (int) (outTensor.sizeBytes() / outVocabSize);
            this.outType = outTensor.type();
            this.cachedOutputRaw = new byte[(int) outTensor.sizeBytes()];
            channel.read(ByteBuffer.wrap(cachedOutputRaw), outTensor.fileOffset());
            this.logitsBuf = new float[outVocabSize];
        } else {
            this.outVocabSize = 0;
            this.outRowBytes = 0;
            this.outType = 0;
            this.cachedOutputRaw = null;
            this.logitsBuf = null;
        }
    }

    public void getEmbedding(int tokenId, float[] outVec) throws Exception {
        GgufTensorIndexer.TensorEntry embd = indexer.getTensor("token_embd.weight");
        if (embd == null || embd.shape().length < 2) {
            Arrays.fill(outVec, 0.0f);
            return;
        }
        int rowBytes = (int) (embd.sizeBytes() / embd.shape()[1]);
        long rowOffset = embd.fileOffset() + (long) tokenId * rowBytes;
        ByteBuffer buf = ByteBuffer.wrap(embdRowRaw, 0, rowBytes);
        buf.clear();
        channel.read(buf, rowOffset);
        GgufDequantizer.dequantize(embd.type(), embdRowRaw, 0, outVec, 0, dim);
    }

    public void computeLayer(int layerIdx, float[] x, Pointer chunkBase,
                             long chunkFileOffset, long chunkSpanBytes,
                             int tokenPos, PersistentKVCache kvCache) throws Exception {
        System.arraycopy(x, 0, residual, 0, dim);

        loadAndApplyRMSNorm("blk." + layerIdx + ".attn_norm.weight",
                x, normX, chunkBase, chunkFileOffset, chunkSpanBytes);

        gemv("blk." + layerIdx + ".attn_q.weight", normX, q, chunkBase, chunkFileOffset, chunkSpanBytes);
        gemv("blk." + layerIdx + ".attn_k.weight", normX, k, chunkBase, chunkFileOffset, chunkSpanBytes);
        gemv("blk." + layerIdx + ".attn_v.weight", normX, v, chunkBase, chunkFileOffset, chunkSpanBytes);

        applyRoPE(q, nHeads, tokenPos);
        applyRoPE(k, nKvHeads, tokenPos);

        kvCache.storeKV(layerIdx, tokenPos, k, v);
        computeAttention(layerIdx, q, attnOut, tokenPos, kvCache);

        gemv("blk." + layerIdx + ".attn_output.weight", attnOut, projAttn,
                chunkBase, chunkFileOffset, chunkSpanBytes);
        for (int i = 0; i < dim; i++) {
            x[i] = residual[i] + projAttn[i];
        }

        System.arraycopy(x, 0, residual, 0, dim);
        loadAndApplyRMSNorm("blk." + layerIdx + ".ffn_norm.weight",
                x, normX, chunkBase, chunkFileOffset, chunkSpanBytes);

        gemv("blk." + layerIdx + ".ffn_gate.weight", normX, gate, chunkBase, chunkFileOffset, chunkSpanBytes);
        gemv("blk." + layerIdx + ".ffn_up.weight",   normX, up,   chunkBase, chunkFileOffset, chunkSpanBytes);

        for (int i = 0; i < ffnDim; i++) {
            float g = gate[i];
            gate[i] = (g / (1.0f + (float) Math.exp(-g))) * up[i];
        }

        gemv("blk." + layerIdx + ".ffn_down.weight", gate, ffnDown, chunkBase, chunkFileOffset, chunkSpanBytes);
        for (int i = 0; i < dim; i++) {
            x[i] = residual[i] + ffnDown[i];
        }
    }

    private void computeAttention(int layerIdx, float[] qVec, float[] outVec,
                                  int pos, PersistentKVCache kvCache) {
        final float scale = (float) (1.0 / Math.sqrt(headDim));
        final int repFactor = Math.max(1, nHeads / nKvHeads);

        // Parallelize across all attention heads
        java.util.stream.IntStream.range(0, nHeads).parallel().forEach(h -> {
            HeadScratch scr = headScratch.get();
            float[] localK = scr.kPos;
            float[] localV = scr.vPos;
            float[] localScores = scr.scores;

            int kvH = h / repFactor;
            int qOff = h * headDim;

            float maxScore = -Float.MAX_VALUE;
            for (int t = 0; t <= pos; t++) {
                kvCache.getK(layerIdx, t, kvH, localK);
                float dot = 0.0f;
                for (int d = 0; d < headDim; d++) {
                    dot += qVec[qOff + d] * localK[d];
                }
                dot *= scale;
                localScores[t] = dot;
                if (dot > maxScore) maxScore = dot;
            }

            float sumExp = 0.0f;
            for (int t = 0; t <= pos; t++) {
                float e = (float) Math.exp(localScores[t] - maxScore);
                localScores[t] = e;
                sumExp += e;
            }
            float inv = sumExp > 0.0f ? 1.0f / sumExp : 0.0f;
            for (int t = 0; t <= pos; t++) {
                localScores[t] *= inv;
            }

            for (int d = 0; d < headDim; d++) {
                outVec[qOff + d] = 0.0f;
            }
            for (int t = 0; t <= pos; t++) {
                float w = localScores[t];
                if (w > 1e-6f) {
                    kvCache.getV(layerIdx, t, kvH, localV);
                    for (int d = 0; d < headDim; d++) {
                        outVec[qOff + d] += w * localV[d];
                    }
                }
            }
        });
    }

    public void computeLayerBatch(int layerIdx, float[][] xBatch, BatchWorkspace ws,
                                  Pointer chunkBase, long chunkFileOffset, long chunkSpanBytes,
                                  int startPos, int batchSize, PersistentKVCache kvCache) throws Exception {
        // 1. Attention Norm
        loadNormWeights("blk." + layerIdx + ".attn_norm.weight", normWeightsScratch,
                chunkBase, chunkFileOffset, chunkSpanBytes);
        for (int b = 0; b < batchSize; b++) {
            System.arraycopy(xBatch[b], 0, ws.resBatch[b], 0, dim);
            rmsNorm(xBatch[b], ws.normXBatch[b], normWeightsScratch);
        }

        // 2. Q, K, V Projections
        gemm("blk." + layerIdx + ".attn_q.weight", ws.normXBatch, ws.qBatch,
                chunkBase, chunkFileOffset, chunkSpanBytes, batchSize);
        gemm("blk." + layerIdx + ".attn_k.weight", ws.normXBatch, ws.kBatch,
                chunkBase, chunkFileOffset, chunkSpanBytes, batchSize);
        gemm("blk." + layerIdx + ".attn_v.weight", ws.normXBatch, ws.vBatch,
                chunkBase, chunkFileOffset, chunkSpanBytes, batchSize);

        // 3. RoPE & KV Cache Store
        for (int b = 0; b < batchSize; b++) {
            int pos = startPos + b;
            applyRoPE(ws.qBatch[b], nHeads, pos);
            applyRoPE(ws.kBatch[b], nKvHeads, pos);
            kvCache.storeKV(layerIdx, pos, ws.kBatch[b], ws.vBatch[b]);
        }

        // 4. Batched Causal Attention
        computeAttentionBatch(layerIdx, ws.qBatch, ws.attnOutBatch, startPos, batchSize, kvCache);

        // 5. Attn Output Projection & Residual
        gemm("blk." + layerIdx + ".attn_output.weight", ws.attnOutBatch, ws.projAttnBatch,
                chunkBase, chunkFileOffset, chunkSpanBytes, batchSize);
        for (int b = 0; b < batchSize; b++) {
            float[] x = xBatch[b];
            float[] res = ws.resBatch[b];
            float[] proj = ws.projAttnBatch[b];
            for (int i = 0; i < dim; i++) {
                x[i] = res[i] + proj[i];
            }
        }

        // 6. FFN Norm
        loadNormWeights("blk." + layerIdx + ".ffn_norm.weight", normWeightsScratch,
                chunkBase, chunkFileOffset, chunkSpanBytes);
        for (int b = 0; b < batchSize; b++) {
            System.arraycopy(xBatch[b], 0, ws.resBatch[b], 0, dim);
            rmsNorm(xBatch[b], ws.normXBatch[b], normWeightsScratch);
        }

        // 7. FFN Gate & Up
        gemm("blk." + layerIdx + ".ffn_gate.weight", ws.normXBatch, ws.gateBatch,
                chunkBase, chunkFileOffset, chunkSpanBytes, batchSize);
        gemm("blk." + layerIdx + ".ffn_up.weight",   ws.normXBatch, ws.upBatch,
                chunkBase, chunkFileOffset, chunkSpanBytes, batchSize);

        // 8. SwiGLU activation
        for (int b = 0; b < batchSize; b++) {
            float[] g = ws.gateBatch[b];
            float[] u = ws.upBatch[b];
            for (int i = 0; i < ffnDim; i++) {
                float val = g[i];
                g[i] = (val / (1.0f + (float) Math.exp(-val))) * u[i];
            }
        }

        // 9. FFN Down & Residual
        gemm("blk." + layerIdx + ".ffn_down.weight", ws.gateBatch, ws.ffnDownBatch,
                chunkBase, chunkFileOffset, chunkSpanBytes, batchSize);
        for (int b = 0; b < batchSize; b++) {
            float[] x = xBatch[b];
            float[] res = ws.resBatch[b];
            float[] down = ws.ffnDownBatch[b];
            for (int i = 0; i < dim; i++) {
                x[i] = res[i] + down[i];
            }
        }
    }

    private void computeAttentionBatch(int layerIdx, float[][] qBatch, float[][] outBatch,
                                       int startPos, int batchSize, PersistentKVCache kvCache) {
        final float scale = (float) (1.0 / Math.sqrt(headDim));
        final int repFactor = Math.max(1, nHeads / nKvHeads);

        java.util.stream.IntStream.range(0, nHeads).parallel().forEach(h -> {
            HeadScratch scr = headScratch.get();
            float[] localK = scr.kPos;
            float[] localV = scr.vPos;
            float[] localScores = scr.scores;

            int kvH = h / repFactor;
            int qOff = h * headDim;

            for (int b = 0; b < batchSize; b++) {
                int curPos = startPos + b;
                float[] qVec = qBatch[b];
                float[] outVec = outBatch[b];

                float maxScore = -Float.MAX_VALUE;
                for (int t = 0; t <= curPos; t++) {
                    kvCache.getK(layerIdx, t, kvH, localK);
                    float dot = 0.0f;
                    for (int d = 0; d < headDim; d++) {
                        dot += qVec[qOff + d] * localK[d];
                    }
                    dot *= scale;
                    localScores[t] = dot;
                    if (dot > maxScore) maxScore = dot;
                }

                float sumExp = 0.0f;
                for (int t = 0; t <= curPos; t++) {
                    float e = (float) Math.exp(localScores[t] - maxScore);
                    localScores[t] = e;
                    sumExp += e;
                }
                float inv = sumExp > 0.0f ? 1.0f / sumExp : 0.0f;
                for (int t = 0; t <= curPos; t++) {
                    localScores[t] *= inv;
                }

                for (int d = 0; d < headDim; d++) {
                    outVec[qOff + d] = 0.0f;
                }
                for (int t = 0; t <= curPos; t++) {
                    float w = localScores[t];
                    if (w > 1e-6f) {
                        kvCache.getV(layerIdx, t, kvH, localV);
                        for (int d = 0; d < headDim; d++) {
                            outVec[qOff + d] += w * localV[d];
                        }
                    }
                }
            }
        });
    }

    public int sampleNextToken(float[] x, float temperature, List<Integer> recentTokens) {
        if (cachedNormWeights != null) {
            rmsNorm(x, normX, cachedNormWeights);
        } else {
            System.arraycopy(x, 0, normX, 0, dim);
        }

        if (cachedOutputRaw == null || outVocabSize == 0) {
            return 3;
        }

        // Parallel fused projection into logitsBuf
        dispatchFusedGemv(outType, cachedOutputRaw, null, 0L, outRowBytes, dim,
                normX, logitsBuf, 0, outVocabSize);

        // Greedy mode (ArgMax) when temperature <= 0.05f
        if (temperature <= 0.05f) {
            float maxLogit = -Float.MAX_VALUE;
            int bestToken = 0;
            for (int t = 0; t < outVocabSize; t++) {
                float logit = logitsBuf[t];
                if (recentTokens != null && recentTokens.contains(t)) {
                    logit -= 0.5f; // lightweight penalty
                }
                if (logit > maxLogit) {
                    maxLogit = logit;
                    bestToken = t;
                }
            }
            return bestToken;
        }

        final float repetitionPenalty = 1.35f;
        Arrays.fill(topScores, -Float.MAX_VALUE);

        for (int t = 0; t < outVocabSize; t++) {
            float dot = logitsBuf[t];
            if (recentTokens != null) {
                int cnt = 0;
                for (int p : recentTokens) {
                    if (p == t) cnt++;
                }
                if (cnt > 0) {
                    if (dot > 0) dot /= (repetitionPenalty * cnt);
                    else         dot *= (repetitionPenalty * cnt);
                }
            }
            if (dot > topScores[TOP_K - 1]) {
                int pos = TOP_K - 1;
                while (pos > 0 && dot > topScores[pos - 1]) {
                    topScores[pos] = topScores[pos - 1];
                    topIndices[pos] = topIndices[pos - 1];
                    pos--;
                }
                topScores[pos] = dot;
                topIndices[pos] = t;
            }
        }

        float maxVal = topScores[0];
        float sum = 0.0f;
        float temp = Math.max(0.1f, temperature);
        for (int i = 0; i < TOP_K; i++) {
            if (topScores[i] <= -Float.MAX_VALUE / 2) {
                probsBuf[i] = 0.0f;
            } else {
                probsBuf[i] = (float) Math.exp((topScores[i] - maxVal) / temp);
                sum += probsBuf[i];
            }
        }
        if (sum <= 0.0f) {
            return topIndices[0] > 0 ? topIndices[0] : 3;
        }

        float r = (float) (Math.random() * sum);
        float cum = 0.0f;
        int selected = topIndices[0];
        for (int i = 0; i < TOP_K; i++) {
            cum += probsBuf[i];
            if (cum >= r) {
                selected = topIndices[i];
                break;
            }
        }
        return selected;
    }

    // ------------------------------------------------------------------
    // Core fused GEMV
    // ------------------------------------------------------------------

    private void gemv(String tensorName, float[] vecIn, float[] vecOut,
                      Pointer chunkBase, long chunkFileOffset, long chunkSpanBytes) throws Exception {
        GgufTensorIndexer.TensorEntry entry = indexer.getTensor(tensorName);
        if (entry == null) {
            Arrays.fill(vecOut, 0.0f);
            return;
        }

        int outRows = (int) entry.shape()[1];
        int inCols  = (int) entry.shape()[0];
        int rowBytes = (int) (entry.sizeBytes() / outRows);
        int type = entry.type();

        long relOffset = entry.fileOffset() - chunkFileOffset;
        boolean inChunk = chunkBase != null && !chunkBase.isNull()
                && relOffset >= 0
                && (relOffset + entry.sizeBytes() <= chunkSpanBytes);

        if (inChunk) {
            // Zero-copy path: work directly on the mmap pointer
            dispatchFusedGemv(type, null, chunkBase, relOffset, rowBytes, inCols,
                    vecIn, vecOut, 0, outRows);
        } else {
            // Fallback: load once into a temporary buffer (still only once per GEMV)
            byte[] tensorRaw = readTensorBytesFull(entry);
            dispatchFusedGemv(type, tensorRaw, null, 0L, rowBytes, inCols,
                    vecIn, vecOut, 0, outRows);
        }
    }

    private void dispatchFusedGemv(int type, byte[] heapData, Pointer ptr, long baseOffset,
                                   int rowBytes, int inCols,
                                   float[] vecIn, float[] vecOut, int rowStart, int rowEnd) {
        int count = rowEnd - rowStart;
        if (count <= 0) return;

        int cores = computePool.getParallelism();
        if (count <= 16 || cores <= 1) {
            computeFusedRowRange(type, heapData, ptr, baseOffset, rowBytes, inCols,
                    vecIn, vecOut, rowStart, rowEnd);
            return;
        }

        // Static core-based chunk partitioning without recursive task thrashing
        int tasks = Math.min(cores, Math.max(1, count / 16));
        int chunkSize = (count + tasks - 1) / tasks;

        java.util.List<java.util.concurrent.ForkJoinTask<?>> futures = new java.util.ArrayList<>(tasks);
        for (int i = 0; i < tasks; i++) {
            int start = rowStart + i * chunkSize;
            int end = Math.min(rowEnd, start + chunkSize);
            if (start < end) {
                futures.add(computePool.submit(() -> computeFusedRowRange(type, heapData, ptr, baseOffset, rowBytes, inCols,
                        vecIn, vecOut, start, end)));
            }
        }
        for (java.util.concurrent.ForkJoinTask<?> f : futures) {
            f.join();
        }
    }

    private void computeFusedRowRange(int type, byte[] heapData, Pointer ptr, long baseOffset,
                                      int rowBytes, int inCols,
                                      float[] vecIn, float[] vecOut, int rowStart, int rowEnd) {
        for (int r = rowStart; r < rowEnd; r++) {
            long rowOff = baseOffset + (long) r * rowBytes;
            switch (type) {
                case 2:  // Q4_0
                    vecOut[r] = fusedDotQ4_0(heapData, ptr, rowOff, vecIn, inCols);
                    break;
                case 8:  // Q8_0
                    vecOut[r] = fusedDotQ8_0(heapData, ptr, rowOff, vecIn, inCols);
                    break;
                case 1:  // F16
                    vecOut[r] = fusedDotF16(heapData, ptr, rowOff, vecIn, inCols);
                    break;
                case 0:  // F32
                    vecOut[r] = fusedDotF32(heapData, ptr, rowOff, vecIn, inCols);
                    break;
                default:
                    // Generic fallback (still uses thread-local scratch)
                    float[] rowBuf = rowScratch.get();
                    if (heapData != null) {
                        GgufDequantizer.dequantize(type, heapData, (int) rowOff, rowBuf, 0, inCols);
                    } else {
                        // slow path for unknown types from pointer
                        byte[] tmp = new byte[rowBytes];
                        for (int i = 0; i < rowBytes; i++) {
                            tmp[i] = ptr.getByte(rowOff + i);
                        }
                        GgufDequantizer.dequantize(type, tmp, 0, rowBuf, 0, inCols);
                    }
                    vecOut[r] = dotProductUnrolled(vecIn, rowBuf, inCols);
                    break;
            }
        }
    }

    private void gemm(String tensorName, float[][] inBatch, float[][] outBatch,
                      Pointer chunkBase, long chunkFileOffset, long chunkSpanBytes,
                      int batchSize) throws Exception {
        GgufTensorIndexer.TensorEntry entry = indexer.getTensor(tensorName);
        if (entry == null) {
            for (int b = 0; b < batchSize; b++) {
                Arrays.fill(outBatch[b], 0.0f);
            }
            return;
        }

        int outRows = (int) entry.shape()[1];
        int inCols  = (int) entry.shape()[0];
        int rowBytes = (int) (entry.sizeBytes() / outRows);
        int type = entry.type();

        long relOffset = entry.fileOffset() - chunkFileOffset;
        boolean inChunk = chunkBase != null && !chunkBase.isNull()
                && relOffset >= 0
                && (relOffset + entry.sizeBytes() <= chunkSpanBytes);

        if (inChunk) {
            dispatchFusedGemm(type, null, chunkBase, relOffset, rowBytes, inCols,
                    inBatch, outBatch, 0, outRows, batchSize);
        } else {
            byte[] tensorRaw = readTensorBytesFull(entry);
            dispatchFusedGemm(type, tensorRaw, null, 0L, rowBytes, inCols,
                    inBatch, outBatch, 0, outRows, batchSize);
        }
    }

    private void dispatchFusedGemm(int type, byte[] heapData, Pointer ptr, long baseOffset,
                                   int rowBytes, int inCols,
                                   float[][] inBatch, float[][] outBatch,
                                   int rowStart, int rowEnd, int batchSize) {
        int count = rowEnd - rowStart;
        if (count <= 0) return;

        int cores = computePool.getParallelism();
        if (count <= 16 || cores <= 1) {
            computeFusedRowRangeBatch(type, heapData, ptr, baseOffset, rowBytes, inCols,
                    inBatch, outBatch, rowStart, rowEnd, batchSize);
            return;
        }

        int tasks = Math.min(cores, Math.max(1, count / 16));
        int chunkSize = (count + tasks - 1) / tasks;

        java.util.List<java.util.concurrent.ForkJoinTask<?>> futures = new java.util.ArrayList<>(tasks);
        for (int i = 0; i < tasks; i++) {
            int start = rowStart + i * chunkSize;
            int end = Math.min(rowEnd, start + chunkSize);
            if (start < end) {
                futures.add(computePool.submit(() -> computeFusedRowRangeBatch(type, heapData, ptr,
                        baseOffset, rowBytes, inCols, inBatch, outBatch, start, end, batchSize)));
            }
        }
        for (java.util.concurrent.ForkJoinTask<?> f : futures) {
            f.join();
        }
    }

    private void computeFusedRowRangeBatch(int type, byte[] heapData, Pointer ptr, long baseOffset,
                                           int rowBytes, int inCols,
                                           float[][] inBatch, float[][] outBatch,
                                           int rowStart, int rowEnd, int batchSize) {
        int[] q4Scratch = new int[32];
        byte[] q8Scratch = new byte[32];

        for (int r = rowStart; r < rowEnd; r++) {
            long rowOff = baseOffset + (long) r * rowBytes;
            switch (type) {
                case 2:  // Q4_0
                    fusedDotQ4_0Batch(heapData, ptr, rowOff, inBatch, outBatch, r, inCols, batchSize, q4Scratch);
                    break;
                case 8:  // Q8_0
                    fusedDotQ8_0Batch(heapData, ptr, rowOff, inBatch, outBatch, r, inCols, batchSize, q8Scratch);
                    break;
                case 1:  // F16
                    for (int b = 0; b < batchSize; b++) {
                        outBatch[b][r] = fusedDotF16(heapData, ptr, rowOff, inBatch[b], inCols);
                    }
                    break;
                case 0:  // F32
                    for (int b = 0; b < batchSize; b++) {
                        outBatch[b][r] = fusedDotF32(heapData, ptr, rowOff, inBatch[b], inCols);
                    }
                    break;
                default:
                    float[] rowBuf = rowScratch.get();
                    if (heapData != null) {
                        GgufDequantizer.dequantize(type, heapData, (int) rowOff, rowBuf, 0, inCols);
                    } else {
                        byte[] tmp = new byte[rowBytes];
                        for (int i = 0; i < rowBytes; i++) {
                            tmp[i] = ptr.getByte(rowOff + i);
                        }
                        GgufDequantizer.dequantize(type, tmp, 0, rowBuf, 0, inCols);
                    }
                    for (int b = 0; b < batchSize; b++) {
                        outBatch[b][r] = dotProductUnrolled(inBatch[b], rowBuf, inCols);
                    }
                    break;
            }
        }
    }

    private static void fusedDotQ4_0Batch(byte[] heap, Pointer ptr, long offset,
                                          float[][] inBatch, float[][] outBatch, int r,
                                          int n, int batchSize, int[] q) {
        final int blocks = n / Q4_0_BLOCK;
        long off = offset;

        for (int b = 0; b < batchSize; b++) {
            outBatch[b][r] = 0.0f;
        }

        for (int b = 0; b < blocks; b++) {
            int scaleBits;
            if (heap != null) {
                scaleBits = (heap[(int) off] & 0xFF) | ((heap[(int) off + 1] & 0xFF) << 8);
            } else {
                scaleBits = (ptr.getByte(off) & 0xFF) | ((ptr.getByte(off + 1) & 0xFF) << 8);
            }
            float scale = halfToFloat((short) scaleBits);
            off += 2;

            if (heap != null) {
                for (int i = 0; i < 16; i++) {
                    int byteVal = heap[(int) off + i] & 0xFF;
                    q[i * 2]     = (byteVal & 0x0F) - 8;
                    q[i * 2 + 1] = (byteVal >>> 4) - 8;
                }
            } else {
                long p0 = ptr.getLong(off);
                long p1 = ptr.getLong(off + 8);
                for (int i = 0; i < 8; i++) {
                    int byteVal = (int) ((p0 >>> (i * 8)) & 0xFF);
                    q[i * 2]     = (byteVal & 0x0F) - 8;
                    q[i * 2 + 1] = (byteVal >>> 4) - 8;
                }
                for (int i = 0; i < 8; i++) {
                    int byteVal = (int) ((p1 >>> (i * 8)) & 0xFF);
                    q[16 + i * 2]     = (byteVal & 0x0F) - 8;
                    q[16 + i * 2 + 1] = (byteVal >>> 4) - 8;
                }
            }
            off += 16;

            final int xBase = b * Q4_0_BLOCK;

            for (int t = 0; t < batchSize; t++) {
                float[] x = inBatch[t];
                float blockSum = 0.0f;
                for (int i = 0; i < 32; i++) {
                    blockSum += q[i] * x[xBase + i];
                }
                outBatch[t][r] += scale * blockSum;
            }
        }
    }

    private static void fusedDotQ8_0Batch(byte[] heap, Pointer ptr, long offset,
                                          float[][] inBatch, float[][] outBatch, int r,
                                          int n, int batchSize, byte[] q) {
        final int blocks = n / Q8_0_BLOCK;
        long off = offset;

        for (int b = 0; b < batchSize; b++) {
            outBatch[b][r] = 0.0f;
        }

        for (int b = 0; b < blocks; b++) {
            int scaleBits = (heap != null)
                    ? ((heap[(int) off] & 0xFF) | ((heap[(int) off + 1] & 0xFF) << 8))
                    : ((ptr.getByte(off) & 0xFF) | ((ptr.getByte(off + 1) & 0xFF) << 8));
            float scale = halfToFloat((short) scaleBits);
            off += 2;

            if (heap != null) {
                System.arraycopy(heap, (int) off, q, 0, 32);
            } else {
                for (int g = 0; g < 4; g++) {
                    long packed = ptr.getLong(off + g * 8L);
                    for (int i = 0; i < 8; i++) {
                        q[g * 8 + i] = (byte) ((packed >>> (i * 8)) & 0xFF);
                    }
                }
            }
            off += 32;

            final int xBase = b * Q8_0_BLOCK;

            for (int t = 0; t < batchSize; t++) {
                float[] x = inBatch[t];
                float blockSum = 0.0f;
                for (int i = 0; i < 32; i++) {
                    blockSum += q[i] * x[xBase + i];
                }
                outBatch[t][r] += scale * blockSum;
            }
        }
    }

    // ------------------------------------------------------------------
    // Fused kernels – no intermediate FP32 row
    // ------------------------------------------------------------------

    /** Q4_0 – unrolled register FMA with zero ThreadLocal and zero horizontal stalls */
    private static float fusedDotQ4_0(byte[] heap, Pointer ptr, long offset,
                                      float[] x, int n) {
        float sum = 0.0f;
        final int blocks = n / Q4_0_BLOCK;
        long off = offset;

        for (int b = 0; b < blocks; b++) {
            int scaleBits;
            if (heap != null) {
                scaleBits = (heap[(int) off] & 0xFF) | ((heap[(int) off + 1] & 0xFF) << 8);
            } else {
                scaleBits = (ptr.getByte(off) & 0xFF) | ((ptr.getByte(off + 1) & 0xFF) << 8);
            }
            float scale = halfToFloat((short) scaleBits);
            off += 2;

            final int xBase = b * Q4_0_BLOCK;

            if (heap != null) {
                float blockSum = 0.0f;
                for (int i = 0; i < 16; i++) {
                    int byteVal = heap[(int) off + i] & 0xFF;
                    int q0 = (byteVal & 0x0F) - 8;
                    int q1 = (byteVal >>> 4) - 8;
                    blockSum += q0 * x[xBase + i * 2] + q1 * x[xBase + i * 2 + 1];
                }
                sum += scale * blockSum;
            } else {
                long p0 = ptr.getLong(off);
                long p1 = ptr.getLong(off + 8);
                float blockSum = 0.0f;
                for (int i = 0; i < 8; i++) {
                    int byteVal = (int) ((p0 >>> (i * 8)) & 0xFF);
                    int q0 = (byteVal & 0x0F) - 8;
                    int q1 = (byteVal >>> 4) - 8;
                    blockSum += q0 * x[xBase + i * 2] + q1 * x[xBase + i * 2 + 1];
                }
                for (int i = 0; i < 8; i++) {
                    int byteVal = (int) ((p1 >>> (i * 8)) & 0xFF);
                    int q0 = (byteVal & 0x0F) - 8;
                    int q1 = (byteVal >>> 4) - 8;
                    blockSum += q0 * x[xBase + 16 + i * 2] + q1 * x[xBase + 16 + i * 2 + 1];
                }
                sum += scale * blockSum;
            }
            off += 16;
        }
        return sum;
    }

    /** Q8_0 – vectorized version */
    private static float fusedDotQ8_0(byte[] heap, Pointer ptr, long offset,
                                      float[] x, int n) {
        float sum = 0.0f;
        final int blocks = n / Q8_0_BLOCK;
        long off = offset;

        for (int b = 0; b < blocks; b++) {
            int scaleBits = (heap != null)
                    ? ((heap[(int) off] & 0xFF) | ((heap[(int) off + 1] & 0xFF) << 8))
                    : ((ptr.getByte(off) & 0xFF) | ((ptr.getByte(off + 1) & 0xFF) << 8));
            float scale = halfToFloat((short) scaleBits);
            off += 2;

            int xBase = b * Q8_0_BLOCK;

            if (heap != null) {
                for (int i = 0; i < 32; i++) {
                    sum += scale * heap[(int) off + i] * x[xBase + i];
                }
            } else {
                // 4 × getLong → 32 signed bytes
                for (int g = 0; g < 4; g++) {
                    long packed = ptr.getLong(off + g * 8L);
                    int xOff = xBase + g * 8;

                    // Simple unrolled (can be further vectorized)
                    for (int i = 0; i < 8; i++) {
                        byte q = (byte) ((packed >>> (i * 8)) & 0xFF);
                        sum += scale * q * x[xOff + i];
                    }
                }
            }
            off += 32;
        }
        return sum;
    }

    private static float fusedDotF16(byte[] heap, Pointer ptr, long offset,
                                     float[] x, int n) {
        float sum = 0.0f;
        if (heap != null) {
            for (int i = 0; i < n; i++) {
                int bits = (heap[(int) (offset + i * 2)] & 0xFF)
                        | ((heap[(int) (offset + i * 2 + 1)] & 0xFF) << 8);
                sum += halfToFloat((short) bits) * x[i];
            }
            return sum;
        }

        int i = 0;
        int limit = n - 3;
        for (; i < limit; i += 4) {
            long packed = ptr.getLong(offset + i * 2L);
            for (int j = 0; j < 4; j++) {
                short bits = (short) ((packed >>> (j * 16)) & 0xFFFF);
                sum += halfToFloat(bits) * x[i + j];
            }
        }
        for (; i < n; i++) {
            short bits = ptr.getShort(offset + i * 2L);
            sum += halfToFloat(bits) * x[i];
        }
        return sum;
    }

    private static float fusedDotF32(byte[] heap, Pointer ptr, long offset,
                                     float[] x, int n) {
        float sum = 0.0f;
        if (heap != null) {
            ByteBuffer bb = ByteBuffer.wrap(heap).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < n; i++) {
                sum += bb.getFloat((int) (offset + i * 4)) * x[i];
            }
        } else {
            for (int i = 0; i < n; i++) {
                sum += ptr.getFloat(offset + i * 4L) * x[i];
            }
        }
        return sum;
    }

    private static float dotProductUnrolled(float[] a, float[] b, int n) {
        float s0 = 0f, s1 = 0f, s2 = 0f, s3 = 0f, s4 = 0f, s5 = 0f, s6 = 0f, s7 = 0f;
        int i = 0;
        int limit = n - 7;
        for (; i < limit; i += 8) {
            s0 += a[i]     * b[i];
            s1 += a[i + 1] * b[i + 1];
            s2 += a[i + 2] * b[i + 2];
            s3 += a[i + 3] * b[i + 3];
            s4 += a[i + 4] * b[i + 4];
            s5 += a[i + 5] * b[i + 5];
            s6 += a[i + 6] * b[i + 6];
            s7 += a[i + 7] * b[i + 7];
        }
        float sum = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;
        for (; i < n; i++) sum += a[i] * b[i];
        return sum;
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    private void loadNormWeights(String tensorName, float[] targetWeights,
                                 Pointer chunkBase, long chunkFileOffset, long chunkSpanBytes) throws Exception {
        GgufTensorIndexer.TensorEntry entry = indexer.getTensor(tensorName);
        if (entry == null) {
            Arrays.fill(targetWeights, 1.0f);
            return;
        }
        byte[] raw = readTensorBytes(entry, chunkBase, chunkFileOffset, chunkSpanBytes);
        GgufDequantizer.dequantize(entry.type(), raw, 0, targetWeights, 0, dim);
    }

    private void loadAndApplyRMSNorm(String tensorName, float[] xIn, float[] out,
                                     Pointer chunkBase, long chunkFileOffset, long chunkSpanBytes) throws Exception {
        loadNormWeights(tensorName, normWeightsScratch, chunkBase, chunkFileOffset, chunkSpanBytes);
        rmsNorm(xIn, out, normWeightsScratch);
    }

    private void rmsNorm(float[] xIn, float[] out, float[] w) {
        double sumSq = 0.0;
        for (float v : xIn) sumSq += (double) v * v;
        float inv = (float) (1.0 / Math.sqrt(sumSq / xIn.length + rmsNormEps));
        for (int i = 0; i < xIn.length; i++) {
            out[i] = xIn[i] * inv * w[i];
        }
    }

    private void applyRoPE(float[] vec, int heads, int pos) {
        for (int h = 0; h < heads; h++) {
            int offset = h * headDim;
            for (int i = 0; i < headDim / 2; i++) {
                double freq = 1.0 / Math.pow(ropeFreqBase, (2.0 * i) / headDim);
                double theta = pos * freq;
                float cos = (float) Math.cos(theta);
                float sin = (float) Math.sin(theta);
                int i0 = offset + 2 * i;
                int i1 = offset + 2 * i + 1;
                float v0 = vec[i0];
                float v1 = vec[i1];
                vec[i0] = v0 * cos - v1 * sin;
                vec[i1] = v0 * sin + v1 * cos;
            }
        }
    }

    private static float halfToFloat(short h) {
        int bits = h & 0xFFFF;
        int s = (bits >>> 15) & 0x1;
        int e = (bits >>> 10) & 0x1F;
        int f = bits & 0x3FF;
        if (e == 0) {
            if (f == 0) return Float.intBitsToFloat(s << 31);
            while ((f & 0x400) == 0) { f <<= 1; e--; }
            e++;
            f &= ~0x400;
        } else if (e == 31) {
            return Float.intBitsToFloat((s << 31) | 0x7F800000 | (f != 0 ? f << 13 : 0));
        }
        e = e + (127 - 15);
        f = f << 13;
        return Float.intBitsToFloat((s << 31) | (e << 23) | f);
    }

    private byte[] readTensorBytesFull(GgufTensorIndexer.TensorEntry entry) throws Exception {
        byte[] data = new byte[(int) entry.sizeBytes()];
        channel.read(ByteBuffer.wrap(data), entry.fileOffset());
        return data;
    }

    private byte[] readTensorBytes(GgufTensorIndexer.TensorEntry entry, Pointer chunkBase,
                                   long chunkFileOffset, long chunkSpanBytes) throws Exception {
        long rel = entry.fileOffset() - chunkFileOffset;
        byte[] data = new byte[(int) entry.sizeBytes()];
        boolean inChunk = chunkBase != null && !chunkBase.isNull()
                && rel >= 0 && (rel + entry.sizeBytes() <= chunkSpanBytes);
        if (inChunk) {
            for (int i = 0; i < data.length; i++) {
                data[i] = chunkBase.getByte(rel + i);
            }
        } else {
            channel.read(ByteBuffer.wrap(data), entry.fileOffset());
        }
        return data;
    }

    private final class FusedGemvTask extends RecursiveAction {
        private final int type;
        private final byte[] heapData;
        private final Pointer ptr;
        private final long baseOffset;
        private final int rowBytes;
        private final int inCols;
        private final float[] vecIn;
        private final float[] vecOut;
        private final int rowStart;
        private final int rowEnd;

        FusedGemvTask(int type, byte[] heapData, Pointer ptr, long baseOffset,
                      int rowBytes, int inCols, float[] vecIn, float[] vecOut,
                      int rowStart, int rowEnd) {
            this.type = type;
            this.heapData = heapData;
            this.ptr = ptr;
            this.baseOffset = baseOffset;
            this.rowBytes = rowBytes;
            this.inCols = inCols;
            this.vecIn = vecIn;
            this.vecOut = vecOut;
            this.rowStart = rowStart;
            this.rowEnd = rowEnd;
        }

        @Override
        protected void compute() {
            int count = rowEnd - rowStart;
            if (count <= MIN_ROWS_PER_TASK) {
                computeFusedRowRange(type, heapData, ptr, baseOffset, rowBytes, inCols,
                        vecIn, vecOut, rowStart, rowEnd);
                return;
            }
            int mid = rowStart + count / 2;
            invokeAll(
                    new FusedGemvTask(type, heapData, ptr, baseOffset, rowBytes, inCols, vecIn, vecOut, rowStart, mid),
                    new FusedGemvTask(type, heapData, ptr, baseOffset, rowBytes, inCols, vecIn, vecOut, mid, rowEnd)
            );
        }
    }

    @Override
    public void close() throws Exception {
        try {
            if (computePool != null) {
                computePool.shutdown();
            }
        } finally {
            if (channel != null && channel.isOpen()) channel.close();
            if (raf != null) raf.close();
        }
    }
}
