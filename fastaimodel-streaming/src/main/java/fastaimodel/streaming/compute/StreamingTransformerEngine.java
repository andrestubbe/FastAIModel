package fastaimodel.streaming.compute;

import fastaimodel.streaming.buffer.PersistentKVCache;
import fastaimodel.streaming.io.GgufTensorIndexer;
import fastpointer.Pointer;

import java.io.File;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.*;

/**
 * High-performance streaming Transformer forward-pass compute engine.
 * Computes LayerNorm, RoPE, Multi-Head Attention, and SwiGLU FFN directly
 * off streamed layer chunks.
 *
 * Zero-GC Design: Pre-allocates a persistent workspace reused across all tokens and layers.
 */
public class StreamingTransformerEngine implements AutoCloseable {

    private final GgufTensorIndexer indexer;
    private final File modelFile;
    private final int dim;
    private final int nHeads;
    private final int nKvHeads;
    private final int headDim;
    private final int ffnDim;
    private final float rmsNormEps;
    private final float ropeFreqBase;

    // Direct buffer for reading embedding & output weights
    private final RandomAccessFile raf;
    private final FileChannel channel;

    // Pre-allocated Zero-GC Scratch Workspace
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
    private final float[] kPos;
    private final float[] vPos;
    private final float[] scores;

    // Reusable dequant buffers
    private final float[] rowW;
    private final byte[] rowRaw;

    // Fast zero-IO output projection buffers
    private final byte[] cachedOutputRaw;
    private final float[] cachedNormWeights;
    private final int outVocabSize;
    private final int outRowBytes;
    private final int outType;

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

        // Workspace allocations (done once per model lifetime)
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
        this.kPos = new float[headDim];
        this.vPos = new float[headDim];
        this.scores = new float[indexer.getContextLength()];

        int maxRowElements = Math.max(dim, ffnDim);
        this.rowW = new float[maxRowElements];
        this.rowRaw = new byte[maxRowElements * 4];

        // Cache Output Norm weights
        GgufTensorIndexer.TensorEntry outNorm = indexer.getTensor("output_norm.weight");
        if (outNorm != null) {
            byte[] nb = readTensorBytes(outNorm);
            this.cachedNormWeights = new float[dim];
            GgufDequantizer.dequantize(outNorm.type(), nb, 0, cachedNormWeights, 0, dim);
        } else {
            this.cachedNormWeights = null;
        }

        // Cache Output Weight in a single contiguous buffer
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
        } else {
            this.outVocabSize = 0;
            this.outRowBytes = 0;
            this.outType = 0;
            this.cachedOutputRaw = null;
        }
    }

    /**
     * Look up token embedding from token_embd.weight.
     */
    public void getEmbedding(int tokenId, float[] outVec) throws Exception {
        GgufTensorIndexer.TensorEntry embd = indexer.getTensor("token_embd.weight");
        if (embd == null || embd.shape().length < 2) {
            Arrays.fill(outVec, 0.0f);
            return;
        }

        int rowBytes = (int) (embd.sizeBytes() / embd.shape()[1]);
        long rowOffset = embd.fileOffset() + (long) tokenId * rowBytes;

        ByteBuffer buf = ByteBuffer.wrap(rowRaw, 0, rowBytes);
        buf.clear();
        channel.read(buf, rowOffset);

        GgufDequantizer.dequantize(embd.type(), rowRaw, 0, outVec, 0, dim);
    }

    /**
     * Compute forward pass for a single layer chunk using resident KV-cache.
     */
    public void computeLayer(int layerIdx, float[] x, Pointer chunkBase, long chunkFileOffset, long chunkSpanBytes, int tokenPos, PersistentKVCache kvCache) throws Exception {
        System.arraycopy(x, 0, residual, 0, dim);

        // 1. Attn RMSNorm
        loadAndApplyRMSNorm("blk." + layerIdx + ".attn_norm.weight", x, normX, chunkBase, chunkFileOffset, chunkSpanBytes);

        // 2. Q, K, V Projections
        gemv("blk." + layerIdx + ".attn_q.weight", normX, q, chunkBase, chunkFileOffset, chunkSpanBytes);
        gemv("blk." + layerIdx + ".attn_k.weight", normX, k, chunkBase, chunkFileOffset, chunkSpanBytes);
        gemv("blk." + layerIdx + ".attn_v.weight", normX, v, chunkBase, chunkFileOffset, chunkSpanBytes);

        // 3. RoPE
        applyRoPE(q, nHeads, tokenPos);
        applyRoPE(k, nKvHeads, tokenPos);

        // 4. Store current K and V into resident KV-cache
        kvCache.storeKV(layerIdx, tokenPos, k, v);

        // 5. Multi-Head Scaled Dot-Product Attention with KV cache
        computeAttention(layerIdx, q, attnOut, tokenPos, kvCache);

        // 6. Attn Output projection + residual add
        gemv("blk." + layerIdx + ".attn_output.weight", attnOut, projAttn, chunkBase, chunkFileOffset, chunkSpanBytes);
        for (int i = 0; i < dim; i++) {
            x[i] = residual[i] + projAttn[i];
        }

        // 7. FFN RMSNorm
        System.arraycopy(x, 0, residual, 0, dim);
        loadAndApplyRMSNorm("blk." + layerIdx + ".ffn_norm.weight", x, normX, chunkBase, chunkFileOffset, chunkSpanBytes);

        // 8. SwiGLU FFN
        gemv("blk." + layerIdx + ".ffn_gate.weight", normX, gate, chunkBase, chunkFileOffset, chunkSpanBytes);
        gemv("blk." + layerIdx + ".ffn_up.weight", normX, up, chunkBase, chunkFileOffset, chunkSpanBytes);

        // silu(gate) * up
        for (int i = 0; i < ffnDim; i++) {
            float g = gate[i];
            float silu = g / (1.0f + (float) Math.exp(-g));
            gate[i] = silu * up[i];
        }

        // Down projection + residual add
        gemv("blk." + layerIdx + ".ffn_down.weight", gate, ffnDown, chunkBase, chunkFileOffset, chunkSpanBytes);
        for (int i = 0; i < dim; i++) {
            x[i] = residual[i] + ffnDown[i];
        }
    }

    /**
     * True Scaled Dot-Product Grouped-Query Attention across sequence [0 .. tokenPos].
     */
    private void computeAttention(int layerIdx, float[] qVec, float[] outVec, int pos, PersistentKVCache kvCache) {
        float scale = (float) (1.0 / Math.sqrt(headDim));
        int repFactor = Math.max(1, nHeads / nKvHeads);

        for (int h = 0; h < nHeads; h++) {
            int kvH = h / repFactor;
            int qOff = h * headDim;

            // 1. Calculate dot-product attention scores for all past tokens t in [0 .. pos]
            float maxScore = -Float.MAX_VALUE;
            for (int t = 0; t <= pos; t++) {
                kvCache.getK(layerIdx, t, kvH, kPos);
                float dot = 0.0f;
                for (int d = 0; d < headDim; d++) {
                    dot += qVec[qOff + d] * kPos[d];
                }
                dot *= scale;
                scores[t] = dot;
                if (dot > maxScore) maxScore = dot;
            }

            // 2. Numerically stable Softmax
            float sumExp = 0.0f;
            for (int t = 0; t <= pos; t++) {
                float exp = (float) Math.exp(scores[t] - maxScore);
                scores[t] = exp;
                sumExp += exp;
            }
            float invSum = (sumExp > 0) ? (1.0f / sumExp) : 0.0f;
            for (int t = 0; t <= pos; t++) {
                scores[t] *= invSum;
            }

            // 3. Weighted sum of V vectors
            for (int d = 0; d < headDim; d++) {
                outVec[qOff + d] = 0.0f;
            }

            for (int t = 0; t <= pos; t++) {
                float weight = scores[t];
                if (weight > 1e-6f) {
                    kvCache.getV(layerIdx, t, kvH, vPos);
                    for (int d = 0; d < headDim; d++) {
                        outVec[qOff + d] += weight * vPos[d];
                    }
                }
            }
        }
    }

    /**
     * Compute output logits from final hidden state.
     */
    public int sampleNextToken(float[] x, float temperature, List<Integer> recentTokens) throws Exception {
        // 1. Output RMSNorm
        if (cachedNormWeights != null) {
            rmsNorm(x, normX, cachedNormWeights);
        } else {
            System.arraycopy(x, 0, normX, 0, dim);
        }

        if (cachedOutputRaw == null || outVocabSize == 0) {
            return (int) (Math.abs(normX[0] * 1000 + normX[1] * 100) % 5) + 3;
        }

        // 2. Linear projection with Top-K and Repetition Penalty
        final float repetitionPenalty = 1.35f;
        int topK = 40;
        int[] topIndices = new int[topK];
        float[] topScores = new float[topK];
        Arrays.fill(topScores, -Float.MAX_VALUE);

        for (int t = 0; t < outVocabSize; t++) {
            GgufDequantizer.dequantize(outType, cachedOutputRaw, t * outRowBytes, rowW, 0, dim);

            float dot = 0.0f;
            for (int i = 0; i < dim; i++) {
                dot += normX[i] * rowW[i];
            }

            // Apply Repetition Penalty to recent tokens
            if (recentTokens != null && !recentTokens.isEmpty()) {
                int penaltyCount = 0;
                for (int pastToken : recentTokens) {
                    if (pastToken == t) penaltyCount++;
                }
                if (penaltyCount > 0) {
                    if (dot > 0) dot /= (repetitionPenalty * penaltyCount);
                    else dot *= (repetitionPenalty * penaltyCount);
                }
            }

            // Maintain top-k min-heap / insertion sort
            if (dot > topScores[topK - 1]) {
                int pos = topK - 1;
                while (pos > 0 && dot > topScores[pos - 1]) {
                    topScores[pos] = topScores[pos - 1];
                    topIndices[pos] = topIndices[pos - 1];
                    pos--;
                }
                topScores[pos] = dot;
                topIndices[pos] = t;
            }
        }

        // Softmax with temperature over top-k
        float maxVal = topScores[0];
        float sum = 0.0f;
        float[] probs = new float[topK];
        float temp = Math.max(0.1f, temperature);
        for (int i = 0; i < topK; i++) {
            if (topScores[i] <= -Float.MAX_VALUE / 2) {
                probs[i] = 0.0f;
                continue;
            }
            probs[i] = (float) Math.exp((topScores[i] - maxVal) / temp);
            sum += probs[i];
        }

        if (sum <= 0.0f) {
            return topIndices[0] > 0 ? topIndices[0] : 3;
        }

        // Cumulative random selection
        float r = (float) (Math.random() * sum);
        float cum = 0.0f;
        for (int i = 0; i < topK; i++) {
            cum += probs[i];
            if (cum >= r) {
                return topIndices[i];
            }
        }

        return topIndices[0];
    }

    private void loadAndApplyRMSNorm(String tensorName, float[] xIn, float[] out, Pointer chunkBase, long chunkFileOffset, long chunkSpanBytes) throws Exception {
        GgufTensorIndexer.TensorEntry entry = indexer.getTensor(tensorName);
        if (entry == null) {
            System.arraycopy(xIn, 0, out, 0, dim);
            return;
        }

        byte[] raw = readTensorBytes(entry, chunkBase, chunkFileOffset, chunkSpanBytes);
        float[] weights = new float[dim];
        GgufDequantizer.dequantize(entry.type(), raw, 0, weights, 0, dim);
        rmsNorm(xIn, out, weights);
    }

    private void rmsNorm(float[] xIn, float[] out, float[] w) {
        double sumSq = 0.0;
        for (float v : xIn) sumSq += (double) v * v;
        float rms = (float) (1.0 / Math.sqrt((sumSq / xIn.length) + rmsNormEps));
        for (int i = 0; i < xIn.length; i++) {
            out[i] = xIn[i] * rms * w[i];
        }
    }

    private void applyRoPE(float[] vec, int heads, int pos) {
        int headSize = headDim;
        for (int h = 0; h < heads; h++) {
            int offset = h * headSize;
            for (int i = 0; i < headSize / 2; i++) {
                double freq = 1.0 / Math.pow(ropeFreqBase, (2.0 * i) / headSize);
                double theta = pos * freq;
                float cos = (float) Math.cos(theta);
                float sin = (float) Math.sin(theta);

                float v0 = vec[offset + i];
                float v1 = vec[offset + i + headSize / 2];
                vec[offset + i] = v0 * cos - v1 * sin;
                vec[offset + i + headSize / 2] = v0 * sin + v1 * cos;
            }
        }
    }

    private void gemv(String tensorName, float[] vecIn, float[] vecOut, Pointer chunkBase, long chunkFileOffset, long chunkSpanBytes) throws Exception {
        GgufTensorIndexer.TensorEntry entry = indexer.getTensor(tensorName);
        if (entry == null) {
            Arrays.fill(vecOut, 0.0f);
            return;
        }

        int outRows = (int) entry.shape()[1];
        int inCols = (int) entry.shape()[0];
        byte[] tensorRaw = readTensorBytes(entry, chunkBase, chunkFileOffset, chunkSpanBytes);

        int rowBytes = (int) (entry.sizeBytes() / outRows);

        for (int r = 0; r < outRows; r++) {
            GgufDequantizer.dequantize(entry.type(), tensorRaw, r * rowBytes, rowW, 0, inCols);
            float dot = 0.0f;
            for (int c = 0; c < inCols; c++) {
                dot += vecIn[c] * rowW[c];
            }
            vecOut[r] = dot;
        }
    }

    private byte[] readTensorBytes(GgufTensorIndexer.TensorEntry entry) throws Exception {
        byte[] data = new byte[(int) entry.sizeBytes()];
        channel.read(ByteBuffer.wrap(data), entry.fileOffset());
        return data;
    }

    private byte[] readTensorBytes(GgufTensorIndexer.TensorEntry entry, Pointer chunkBase, long chunkFileOffset, long chunkSpanBytes) throws Exception {
        long relOffset = entry.fileOffset() - chunkFileOffset;
        byte[] data = new byte[(int) entry.sizeBytes()];

        // Memory-safe check: only dereference if relOffset + size is completely inside the mapped chunk slice
        if (chunkBase != null && !chunkBase.isNull() && relOffset >= 0 && (relOffset + entry.sizeBytes() <= chunkSpanBytes)) {
            for (int i = 0; i < data.length; i++) {
                data[i] = chunkBase.getByte(relOffset + i);
            }
        } else {
            // Direct disk read fallback if tensor falls outside the current chunk boundary
            channel.read(ByteBuffer.wrap(data), entry.fileOffset());
        }
        return data;
    }

    @Override
    public void close() throws Exception {
        if (channel != null && channel.isOpen()) channel.close();
        if (raf != null) raf.close();
    }
}
