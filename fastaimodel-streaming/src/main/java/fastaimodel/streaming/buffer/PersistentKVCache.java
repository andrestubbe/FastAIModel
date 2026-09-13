package fastaimodel.streaming.buffer;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Resident Attention Key-Value Cache with structured multi-head layout:
 * [layer][tokenPos][kvHead][headDim] in 16-bit FP16 precision.
 */
public class PersistentKVCache implements AutoCloseable {

    private final int maxTokens;
    private final int nKvHeads;
    private final int headDim;
    private final int numLayers;

    // Off-heap direct buffers for K and V
    private final ByteBuffer kCache;
    private final ByteBuffer vCache;
    private int currentTokenIndex = 0;

    public PersistentKVCache(int maxTokens, int nKvHeads, int headDim, int numLayers) {
        this.maxTokens = maxTokens;
        this.nKvHeads = nKvHeads;
        this.headDim = headDim;
        this.numLayers = numLayers;

        // Size in bytes: layers * maxTokens * nKvHeads * headDim * 2 bytes (FP16)
        long kvElements = (long) numLayers * maxTokens * nKvHeads * headDim;
        long totalBytes = kvElements * 2L;

        int safeSize = (int) Math.min(Integer.MAX_VALUE - 1024, Math.max(1024 * 1024, totalBytes));
        this.kCache = ByteBuffer.allocateDirect(safeSize).order(ByteOrder.LITTLE_ENDIAN);
        this.vCache = ByteBuffer.allocateDirect(safeSize).order(ByteOrder.LITTLE_ENDIAN);
    }

    public int getCurrentTokenIndex() { return currentTokenIndex; }
    public void advanceToken() { if (currentTokenIndex < maxTokens - 1) currentTokenIndex++; }
    public void advanceTokens(int count) { currentTokenIndex = Math.min(maxTokens - 1, currentTokenIndex + count); }
    public void reset() { currentTokenIndex = 0; kCache.clear(); vCache.clear(); }
    public int getMaxTokens() { return maxTokens; }

    /**
     * Returns a MemorySegment slice pointing to the start of K cache for the given layer.
     * Byte offset = layer * maxTokens * nKvHeads * headDim * 2 bytes.
     */
    public java.lang.foreign.MemorySegment getKSegment(int layer) {
        long baseOffset = (long) layer * maxTokens * nKvHeads * headDim * 2L;
        long layerBytes = (long) maxTokens * nKvHeads * headDim * 2L;
        return java.lang.foreign.MemorySegment.ofBuffer(kCache).asSlice(baseOffset, layerBytes);
    }

    /**
     * Returns a MemorySegment slice pointing to the start of V cache for the given layer.
     * Byte offset = layer * maxTokens * nKvHeads * headDim * 2 bytes.
     */
    public java.lang.foreign.MemorySegment getVSegment(int layer) {
        long baseOffset = (long) layer * maxTokens * nKvHeads * headDim * 2L;
        long layerBytes = (long) maxTokens * nKvHeads * headDim * 2L;
        return java.lang.foreign.MemorySegment.ofBuffer(vCache).asSlice(baseOffset, layerBytes);
    }

    /**
     * Store current token K and V vectors for a given layer.
     */
    public void storeKV(int layer, int pos, float[] kVec, float[] vVec) {
        if (pos >= maxTokens) return;

        long baseByteOffset = (((long) layer * maxTokens + pos) * nKvHeads * headDim) * 2L;
        if (baseByteOffset + (long) nKvHeads * headDim * 2L > kCache.capacity()) return;

        int offset = (int) baseByteOffset;
        for (int i = 0; i < nKvHeads * headDim; i++) {
            kCache.putShort(offset + i * 2, floatToHalf(kVec[i]));
            vCache.putShort(offset + i * 2, floatToHalf(vVec[i]));
        }
    }

    /**
     * Read K vector slice for head kvH at position pos.
     */
    public void getK(int layer, int pos, int kvH, float[] outVec) {
        long byteOffset = ((((long) layer * maxTokens + pos) * nKvHeads + kvH) * headDim) * 2L;
        int offset = (int) byteOffset;
        for (int d = 0; d < headDim; d++) {
            outVec[d] = halfToFloat(kCache.getShort(offset + d * 2));
        }
    }

    /**
     * Read V vector slice for head kvH at position pos.
     */
    public void getV(int layer, int pos, int kvH, float[] outVec) {
        long byteOffset = ((((long) layer * maxTokens + pos) * nKvHeads + kvH) * headDim) * 2L;
        int offset = (int) byteOffset;
        for (int d = 0; d < headDim; d++) {
            outVec[d] = halfToFloat(vCache.getShort(offset + d * 2));
        }
    }

    public static short floatToHalf(float f) {
        int fbits = Float.floatToIntBits(f);
        int sign = (fbits >>> 16) & 0x8000;
        int val = (fbits & 0x7fffffff) + 0x1000;

        if (val >= 0x47800000) {
            if ((fbits & 0x7fffffff) >= 0x47800000) {
                if (val < 0x7f800000) return (short) (sign | 0x7c00);
                return (short) (sign | 0x7c00 | ((fbits & 0x007fffff) >>> 13));
            }
            return (short) (sign | 0x7bff);
        }
        if (val >= 0x38800000) return (short) (sign | ((val - 0x38000000) >>> 13));
        if (val < 0x33000000) return (short) sign;
        val = (fbits & 0x7fffffff) >>> 23;
        return (short) (sign | ((((fbits & 0x7fffff) | 0x800000) + (0x800000 >>> (val - 102))) >>> (126 - val)));
    }

    public static float halfToFloat(short h) {
        int bits = h & 0xFFFF;
        int s = (bits >>> 15) & 0x00000001;
        int e = (bits >>> 10) & 0x0000001F;
        int f = (bits) & 0x000003FF;

        if (e == 0) {
            if (f == 0) return Float.intBitsToFloat(s << 31);
            while ((f & 0x00000400) == 0) { f <<= 1; e -= 1; }
            e++;
            f &= ~0x00000400;
        } else if (e == 31) {
            return Float.intBitsToFloat((s << 31) | 0x7F800000 | (f != 0 ? (f << 13) : 0));
        }

        e = e + (127 - 15);
        f = f << 13;
        return Float.intBitsToFloat((s << 31) | (e << 23) | f);
    }

    @Override
    public void close() {
        kCache.clear();
        vCache.clear();
    }
}
