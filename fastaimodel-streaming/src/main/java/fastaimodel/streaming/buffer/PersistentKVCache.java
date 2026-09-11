package fastaimodel.streaming.buffer;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Resident Attention Key-Value Cache.
 * Remains persistently allocated in RAM (typically 100-300 MB) across layer streaming cycles.
 */
public class PersistentKVCache {

    private final int maxTokens;
    private final int hiddenDim;
    private final int numLayers;
    private final ByteBuffer cacheBuffer;
    private int currentTokenIndex = 0;

    public PersistentKVCache(int maxTokens, int hiddenDim, int numLayers) {
        this.maxTokens = maxTokens;
        this.hiddenDim = hiddenDim;
        this.numLayers = numLayers;

        // KV cache size = 2 (K + V) * maxTokens * hiddenDim * 2 bytes (FP16)
        long requiredBytes = 2L * maxTokens * hiddenDim * 2L * numLayers;
        int safeSize = (int) Math.min(Integer.MAX_VALUE - 1024, Math.max(1024 * 1024, requiredBytes));
        this.cacheBuffer = ByteBuffer.allocateDirect(safeSize).order(ByteOrder.LITTLE_ENDIAN);
    }

    public int getCurrentTokenIndex() { return currentTokenIndex; }
    public void advanceToken() { currentTokenIndex++; }
    public void reset() { currentTokenIndex = 0; cacheBuffer.clear(); }
    public ByteBuffer getCacheBuffer() { return cacheBuffer; }
    public int getMaxTokens() { return maxTokens; }
}
