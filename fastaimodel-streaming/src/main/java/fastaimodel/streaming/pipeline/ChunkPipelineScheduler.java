package fastaimodel.streaming.pipeline;

import fastaimodel.streaming.buffer.DoubleBufferRing;
import fastaimodel.streaming.io.GgufTensorIndexer.LayerChunk;
import fastaimodel.streaming.io.NativeChunkMmap;

import java.nio.ByteBuffer;
import java.util.List;
import java.util.concurrent.*;

/**
 * Asynchronous Overlapped I/O and Compute Pipeline Scheduler.
 * Prefetches Chunk N+1 from disk via Virtual Threads while Chunk N is actively evaluated.
 */
public class ChunkPipelineScheduler implements AutoCloseable {

    private final DoubleBufferRing ring;
    private final NativeChunkMmap mmap;
    private final List<LayerChunk> chunks;
    private final ExecutorService prefetchExecutor;

    public ChunkPipelineScheduler(DoubleBufferRing ring, NativeChunkMmap mmap, List<LayerChunk> chunks) {
        this.ring = ring;
        this.mmap = mmap;
        this.chunks = chunks;
        // Use Java virtual threads or lightweight daemon threads
        this.prefetchExecutor = Executors.newSingleThreadExecutor(r -> {
            Thread t = new Thread(r, "FastAI-Streaming-Prefetcher");
            t.setDaemon(true);
            return t;
        });
    }

    public interface ChunkComputeCallback {
        void compute(LayerChunk chunk, ByteBuffer weightsBuffer) throws Exception;
    }

    /**
     * Executes one complete forward pass across all chunks for a token generation step.
     */
    public void executeTokenStep(ChunkComputeCallback computeCallback) throws Exception {
        if (chunks.isEmpty()) return;

        // 1. Prime the pump: synchronously load Chunk 0 into Slot A
        LayerChunk firstChunk = chunks.get(0);
        loadChunkIntoSlot(firstChunk, ring.getActiveComputeSlot());

        Future<?> prefetchFuture = null;

        for (int i = 0; i < chunks.size(); i++) {
            LayerChunk currentChunk = chunks.get(i);
            DoubleBufferRing.MemorySlot computeSlot = ring.getActiveComputeSlot();

            // Asynchronously prefetch Chunk i+1 into prefetch slot
            if (i + 1 < chunks.size()) {
                LayerChunk nextChunk = chunks.get(i + 1);
                DoubleBufferRing.MemorySlot prefetchSlot = ring.getPrefetchSlot();
                prefetchFuture = prefetchExecutor.submit(() -> {
                    try {
                        loadChunkIntoSlot(nextChunk, prefetchSlot);
                    } catch (Exception e) {
                        throw new RuntimeException("Async prefetch failed for chunk " + nextChunk.chunkIndex(), e);
                    }
                });
            }

            // Execute compute on currentChunk in parallel with background prefetch
            computeCallback.compute(currentChunk, computeSlot.getBuffer());

            // Wait for prefetch to finish before swapping
            if (prefetchFuture != null) {
                prefetchFuture.get();
                prefetchFuture = null;
            }

            // Swap slots: prefetch slot becomes active compute slot for next iteration
            if (i + 1 < chunks.size()) {
                ring.swapSlots();
            }
        }
    }

    private void loadChunkIntoSlot(LayerChunk chunk, DoubleBufferRing.MemorySlot slot) throws Exception {
        slot.setState(DoubleBufferRing.SlotState.LOADING);
        
        long startOffset = chunk.tensors().get(0).offset();
        long totalBytes = chunk.totalBytes();

        ByteBuffer slice = mmap.mapChunk(startOffset, totalBytes);
        ByteBuffer slotBuf = slot.getBuffer();
        slotBuf.clear();

        // Direct transfer into slot buffer
        int toTransfer = (int) Math.min(slotBuf.capacity(), slice.remaining());
        slice.limit(slice.position() + toTransfer);
        slotBuf.put(slice);
        slotBuf.flip();

        slot.setCurrentChunkIndex(chunk.chunkIndex());
        slot.setState(DoubleBufferRing.SlotState.READY);
    }

    @Override
    public void close() {
        prefetchExecutor.shutdownNow();
    }
}
