package fastaimodel.streaming.pipeline;

import fastaimodel.streaming.buffer.DoubleBufferRing;
import fastaimodel.streaming.io.GgufTensorIndexer.LayerChunk;
import fastaimodel.streaming.io.NativeChunkMmap;
import fastpointer.Pointer;
import fastsimd.SIMD;

import java.util.List;
import java.util.concurrent.*;

/**
 * Asynchronous Overlapped I/O and Compute Pipeline Scheduler.
 * Loads chunks using true physical file spans, enforcing strict memory boundaries.
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
        this.prefetchExecutor = Executors.newSingleThreadExecutor(r -> {
            Thread t = new Thread(r, "FastAI-Streaming-Prefetcher");
            t.setDaemon(true);
            return t;
        });
    }

    public interface ChunkComputeCallback {
        void compute(LayerChunk chunk, Pointer weightsPointer) throws Exception;
    }

    /**
     * Executes one complete forward pass across all chunks for a token generation step.
     */
    public void executeTokenStep(ChunkComputeCallback computeCallback) throws Exception {
        if (chunks.isEmpty()) return;

        // 1. Synchronously prime Chunk 0 into Slot A
        LayerChunk firstChunk = chunks.get(0);
        loadChunkIntoSlot(firstChunk, ring.getActiveComputeSlot());

        Future<?> prefetchFuture = null;

        for (int i = 0; i < chunks.size(); i++) {
            LayerChunk currentChunk = chunks.get(i);
            DoubleBufferRing.MemorySlot computeSlot = ring.getActiveComputeSlot();

            // Asynchronously prefetch Chunk i+1 into background prefetch slot
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

            // Compute current chunk via FastPointer
            computeCallback.compute(currentChunk, computeSlot.getPointer());

            // Await async prefetch completion
            if (prefetchFuture != null) {
                prefetchFuture.get();
                prefetchFuture = null;
            }

            // Swap slots: prefetch slot becomes active compute slot
            if (i + 1 < chunks.size()) {
                ring.swapSlots();
            }
        }
    }

    private void loadChunkIntoSlot(LayerChunk chunk, DoubleBufferRing.MemorySlot slot) throws Exception {
        slot.setState(DoubleBufferRing.SlotState.LOADING);

        long startOffset = chunk.physicalStartOffset();
        long spanBytes = chunk.physicalSpanBytes();

        Pointer srcPointer = mmap.mapChunkPointer(startOffset, spanBytes);
        Pointer dstPointer = slot.getPointer();

        long bytesToCopy = Math.min(slot.getCapacityBytes(), spanBytes);

        // FastSIMD 256-bit AVX2 hardware copy
        SIMD.copy(srcPointer, dstPointer, (int) bytesToCopy);

        // Release/unmap temporary file mapping to prevent virtual address exhaustion
        mmap.unmapPointer(srcPointer);

        slot.setCurrentChunkIndex(chunk.chunkIndex());
        slot.setState(DoubleBufferRing.SlotState.READY);
    }

    @Override
    public void close() {
        prefetchExecutor.shutdownNow();
    }
}
