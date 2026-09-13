package fastaimodel.streaming.pipeline;

import fastaimodel.streaming.buffer.DoubleBufferRing;
import fastaimodel.streaming.io.GgufTensorIndexer.LayerChunk;
import fastaimodel.streaming.io.NativeChunkMmap;
import fastpointer.Pointer;

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Zero-copy chunk provider. Each layer chunk is memory-mapped ONCE for the
 * lifetime of the model and the resulting native pointer is cached — previous
 * versions re-mapped and unmapped the OS view on every single token step,
 * which is a syscall-heavy operation that dominated per-token latency.
 */
public class ChunkPipelineScheduler implements AutoCloseable {

    private final DoubleBufferRing ring; // retained for API compatibility; unused in cached-mmap mode
    private final NativeChunkMmap mmap;
    private final List<LayerChunk> chunks;
    private final ExecutorService prefetchExecutor;
    private final Pointer[] cachedPointers;

    public ChunkPipelineScheduler(DoubleBufferRing ring, NativeChunkMmap mmap, List<LayerChunk> chunks) {
        this.ring = ring;
        this.mmap = mmap;
        this.chunks = chunks;
        this.prefetchExecutor = Executors.newSingleThreadExecutor(r -> {
            Thread t = new Thread(r, "FastAI-Streaming-Prefetcher");
            t.setDaemon(true);
            return t;
        });
        this.cachedPointers = new Pointer[chunks.size()];
    }

    public interface ChunkComputeCallback {
        void compute(LayerChunk chunk, Pointer weightsPointer) throws Exception;
    }

    public synchronized void executeTokenStep(ChunkComputeCallback computeCallback) throws Exception {
        if (chunks.isEmpty()) return;

        for (int i = 0; i < chunks.size(); i++) {
            LayerChunk currentChunk = chunks.get(i);
            Pointer ptr = cachedPointers[i];
            if (ptr == null) {
                // First touch: map once, keep it mapped for all future tokens.
                ptr = mmap.mapChunkPointer(currentChunk.physicalStartOffset(), currentChunk.physicalSpanBytes());
                cachedPointers[i] = ptr;
            }
            computeCallback.compute(currentChunk, ptr);
        }
    }

    @Override
    public void close() {
        prefetchExecutor.shutdownNow();
        for (Pointer p : cachedPointers) {
            if (p != null) mmap.unmapPointer(p);
        }
    }
}
