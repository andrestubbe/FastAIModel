package fastaimodel.streaming;

import fastaimodel.streaming.buffer.DoubleBufferRing;
import fastaimodel.streaming.buffer.PersistentKVCache;
import fastaimodel.streaming.compute.StreamingTransformerEngine;
import fastaimodel.streaming.io.GgufTensorIndexer;
import fastaimodel.streaming.io.NativeChunkMmap;
import fastaimodel.streaming.pipeline.ChunkPipelineScheduler;
import fastaimodel.streaming.tokenizer.GgufTokenizer;
import fastgpu.FastGPU;

import java.io.File;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;

/**
 * FastAIStreamingModel — Real layer-streaming transformer runtime.
 * Executes full layer forward-pass under fixed off-heap double-buffer budgets.
 * Complete with microsecond/millisecond step instrumentation.
 */
public class FastAIStreamingModel implements AutoCloseable {

    private final StreamingConfig config;
    private final GgufTensorIndexer indexer;
    private final GgufTokenizer tokenizer;
    private final NativeChunkMmap mmap;
    private final DoubleBufferRing bufferRing;
    private final PersistentKVCache kvCache;
    private final StreamingTransformerEngine computeEngine;
    private final ChunkPipelineScheduler scheduler;
    private final List<GgufTensorIndexer.LayerChunk> chunks;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    private FastGPU gpuContext;
    private boolean isGpuActive = false;

    public FastAIStreamingModel(File modelFile, StreamingConfig config) throws Exception {
        this.config = config;

        // 1. File verification
        if (modelFile == null || !modelFile.exists() || !modelFile.isFile()) {
            throw new IllegalArgumentException("Model file does not exist: " + (modelFile != null ? modelFile.getAbsolutePath() : "null"));
        }

        // 2. Parse GGUF Metadata and Index Tensors
        this.indexer = new GgufTensorIndexer(modelFile);
        this.indexer.parse();

        // 3. Load Pure-Java GGUF Tokenizer
        this.tokenizer = GgufTokenizer.loadFromGguf(modelFile);

        // 4. Memory Partitioning: Group layers into physical chunks
        long chunkBudgetBytes = (long) config.getChunkBudgetMB() * 1024L * 1024L;
        this.chunks = indexer.createChunks(chunkBudgetBytes);

        // Calculate max physical chunk size to avoid ring slot truncation
        long maxPhysicalChunkSize = chunkBudgetBytes;
        for (GgufTensorIndexer.LayerChunk c : chunks) {
            if (c.physicalSpanBytes() > maxPhysicalChunkSize) {
                maxPhysicalChunkSize = c.physicalSpanBytes();
            }
        }

        // 5. Native Win32 Memory-Mapped File Slicer
        this.mmap = new NativeChunkMmap(modelFile);

        // 6. Fixed-Footprint Double-Buffer Ring
        this.bufferRing = new DoubleBufferRing(maxPhysicalChunkSize);

        // 7. FP16 Persistent Resident KV-Cache
        int nLayers = indexer.getLayerCount();
        int maxTokens = Math.min(2048, indexer.getContextLength());
        int nKvHeads = indexer.getHeadCountKv();
        int headDim = indexer.getEmbeddingLength() / Math.max(1, indexer.getHeadCount());
        this.kvCache = new PersistentKVCache(maxTokens, nKvHeads, headDim, nLayers);

        // 8. Transformer Forward-Pass Engine
        this.computeEngine = new StreamingTransformerEngine(indexer);

        // Acceleration Backend: FastGPU fallback
        if (config.isUseGPU()) {
            try {
                this.gpuContext = FastGPU.openDefault();
                this.isGpuActive = true;
            } catch (Throwable t) {
                this.gpuContext = null;
                this.isGpuActive = false;
            }
        }

        // 9. Asynchronous Overlapped I/O Pipeline Scheduler
        this.scheduler = new ChunkPipelineScheduler(bufferRing, mmap, chunks);
    }

    /**
     * Stream response tokens via real layer-streaming inference.
     */
    public void stream(String prompt, Consumer<String> tokenCallback) throws Exception {
        stream(prompt, 64, tokenCallback);
    }

    /**
     * Stream response tokens with custom maxTokens limit and detailed millisecond timing for each step.
     */
    public void stream(String prompt, int maxTokens, Consumer<String> tokenCallback) throws Exception {
        if (closed.get()) throw new IllegalStateException("FastAIStreamingModel is closed");

        long t0 = System.nanoTime();
        List<Integer> promptTokens = tokenizer.encode(prompt);
        if (promptTokens.isEmpty()) {
            promptTokens.add(tokenizer.getBosTokenId());
        }
        long tTokenizeMs = (System.nanoTime() - t0) / 1_000_000;
        System.out.printf("[Timing] Tokenization: %d ms (%d prompt tokens: %s)%n", 
                tTokenizeMs, promptTokens.size(), promptTokens);

        int dim = indexer.getEmbeddingLength();
        float[] hiddenState = new float[dim];
        int currentPos = 0;

        // Prefill prompt tokens
        for (int i = 0; i < promptTokens.size(); i++) {
            int token = promptTokens.get(i);
            long tEmbd0 = System.nanoTime();
            computeEngine.getEmbedding(token, hiddenState);
            long tEmbdMs = (System.nanoTime() - tEmbd0) / 1_000_000;

            final int tokenPos = currentPos;
            long tLayer0 = System.nanoTime();
            scheduler.executeTokenStep((chunk, weightsPointer) -> {
                long chunkFileOffset = chunk.physicalStartOffset();
                long chunkSpanBytes = chunk.physicalSpanBytes();
                for (int l = chunk.startLayer(); l <= chunk.endLayer(); l++) {
                    computeEngine.computeLayer(l, hiddenState, weightsPointer, chunkFileOffset, chunkSpanBytes, tokenPos, kvCache);
                }
            });
            long tLayerMs = (System.nanoTime() - tLayer0) / 1_000_000;

            kvCache.advanceToken();
            currentPos++;

            System.out.printf("[Timing] Prefill Token %d/%d (id=%d): Embd=%d ms | Layers=%d ms | Total=%d ms%n",
                    i + 1, promptTokens.size(), token, tEmbdMs, tLayerMs, (tEmbdMs + tLayerMs));
        }

        // Autoregressive generation loop
        int generatedCount = 0;
        int nextToken = tokenizer.getBosTokenId();
        List<Integer> recentTokens = new java.util.ArrayList<>();

        long totalSampleTimeMs = 0;
        long totalEmbdTimeMs = 0;
        long totalLayersTimeMs = 0;

        while (generatedCount < maxTokens) {
            long tGenStep0 = System.nanoTime();

            // 1. Sample next token from logits with Top-K and Repetition Penalty
            long tSample0 = System.nanoTime();
            nextToken = computeEngine.sampleNextToken(hiddenState, 0.7f, recentTokens);
            long tSampleMs = (System.nanoTime() - tSample0) / 1_000_000;
            totalSampleTimeMs += tSampleMs;

            if (nextToken == tokenizer.getEosTokenId()) {
                break;
            }

            recentTokens.add(nextToken);
            if (recentTokens.size() > 64) recentTokens.remove(0);

            String piece = tokenizer.decode(nextToken);
            tokenCallback.accept(piece);
            generatedCount++;

            // 2. Forward step for sampled token
            long tStepEmbd0 = System.nanoTime();
            computeEngine.getEmbedding(nextToken, hiddenState);
            long tStepEmbdMs = (System.nanoTime() - tStepEmbd0) / 1_000_000;
            totalEmbdTimeMs += tStepEmbdMs;

            final int tokenPos = currentPos;
            long tStepLayers0 = System.nanoTime();
            scheduler.executeTokenStep((chunk, weightsPointer) -> {
                long chunkFileOffset = chunk.physicalStartOffset();
                long chunkSpanBytes = chunk.physicalSpanBytes();
                for (int l = chunk.startLayer(); l <= chunk.endLayer(); l++) {
                    computeEngine.computeLayer(l, hiddenState, weightsPointer, chunkFileOffset, chunkSpanBytes, tokenPos, kvCache);
                }
            });
            long tStepLayersMs = (System.nanoTime() - tStepLayers0) / 1_000_000;
            totalLayersTimeMs += tStepLayersMs;

            kvCache.advanceToken();
            currentPos++;
        }

        if (generatedCount > 0) {
            System.out.printf("%n--------------------------------------------------------------------------------%n");
            System.out.printf(" Inference Step Timings Summary (%d tokens generated):%n", generatedCount);
            System.out.printf("  • Avg Sampling / Logits:  %6.1f ms / token%n", (double) totalSampleTimeMs / generatedCount);
            System.out.printf("  • Avg Embedding Lookup:   %6.1f ms / token%n", (double) totalEmbdTimeMs / generatedCount);
            System.out.printf("  • Avg Layer Forward-Pass: %6.1f ms / token%n", (double) totalLayersTimeMs / generatedCount);
            System.out.printf("  • Total Throughput:       %6.2f tokens / sec%n",
                    (double) generatedCount / ((totalSampleTimeMs + totalEmbdTimeMs + totalLayersTimeMs) / 1000.0));
            System.out.printf("--------------------------------------------------------------------------------%n");
        }
    }

    public boolean isGpuActive() { return isGpuActive; }
    public int getChunkCount() { return chunks.size(); }
    public List<GgufTensorIndexer.LayerChunk> getChunks() { return chunks; }
    public GgufTensorIndexer getIndexer() { return indexer; }
    public GgufTokenizer getTokenizer() { return tokenizer; }
    public PersistentKVCache getKvCache() { return kvCache; }

    @Override
    public void close() throws Exception {
        if (closed.compareAndSet(false, true)) {
            if (scheduler != null) scheduler.close();
            if (bufferRing != null) bufferRing.close();
            if (mmap != null) mmap.close();
            if (computeEngine != null) computeEngine.close();
            if (kvCache != null) kvCache.close();
            if (gpuContext != null) {
                gpuContext.close();
                gpuContext = null;
            }
        }
    }
}
