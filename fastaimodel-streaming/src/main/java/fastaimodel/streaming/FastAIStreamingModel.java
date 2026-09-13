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

    public static FastAIStreamingModel open(String modelPath) throws Exception {
        return new FastAIStreamingModel(modelPath);
    }

    public static FastAIStreamingModel open(File modelFile) throws Exception {
        return new FastAIStreamingModel(modelFile);
    }

    public static Builder builder() {
        return new Builder();
    }

    public FastAIStreamingModel(String modelPath) throws Exception {
        this(modelPath, 512, false);
    }

    public FastAIStreamingModel(File modelFile) throws Exception {
        this(modelFile, StreamingConfig.builder().chunkBudgetMB(512).useGPU(false).build());
    }

    public FastAIStreamingModel(String modelPath, int chunkBudgetMB, boolean useGPU) throws Exception {
        this(fastaimodel.streaming.io.ModelResolver.resolve(modelPath),
                StreamingConfig.builder().chunkBudgetMB(chunkBudgetMB).useGPU(useGPU).build());
    }

    public FastAIStreamingModel(String modelPath, int chunkBudgetMB, boolean useGPU, int contextLength) throws Exception {
        this(fastaimodel.streaming.io.ModelResolver.resolve(modelPath),
                StreamingConfig.builder().chunkBudgetMB(chunkBudgetMB).useGPU(useGPU).contextLength(contextLength).build());
    }

    public FastAIStreamingModel(File modelFile, int chunkBudgetMB, boolean useGPU) throws Exception {
        this(modelFile, StreamingConfig.builder().chunkBudgetMB(chunkBudgetMB).useGPU(useGPU).build());
    }

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

        // 8. Acceleration Backend: FastGPU fallback
        if (config.isUseGPU()) {
            try {
                this.gpuContext = FastGPU.openDefault();
                this.isGpuActive = true;
            } catch (Throwable t) {
                this.gpuContext = null;
                this.isGpuActive = false;
            }
        }

        // 9. Transformer Forward-Pass Engine
        this.computeEngine = new StreamingTransformerEngine(indexer, gpuContext);

        // 10. Asynchronous Overlapped I/O Pipeline Scheduler
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

        String formattedPrompt = formatPrompt(prompt);

        List<Integer> promptTokens = tokenizer.encode(formattedPrompt, false);
        if (promptTokens.isEmpty()) {
            promptTokens.add(tokenizer.getBosTokenId());
        }
        long tTokenizeMs = (System.nanoTime() - t0) / 1_000_000;
        if (config.isVerbose()) {
            System.out.printf("[Timing] Tokenization: %d ms (%d prompt tokens: %s)%n", 
                    tTokenizeMs, promptTokens.size(), promptTokens);
        }

        int dim = indexer.getEmbeddingLength();
        float[] hiddenState = new float[dim];
        int currentPos = 0;

        // Batched Prefill for all prompt tokens in a single layer-streaming pass
        int numPromptTokens = promptTokens.size();
        float[][] promptXBatch = new float[numPromptTokens][dim];
        long tEmbd0 = System.nanoTime();
        for (int i = 0; i < numPromptTokens; i++) {
            computeEngine.getEmbedding(promptTokens.get(i), promptXBatch[i]);
        }
        long tEmbdMs = (System.nanoTime() - tEmbd0) / 1_000_000;

        StreamingTransformerEngine.BatchWorkspace ws = computeEngine.getOrCreateBatchWorkspace(numPromptTokens);

        long tBatchLayers0 = System.nanoTime();
        scheduler.executeTokenStep((chunk, weightsPointer) -> {
            long chunkFileOffset = chunk.physicalStartOffset();
            long chunkSpanBytes = chunk.physicalSpanBytes();
            for (int l = chunk.startLayer(); l <= chunk.endLayer(); l++) {
                computeEngine.computeLayerBatch(l, promptXBatch, ws, weightsPointer, chunkFileOffset, chunkSpanBytes, 0, numPromptTokens, kvCache);
            }
        });
        long tBatchLayersMs = (System.nanoTime() - tBatchLayers0) / 1_000_000;

        kvCache.advanceTokens(numPromptTokens);
        currentPos = numPromptTokens;
        System.arraycopy(promptXBatch[numPromptTokens - 1], 0, hiddenState, 0, dim);

        if (config.isVerbose()) {
            System.out.printf("[Timing] Batched Prefill (%d tokens): Embd=%d ms | Layers=%d ms | Total=%d ms%n",
                    numPromptTokens, tEmbdMs, tBatchLayersMs, (tEmbdMs + tBatchLayersMs));
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

            // 1. Sample next token from logits with Top-K or Greedy ArgMax
            long tSample0 = System.nanoTime();
            nextToken = computeEngine.sampleNextToken(hiddenState, config.getTemperature(), recentTokens);
            long tSampleMs = (System.nanoTime() - tSample0) / 1_000_000;
            totalSampleTimeMs += tSampleMs;

            if (tokenizer.isStopToken(nextToken)) {
                break;
            }

            recentTokens.add(nextToken);
            if (recentTokens.size() > 64) recentTokens.remove(0);

            String piece = tokenizer.decode(nextToken);
            if (!piece.isEmpty()) {
                tokenCallback.accept(piece);
            }
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

        if (config.isVerbose() && generatedCount > 0) {
            System.out.printf("%n--------------------------------------------------------------------------------%n");
            System.out.printf(" Inference Step Timings Summary (%d tokens generated):%n", generatedCount);
            System.out.printf("  • Avg Sampling / Logits:  %6.1f ms / token%n", (double) totalSampleTimeMs / generatedCount);
            System.out.printf("  • Avg Embedding Lookup:   %6.1f ms / token%n", (double) totalEmbdTimeMs / generatedCount);
            System.out.printf("  • Avg Layer Forward-Pass: %6.1f ms / token%n", (double) totalLayersTimeMs / generatedCount);
            System.out.printf("  • Total Throughput:       %6.2f tokens / sec%n",
                    (double) generatedCount / ((totalSampleTimeMs + totalEmbdTimeMs + totalLayersTimeMs) / 1000.0));
            System.out.printf("  • Dispatch Execution:     Native AVX2=%d | Java Fallback=%d%n",
                    StreamingTransformerEngine.nativeCalls.get(), StreamingTransformerEngine.fallbackCalls.get());
            System.out.printf("  • Memory Slicing:         Zero-Copy mmap=%d | Disk I/O Heap=%d%n",
                    StreamingTransformerEngine.zeroCopyMmapCalls.get(), StreamingTransformerEngine.heapDiskCalls.get());
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

    private String formatPrompt(String prompt) {
        if (prompt == null || prompt.isBlank()) return prompt;
        String arch = (indexer.getArchitecture() != null ? indexer.getArchitecture() : "").toLowerCase();

        // Already formatted by caller
        if (prompt.contains("<|im_start|>") || prompt.contains("[INST]") || prompt.contains("<|start_header_id|>")) {
            return prompt;
        }

        // Mistral / Llama-2 style with [INST]
        if (arch.contains("mistral") || !tokenizer.isByteBpe()) {
            return "[INST] " + prompt.trim() + " [/INST]";
        }

        // Default ChatML for Byte-BPE models (SmolLM, Qwen, etc.)
        return "<|im_start|>system\nYou are a helpful, respectful and honest assistant.<|im_end|>\n"
             + "<|im_start|>user\n" + prompt.trim() + "<|im_end|>\n"
             + "<|im_start|>assistant\n";
    }

    public static class Builder {
        private String modelPath;
        private File modelFile;
        private int chunkBudgetMB = 512;
        private int contextLength = 2048;
        private boolean useGPU = false;
        private boolean overlapIO = true;
        private float temperature = 0.7f;
        private boolean verbose = false;

        public Builder model(String modelPath) {
            this.modelPath = modelPath;
            return this;
        }

        public Builder model(File modelFile) {
            this.modelFile = modelFile;
            return this;
        }

        public Builder chunkBudgetMB(int mb) {
            this.chunkBudgetMB = mb;
            return this;
        }

        public Builder contextLength(int contextLength) {
            this.contextLength = contextLength;
            return this;
        }

        public Builder useGPU(boolean useGPU) {
            this.useGPU = useGPU;
            return this;
        }

        public Builder overlapIO(boolean overlapIO) {
            this.overlapIO = overlapIO;
            return this;
        }

        public Builder temperature(float temperature) {
            this.temperature = temperature;
            return this;
        }

        public Builder verbose(boolean verbose) {
            this.verbose = verbose;
            return this;
        }

        public FastAIStreamingModel build() throws Exception {
            File targetFile = modelFile;
            if (targetFile == null && modelPath != null && !modelPath.isBlank()) {
                targetFile = fastaimodel.streaming.io.ModelResolver.resolve(modelPath);
            }
            if (targetFile == null) {
                throw new IllegalArgumentException("Model file or model path must be specified");
            }
            StreamingConfig cfg = StreamingConfig.builder()
                    .chunkBudgetMB(chunkBudgetMB)
                    .contextLength(contextLength)
                    .useGPU(useGPU)
                    .overlapIO(overlapIO)
                    .temperature(temperature)
                    .verbose(verbose)
                    .build();
            return new FastAIStreamingModel(targetFile, cfg);
        }
    }
}
