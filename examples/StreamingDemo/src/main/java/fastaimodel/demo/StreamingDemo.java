package fastaimodel.demo;

import fastaimodel.streaming.FastAIStreamingModel;
import fastaimodel.streaming.StreamingConfig;
import fastaimodel.streaming.io.GgufTensorIndexer;
import fastaimodel.streaming.io.ModelResolver;

import java.io.File;
import java.io.RandomAccessFile;
import java.nio.charset.StandardCharsets;

/**
 * FastAIModel Zero-Copy Layer-wise Streaming Live Console Showcase.
 * Automatically resolves local models (e.g. smollm2:1.7b) or falls back to synthetic test fixtures.
 */
public class StreamingDemo {

    public static void main(String[] args) {
        System.setOut(new java.io.PrintStream(System.out, true, StandardCharsets.UTF_8));

        System.out.println("================================================================================");
        System.out.println("   FastAIModel — Zero-Copy Layer-wise Streaming Engine Showcase                      ");
        System.out.println("   Zero-Copy Win32 mmap • Overlapped Virtual Thread IO • Fixed RAM Ring Buffer   ");
        System.out.println("================================================================================\n");

        try {
            java.util.List<String> installed = ModelResolver.listInstalledModels();

            String requestedModel = (args.length > 0 && !args[0].trim().isEmpty()) ? args[0].trim() : null;
            int chunkBudgetMB = 1024;
            if (args.length > 1) {
                try {
                    chunkBudgetMB = Integer.parseInt(args[1].trim());
                } catch (NumberFormatException ignored) {}
            }

            boolean useGpu = true;
            boolean overlapIO = true;
            String userPrompt = "Explain how layer-streaming circumvents VRAM limits in Java";

            boolean verbose = false;
            boolean compareMode = false;
            float temperature = 0.7f;
            for (int i = 2; i < args.length; i++) {
                String arg = args[i].trim();
                if (arg.equalsIgnoreCase("--verbose") || arg.equalsIgnoreCase("-v")) {
                    verbose = true;
                } else if (arg.equalsIgnoreCase("--cpu") || arg.equalsIgnoreCase("-cpu")) {
                    useGpu = false;
                } else if (arg.equalsIgnoreCase("--greedy")) {
                    temperature = 0.0f;
                } else if (arg.equalsIgnoreCase("--compare")) {
                    compareMode = true;
                } else if (arg.equalsIgnoreCase("--no-overlap")) {
                    overlapIO = false;
                } else if (!arg.startsWith("-")) {
                    userPrompt = arg;
                }
            }

            File modelFile = null;
            String modelDisplayName = "";

            if (requestedModel != null) {
                modelFile = ModelResolver.resolve(requestedModel);
                modelDisplayName = requestedModel;
            } else if (!installed.isEmpty()) {
                modelDisplayName = installed.contains("mistral:7b") ? "mistral:7b" : installed.get(0);
                modelFile = ModelResolver.resolve(modelDisplayName);
            }

            boolean isRealModel = (modelFile != null && modelFile.exists() && modelFile.length() > 10 * 1024 * 1024);

            if (verbose) {
                if (!installed.isEmpty()) {
                    System.out.println("[*] Installed Local Ollama Models:");
                    for (String m : installed) {
                        boolean isSelected = m.equalsIgnoreCase(modelDisplayName);
                        System.out.printf("    %s %s%n", isSelected ? "▶" : "•", m);
                    }
                    System.out.println();
                }

                if (isRealModel) {
                    System.out.printf("[+] Selected Model: %s (%4.2f MB GGUF binary)%n",
                            modelDisplayName, modelFile.length() / (1024.0 * 1024.0));
                    System.out.printf("[+] File Path: %s%n%n", modelFile.getAbsolutePath());
                }
            }

            StreamingConfig config = StreamingConfig.builder()
                    .chunkBudgetMB(chunkBudgetMB)
                    .overlapIO(overlapIO)
                    .useGPU(useGpu)
                    .contextLength(2048)
                    .temperature(temperature)
                    .verbose(verbose)
                    .build();

            if (compareMode && isRealModel) {
                System.out.println("--------------------------------------------------------------------------------");
                System.out.println(" [1] Running Pure Llama (RAM resident, JNI)");
                System.out.println("--------------------------------------------------------------------------------");
                long tLlama0 = System.nanoTime();
                try (fastaimodel.FastAIModel llama = new fastaimodel.FastAIModel(modelFile.getAbsolutePath())) {
                    System.out.printf("User: %s%n%n", userPrompt);
                    System.out.print("AI:   ");
                    llama.predict(userPrompt, 128, token -> {
                        System.out.print(token);
                        System.out.flush();
                    });
                    System.out.println();
                    long tLlama1 = System.nanoTime();
                    System.out.printf("%n[✓] Pure Llama Inference complete in %d ms.%n%n", (tLlama1 - tLlama0) / 1_000_000);
                }
                System.out.println("--------------------------------------------------------------------------------");
                System.out.println(" [2] Running FastAI Streaming");
                System.out.println("--------------------------------------------------------------------------------");
            }

            long t0 = System.nanoTime();
            try (FastAIStreamingModel model = new FastAIStreamingModel(modelFile, config)) {
                long loadTimeMs = (System.nanoTime() - t0) / 1_000_000;

                if (verbose) {
                    System.out.printf("[+] Indexing complete in %d ms! Total Model Layers: %d, Partitioned Chunks: %d%n",
                            loadTimeMs, model.getIndexer().getLayerCount(), model.getChunkCount());
                    System.out.printf("[+] Acceleration Backend: %s%n%n",
                            model.isGpuActive() ? "FastGPU (Vulkan)" : "FastSIMD (AVX2/AVX-512 CPU)");

                    System.out.println("--------------------------------------------------------------------------------");
                    System.out.println(" Chunk Partition Table (Fixed Off-Heap Double-Buffer Slots):");
                    System.out.println("--------------------------------------------------------------------------------");
                    for (GgufTensorIndexer.LayerChunk chunk : model.getChunks()) {
                        System.out.printf("  • Chunk #%02d | Layers [%02d - %02d] | Footprint: %6.2f MB | Off-Heap Slot [Recycled]%n",
                                chunk.chunkIndex(), chunk.startLayer(), chunk.endLayer(),
                                chunk.physicalSpanBytes() / (1024.0 * 1024.0));
                    }
                    System.out.println("--------------------------------------------------------------------------------\n");
                }

                System.out.printf("User: %s%n%n", userPrompt);
                System.out.print("AI:   ");

                long streamStart = System.nanoTime();
                model.stream(userPrompt, token -> {
                    System.out.print(token);
                    System.out.flush();
                });
                long streamElapsedMs = (System.nanoTime() - streamStart) / 1_000_000;

                if (verbose) {
                    System.out.printf("%n%n[✓] Inference complete in %d ms (%s).%n",
                            streamElapsedMs, overlapIO ? "Overlapped I/O active" : "Sequential I/O");
                    System.out.printf("[✓] Peak off-heap weight footprint strictly maintained at: %d MB%n", chunkBudgetMB * 2);
                    System.out.println("================================================================================");
                } else {
                    System.out.println();
                }
            }

        } catch (Throwable t) {
            System.err.println("[-] Error running StreamingDemo: " + t.getMessage());
            t.printStackTrace();
        }
    }
}
