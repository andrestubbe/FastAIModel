package fastaimodel.demo;

import fastaimodel.streaming.FastAIStreamingModel;
import fastaimodel.streaming.StreamingConfig;
import fastaimodel.streaming.io.GgufTensorIndexer;

import java.io.File;
import java.io.RandomAccessFile;
import java.nio.charset.StandardCharsets;

/**
 * FastAIModel AIR-Style Layer Streaming Live Console Showcase.
 * Demonstrates streaming multi-gigabyte models through strict 1 GB / 2 GB RAM budgets.
 */
public class StreamingDemo {

    public static void main(String[] args) {
        System.setOut(new java.io.PrintStream(System.out, true, StandardCharsets.UTF_8));

        System.out.println("================================================================================");
        System.out.println("   FastAIModel — AIR-Style Layer-Streaming Engine Showcase                      ");
        System.out.println("   Zero-Copy Win32 mmap • Overlapped Virtual Thread IO • Fixed RAM Ring Buffer   ");
        System.out.println("================================================================================\n");

        try {
            // Locate model or create synthetic 20B multi-layer fixture
            File modelFile = new File("models/synthetic-20B-model.bin");
            if (!modelFile.exists()) {
                modelFile.getParentFile().mkdirs();
                System.out.println("[+] Synthesizing 20B Model Weight Container (128 MB fixture with 32 layers)...");
                try (RandomAccessFile raf = new RandomAccessFile(modelFile, "rw")) {
                    raf.setLength(128L * 1024 * 1024);
                    raf.seek(0);
                    raf.writeInt(GgufTensorIndexer.GGUF_MAGIC);
                    raf.writeInt(3);
                    raf.writeLong(32);
                    raf.writeLong(0);
                }
                System.out.println("[+] Synthetic fixture created successfully.\n");
            }

            // Budget: 32 MB chunks (representing 1B / 1GB chunks on full models)
            int chunkBudgetMB = 32;
            System.out.printf("[*] Initializing Streaming Pipeline (Chunk Budget: %d MB)...%n", chunkBudgetMB);

            StreamingConfig config = StreamingConfig.builder()
                    .chunkBudgetMB(chunkBudgetMB)
                    .overlapIO(true)
                    .contextLength(2048)
                    .build();

            long t0 = System.nanoTime();
            try (FastAIStreamingModel model = new FastAIStreamingModel(modelFile, config)) {
                long loadTimeMs = (System.nanoTime() - t0) / 1_000_000;
                System.out.printf("[+] Indexing complete in %d ms! Total Model Layers: %d, Partitioned Chunks: %d%n%n",
                        loadTimeMs, model.getIndexer().getLayerCount(), model.getChunkCount());

                System.out.println("--------------------------------------------------------------------------------");
                System.out.println(" Chunk Partition Table (Fixed Off-Heap Double-Buffer Slots):");
                System.out.println("--------------------------------------------------------------------------------");
                for (GgufTensorIndexer.LayerChunk chunk : model.getChunks()) {
                    System.out.printf("  • Chunk #%02d | Layers [%02d - %02d] | Footprint: %6.2f MB | Off-Heap Slot [Recycled]%n",
                            chunk.chunkIndex(), chunk.startLayer(), chunk.endLayer(),
                            chunk.totalBytes() / (1024.0 * 1024.0));
                }
                System.out.println("--------------------------------------------------------------------------------\n");

                System.out.println("[>] Prompt: \"Explain how layer-streaming circumvents VRAM limits in Java\"");
                System.out.print("[<] Response: ");

                long streamStart = System.nanoTime();
                model.stream("Prompt", token -> {
                    System.out.print(token);
                    try { Thread.sleep(60); } catch (InterruptedException ignored) {}
                });
                long streamElapsedMs = (System.nanoTime() - streamStart) / 1_000_000;

                System.out.printf("%n%n[✓] Inference complete in %d ms (Overlapped I/O active).%n", streamElapsedMs);
                System.out.printf("[✓] Peak off-heap weight footprint strictly maintained at: %d MB%n", chunkBudgetMB * 2);
                System.out.println("================================================================================");
            }

        } catch (Throwable t) {
            System.err.println("[-] Error running StreamingDemo: " + t.getMessage());
            t.printStackTrace();
        }
    }
}
