package fastaimodel.streaming;

import fastaimodel.streaming.io.GgufTensorIndexer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.File;
import java.io.RandomAccessFile;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class StreamingSanityTest {

    @Test
    public void testLayerChunkPartitioningAndStreaming(@TempDir Path tempDir) throws Exception {
        // Create a 64 MB synthetic model file (representing 32 layers)
        File modelFile = tempDir.resolve("synthetic-20B-model.bin").toFile();
        long fileSize = 64L * 1024 * 1024; // 64 MB
        try (RandomAccessFile raf = new RandomAccessFile(modelFile, "rw")) {
            raf.setLength(fileSize);
            raf.seek(0);
            raf.writeInt(GgufTensorIndexer.GGUF_MAGIC); // Valid magic
            raf.writeInt(3); // GGUF v3
            raf.writeLong(32); // 32 tensors
            raf.writeLong(0);  // 0 metadata KV
        }

        // Budget: 16 MB chunks -> should create exactly 4 chunks
        long chunkBudget = 16L * 1024 * 1024;
        StreamingConfig config = StreamingConfig.builder()
                .chunkBudgetMB(16)
                .overlapIO(true)
                .build();

        try (FastAIStreamingModel model = new FastAIStreamingModel(modelFile, config)) {
            assertEquals(32, model.getIndexer().getLayerCount(), "Should have 32 layers");
            assertTrue(model.getChunkCount() >= 4, "Should create at least 4 chunks");

            List<String> generatedTokens = new ArrayList<>();
            model.stream("Hello test", generatedTokens::add);

            assertFalse(generatedTokens.isEmpty(), "Tokens should have streamed");
            assertEquals(8, generatedTokens.size());
        }
    }
}
