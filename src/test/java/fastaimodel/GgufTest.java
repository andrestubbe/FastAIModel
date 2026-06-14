package fastaimodel;

import org.junit.jupiter.api.Test;
import java.io.File;
import static org.junit.jupiter.api.Assertions.*;

public class GgufTest {

    @Test
    public void testGgufLoading() {
        String modelPath = "../FastAI/examples/Demo/models/Llama-3.2-1B-Instruct-Q8_0.gguf";
        if (!new File(modelPath).exists()) {
            modelPath = "../../FastAI/examples/Demo/models/Llama-3.2-1B-Instruct-Q8_0.gguf";
        }
        File modelFile = new File(modelPath);
        
        if (modelFile.exists()) {
            System.out.println("Model file found, running GGUF load test...");
            try (FastAIModel model = new FastAIModel(modelPath, 128, 0)) {
                assertNotNull(model);
                System.out.println("✅ GGUF Model loaded successfully!");
            }
        } else {
            System.out.println("GGUF model file not found at " + modelPath + ", skipping test.");
        }
    }
}
