package fastaimodel;

public class Demo {
    public static void main(String[] args) {
        System.out.println("Starting FastAIModel native streaming test with Llama-3.2-1B-Instruct...");
        System.out.println("----------------------------------------------------------------------");

        String modelPath = System.getProperty("model.path");
        if (modelPath == null || modelPath.isEmpty()) {
            modelPath = "../FastAI/examples/Demo/models/Llama-3.2-1B-Instruct-Q8_0.gguf";
            if (!new java.io.File(modelPath).exists()) {
                modelPath = "../../FastAI/examples/Demo/models/Llama-3.2-1B-Instruct-Q8_0.gguf";
            }
        }
        modelPath = new java.io.File(modelPath).getAbsolutePath();

        try (FastAIModel model = new FastAIModel(modelPath, 2048, 0)) {
            String prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>" +
                    "<|start_header_id|>user<|end_header_id|>\n\nWho are you?<|eot_id|>" +
                    "<|start_header_id|>assistant<|end_header_id|>\n\n";

            System.out.println("\n==================== START OF GENERATION ====================");
            model.predict(prompt, 100, token -> {
                System.out.print(token);
                System.out.flush();
            });
            System.out.println("\n===================== END OF GENERATION =====================\n");
        }

        System.out.println("\n----------------------------------------------------------------------");
        System.out.println("Test completed successfully!");
    }
}
