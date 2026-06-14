package fastaimodel;

public class Demo {
    public static void main(String[] args) {
        System.out.println("Starting FastAIModel native streaming test with Llama-3.2-1B-Instruct...");
        System.out.println("----------------------------------------------------------------------");

        String modelPath = "../FastAI/examples/Demo/models/Llama-3.2-1B-Instruct-Q8_0.gguf";

        try (FastAIModel model = new FastAIModel(modelPath, 2048, 0)) {
            String prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" +
                    "Korrigiere diesen text in makelloses Deutsch.<|eot_id|>" +
                    "<|start_header_id|>user<|end_header_id|>\n\n" +
                    "ich habe demo2 gesehen. nicht bei FAstAI direkt reinarbeiten . ich denke die Demo muss zu FastAIMemory<|eot_id|>" +
                    "<|start_header_id|>assistant<|end_header_id|>\n\n";

            model.predict(prompt, 100, token -> {
                System.out.print(token);
                System.out.flush();
            });
        }

        System.out.println("\n----------------------------------------------------------------------");
        System.out.println("Test completed successfully!");
    }
}
