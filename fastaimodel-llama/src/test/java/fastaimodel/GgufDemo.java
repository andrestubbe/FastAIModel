package fastaimodel;

public class GgufDemo {
    public static void main(String[] args) {
        String modelArg = System.getProperty("model.path");
        if (modelArg == null || modelArg.isBlank()) {
            modelArg = "llama3.2:1b";
        }

        boolean verbose = Boolean.parseBoolean(System.getProperty("verbose", "true"));
        if (verbose) {
            FastAIModel.setVerbose(true);
        }

        System.out.println("=== FastAIModel GGUF In-Process Demo ===");
        System.out.println("Model: " + modelArg);
        System.out.println();

        try (FastAIModel model = new FastAIModel(modelArg, 2048, 0)) {
            System.out.println("\n--- Generating ---");
            model.predict("Write a 3-sentence summary of artificial intelligence in German:", 128, token -> {
                System.out.print(token);
                System.out.flush();
            });
            System.out.println("\n------------------");
        } catch (Exception e) {
            System.err.println("Error in GgufDemo: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
