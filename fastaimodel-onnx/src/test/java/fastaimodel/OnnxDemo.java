package fastaimodel;

public class OnnxDemo {
    public static void main(String[] args) {
        String modelArg = System.getProperty("model.path");
        if (modelArg == null || modelArg.isBlank()) {
            modelArg = "models/bge-micro-v2.onnx";
        }

        System.out.println("=== FastAIModel ONNX Demo ===");
        System.out.println("Model Path: " + modelArg);
        System.out.println();

        try (FastAIOnnxModel onnx = new FastAIOnnxModel(modelArg)) {
            System.out.println("✓ ONNX Session created successfully: " + onnx.getSession());
            System.out.println("✓ Environment: " + onnx.getEnv());
        } catch (Exception e) {
            System.err.println("Error in OnnxDemo: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
