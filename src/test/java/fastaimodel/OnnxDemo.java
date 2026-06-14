package fastaimodel;

import ai.onnxruntime.OrtEnvironment;

public class OnnxDemo {
    public static void main(String[] args) {
        System.out.println("Starting FastAIModel ONNX streaming test...");
        System.out.println("------------------------------------------------");
        try {
            OrtEnvironment env = OrtEnvironment.getEnvironment();
            System.out.println("✅ ONNX Runtime Environment loaded successfully!");
            env.close();
        } catch (Exception e) {
            System.err.println("❌ Failed to initialize ONNX Runtime: " + e.getMessage());
            e.printStackTrace();
        }
        System.out.println("------------------------------------------------");
    }
}
