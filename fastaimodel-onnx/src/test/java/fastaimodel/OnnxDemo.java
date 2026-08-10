package fastaimodel;

import ai.onnxruntime.OrtSession;

/**
 * FastAIOnnxModel In-Process ONNX Inference Engine Demo.
 * Demonstrates loading and running ONNX models (Embeddings, Piper TTS, Kokoro)
 * with zero C++ Llama DLL dependencies.
 */
public class OnnxDemo {
    public static void main(String[] args) {
        String modelPath = System.getProperty("model.path");
        if (modelPath == null || modelPath.isBlank()) {
            modelPath = "models/bge-micro-v2.onnx";
        }

        System.out.println("=== FastAIModel ONNX In-Process Demo ===");
        System.out.println("ONNX Model Path: " + modelPath);
        System.out.println();

        try (FastAIOnnxModel onnx = new FastAIOnnxModel(modelPath)) {
            OrtSession session = onnx.getSession();
            System.out.println("✓ ONNX Model Loaded Successfully!");
            System.out.println("  Input Nodes:  " + session.getInputNames());
            System.out.println("  Output Nodes: " + session.getOutputNames());
            System.out.println();
            System.out.println("✓ ONNX Runtime Environment ready for zero-overhead in-process inference.");
        } catch (Exception e) {
            System.err.println("Error in ONNX Demo: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
