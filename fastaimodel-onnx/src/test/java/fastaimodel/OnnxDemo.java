package fastaimodel;

import ai.onnxruntime.OrtSession;

import java.io.File;

/**
 * FastAIOnnxModel In-Process ONNX Inference Engine Demo.
 * Demonstrates loading and verifying ONNX sessions (Embeddings, TTS, Computer Vision)
 * with zero C++ Llama DLL dependencies.
 */
public class OnnxDemo {
    public static void main(String[] args) {
        String modelPath = System.getProperty("model.path");
        if (modelPath == null || modelPath.isBlank()) {
            File germanPiper = new File("models/de_DE-thorsten-medium.onnx");
            File bgeModel = new File("models/bge-micro-v2.onnx");
            if (germanPiper.exists()) {
                modelPath = germanPiper.getAbsolutePath();
            } else if (bgeModel.exists()) {
                modelPath = bgeModel.getAbsolutePath();
            } else {
                modelPath = "models/bge-micro-v2.onnx";
            }
        }

        System.out.println("=== FastAIModel ONNX In-Process Engine Demo ===");
        System.out.println("ONNX Model Path: " + modelPath);
        System.out.println("-----------------------------------------------");

        try (FastAIOnnxModel onnx = new FastAIOnnxModel(modelPath)) {
            OrtSession session = onnx.getSession();
            System.out.println("✓ ONNX Model Loaded Successfully!");
            System.out.println("  Input Nodes:  " + session.getInputNames());
            System.out.println("  Output Nodes: " + session.getOutputNames());
            System.out.println();
            System.out.println("✓ ONNX Runtime Environment ready for in-process inference.");
        } catch (Exception e) {
            System.err.println("Error in ONNX Demo: " + e.getMessage());
            e.printStackTrace();
        }
        System.out.println("-----------------------------------------------");
    }
}
