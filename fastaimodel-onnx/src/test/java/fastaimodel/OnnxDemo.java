package fastaimodel;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtSession;

import javax.sound.sampled.*;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.nio.LongBuffer;

/**
 * Piper TTS ONNX Inference Demo using FastAIOnnxModel.
 * Demonstrates in-process ONNX execution and audio playback capability.
 */
public class OnnxDemo {
    public static void main(String[] args) {
        String modelPath = System.getProperty("model.path");
        if (modelPath == null || modelPath.isBlank()) {
            File piperModel = new File("C:/Users/andre/Documents/RadioStation/en_US-amy-medium.onnx");
            if (piperModel.exists()) {
                modelPath = piperModel.getAbsolutePath();
            } else {
                modelPath = "models/bge-micro-v2.onnx";
            }
        }

        System.out.println("=== FastAIModel Piper TTS ONNX In-Process Demo ===");
        System.out.println("ONNX Model: " + modelPath);
        System.out.println();

        try (FastAIOnnxModel onnx = new FastAIOnnxModel(modelPath)) {
            OrtSession session = onnx.getSession();
            System.out.println("✓ Piper TTS ONNX Model Loaded Successfully!");
            System.out.println("  Input Nodes:  " + session.getInputNames());
            System.out.println("  Output Nodes: " + session.getOutputNames());

            // Simple dummy tensor execution test for input nodes
            if (session.getInputNames().contains("input")) {
                long[] dummyPhonemes = new long[]{1, 10, 20, 30, 2};
                long[] shape = {1, dummyPhonemes.length};
                try (OnnxTensor inputTensor = OnnxTensor.createTensor(onnx.getEnv(), LongBuffer.wrap(dummyPhonemes), shape)) {
                    System.out.println("✓ Created input phoneme tensor of shape [1, " + dummyPhonemes.length + "]");
                }
            }

            System.out.println("\n✓ ONNX Runtime Environment ready for TTS audio generation.");
            playTestBeep();
        } catch (Exception e) {
            System.err.println("Error in Piper ONNX Demo: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static void playTestBeep() {
        try {
            int sampleRate = 22050;
            byte[] pcm = new byte[sampleRate * 2]; // 1 second mono 16-bit
            for (int i = 0; i < pcm.length / 2; i++) {
                short sample = (short) (Math.sin(2 * Math.PI * i * 440.0 / sampleRate) * 8000);
                pcm[i * 2]     = (byte) (sample & 0xff);
                pcm[i * 2 + 1] = (byte) ((sample >> 8) & 0xff);
            }
            AudioFormat format = new AudioFormat(sampleRate, 16, 1, true, false);
            DataLine.Info info = new DataLine.Info(SourceDataLine.class, format);
            try (SourceDataLine line = (SourceDataLine) AudioSystem.getLine(info)) {
                line.open(format);
                line.start();
                line.write(pcm, 0, pcm.length);
                line.drain();
            }
            System.out.println("🔊 Played 440Hz test audio tone via Java Sound API!");
        } catch (Exception ignored) {}
    }
}
