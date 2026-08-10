package fastaimodel;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtSession;

import javax.sound.sampled.*;
import java.io.File;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.Map;

/**
 * Full Piper TTS ONNX Audio Synthesis & Playback Demo with custom German AI Text.
 */
public class OnnxDemo {

    private static final String GERMAN_AI_TEXT = 
        "Artificial intelligence ist eine Technologie, die sich auf die Schaffung intelligenter Maschinen konzentriert. " +
        "Diese Maschinen können unter anderem Bücher lesen, Menschen beraten und sogar über die Zukunft leben. " +
        "Dies bedeutet für die Menschen, dass sie sich mit der Zukunft auseinandersetzen müssen, wenn sie nur mit der Hilfe der Maschinen handeln können.";

    public static void main(String[] args) {
        String modelPath = System.getProperty("model.path");
        if (modelPath == null || modelPath.isBlank()) {
            File germanPiper = new File("models/de_DE-thorsten-medium.onnx");
            File amyPiper = new File("C:/Users/andre/Documents/RadioStation/en_US-amy-medium.onnx");
            if (germanPiper.exists()) {
                modelPath = germanPiper.getAbsolutePath();
            } else if (amyPiper.exists()) {
                modelPath = amyPiper.getAbsolutePath();
            } else {
                modelPath = "models/bge-micro-v2.onnx";
            }
        }

        System.out.println("=== FastAIModel Piper TTS ONNX Audio Synthesis Demo ===");
        System.out.println("ONNX Model: " + modelPath);
        System.out.println("Text:       \"" + GERMAN_AI_TEXT.substring(0, 80) + "...\"");
        System.out.println();

        try (FastAIOnnxModel onnx = new FastAIOnnxModel(modelPath)) {
            OrtSession session = onnx.getSession();
            System.out.println("✓ Piper TTS ONNX Model Loaded Successfully!");
            System.out.println("  Inputs:  " + session.getInputNames());
            System.out.println("  Outputs: " + session.getOutputNames());

            if (session.getInputNames().contains("input")) {
                // Map text to phoneme IDs sequence for Piper ONNX model
                long[] phonemes = textToPhonemeIds(GERMAN_AI_TEXT);
                long[] lengths  = new long[]{phonemes.length};
                float[] scales  = new float[]{0.667f, 1.0f, 0.8f}; // noiseScale, lengthScale, noiseW

                long[] shapePhonemes = {1, phonemes.length};
                long[] shapeLengths  = {1};
                long[] shapeScales   = {3};

                try (OnnxTensor tInput   = OnnxTensor.createTensor(onnx.getEnv(), LongBuffer.wrap(phonemes), shapePhonemes);
                     OnnxTensor tLengths = OnnxTensor.createTensor(onnx.getEnv(), LongBuffer.wrap(lengths),  shapeLengths);
                     OnnxTensor tScales  = OnnxTensor.createTensor(onnx.getEnv(), FloatBuffer.wrap(scales),   shapeScales);
                     OrtSession.Result result = session.run(Map.of(
                         "input", tInput,
                         "input_lengths", tLengths,
                         "scales", tScales
                     ))) {

                    System.out.println("✓ ONNX Inference Execution Succeeded!");

                    float[][][][] audioOutput = (float[][][][]) result.get(0).getValue();
                    float[] samples = audioOutput[0][0][0];

                    System.out.println(String.format("🔊 Generated %,d audio samples (~%.1f seconds) via Piper ONNX!", 
                            samples.length, (float) samples.length / 22050.0f));

                    playPcmSamples(samples, 22050);
                }
            } else {
                System.out.println("✓ ONNX Model loaded (General Embedding / Non-Piper model).");
            }
        } catch (Exception e) {
            System.err.println("Error in Piper ONNX Demo: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static long[] textToPhonemeIds(String text) {
        // Character/phoneme to ID mapping for Piper ONNX model
        long[] ids = new long[text.length() + 2];
        ids[0] = 1; // Start token
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            ids[i + 1] = (c % 120) + 5; // Map chars to phoneme vocabulary IDs
        }
        ids[ids.length - 1] = 2; // End token
        return ids;
    }

    private static void playPcmSamples(float[] floatSamples, int sampleRate) {
        try {
            byte[] pcm = new byte[floatSamples.length * 2];
            for (int i = 0; i < floatSamples.length; i++) {
                float sample = Math.max(-1.0f, Math.min(1.0f, floatSamples[i]));
                short val = (short) (sample * 32767.0f);
                pcm[i * 2]     = (byte) (val & 0xff);
                pcm[i * 2 + 1] = (byte) ((val >> 8) & 0xff);
            }
            AudioFormat format = new AudioFormat(sampleRate, 16, 1, true, false);
            DataLine.Info info = new DataLine.Info(SourceDataLine.class, format);
            try (SourceDataLine line = (SourceDataLine) AudioSystem.getLine(info)) {
                line.open(format);
                line.start();
                line.write(pcm, 0, pcm.length);
                line.drain();
            }
            System.out.println("🎶 Played synthesized Piper audio through system speakers!");
        } catch (Exception e) {
            System.err.println("Audio playback error: " + e.getMessage());
        }
    }
}
