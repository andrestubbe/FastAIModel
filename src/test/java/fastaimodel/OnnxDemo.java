package fastaimodel;

import javax.sound.sampled.*;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;

public class OnnxDemo {
    public static void main(String[] args) {
        System.out.println("Starting FastAIModel ONNX streaming test (TTS)...");
        System.out.println("------------------------------------------------");

        String modelPath = System.getProperty("model.path");
        String piperPath = "piper.exe";
        if (modelPath == null || modelPath.isEmpty()) {
            String basePath = "../FastBot/examples/Demo/";
            if (!new File(basePath + "piper.exe").exists()) {
                basePath = "../../FastBot/examples/Demo/";
            }
            piperPath = new File(basePath + "piper.exe").getAbsolutePath();
            modelPath = new File(basePath + "en_US-danny-low.onnx").getAbsolutePath();
        } else {
            modelPath = new File(modelPath).getAbsolutePath();
            piperPath = new File(modelPath).getParent() + File.separator + "piper.exe";
        }

        if (new File(piperPath).exists() && new File(modelPath).exists()) {
            System.out.println("[OK] Found Piper executable and voice model.");
            String text = "I'm an AI assistant, which means I'm a computer program designed to understand and respond to human language. I'm here to help answer your questions, provide information, and engage in conversation. I don't have a personal identity or consciousness, but I'm here to assist you with any questions or topics you'd like to discuss. I'm a large language model, trained on a vast amount of text data, which allows me to generate human-like responses to a wide range of questions and topics.";
            System.out.println("Text to speak: \"" + text + "\"");
            
            try {
                Path tempOutput = Files.createTempFile("piper_out", ".wav");
                ProcessBuilder pb = new ProcessBuilder(
                    new File(piperPath).getAbsolutePath(),
                    "--model", new File(modelPath).getAbsolutePath(),
                    "--output_file", tempOutput.toAbsolutePath().toString()
                );
                pb.directory(new File(piperPath).getParentFile());
                pb.redirectError(ProcessBuilder.Redirect.DISCARD);
                
                Process p = pb.start();
                try (OutputStream os = p.getOutputStream();
                     BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(os, "UTF-8"))) {
                    writer.write(text);
                    writer.flush();
                }
                
                if (p.waitFor() == 0 && Files.exists(tempOutput) && Files.size(tempOutput) > 0) {
                    System.out.println("[OK] Audio generated successfully!");
                    playWav(tempOutput.toFile());
                } else {
                    System.err.println("[ERROR] Piper execution failed or generated empty file.");
                }
                Files.deleteIfExists(tempOutput);
            } catch (Exception e) {
                System.err.println("[ERROR] Error during synthesis: " + e.getMessage());
                e.printStackTrace();
            }
        } else {
            System.out.println("[ERROR] Piper or ONNX model not found. Check paths:");
            System.out.println("Piper path: " + new File(piperPath).getAbsolutePath());
            System.out.println("Model path: " + new File(modelPath).getAbsolutePath());
        }
        System.out.println("------------------------------------------------");
    }

    private static void playWav(File wavFile) {
        try (AudioInputStream audioIn = AudioSystem.getAudioInputStream(wavFile)) {
            Clip clip = AudioSystem.getClip();
            clip.open(audioIn);
            clip.start();
            System.out.println("[PLAY] Playing audio...");
            Thread.sleep(clip.getMicrosecondLength() / 1000 + 500);
        } catch (Exception e) {
            System.err.println("Failed to play audio: " + e.getMessage());
        }
    }
}
