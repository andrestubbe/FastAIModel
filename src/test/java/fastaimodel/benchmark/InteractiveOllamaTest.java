package fastaimodel.benchmark;

import fastaimodel.FastAIModel;
import fastaimodel.OllamaModelResolver;

public class InteractiveOllamaTest {
    public static void main(String[] args) {
        String inputModel = (args.length > 0 && !args[0].isEmpty()) ? args[0] : "ollama:qwen2.5-coder:0.5b";
        String resolvedPath = OllamaModelResolver.resolve(inputModel);
        String prompt = "Write a quicksort method in Java:";
        int maxTokens = 64;

        System.out.println("==================================================");
        System.out.println("⚡ FastAIModel Direct CPU vs Intel Iris GPU Comparison");
        System.out.println("Input Model: " + inputModel);
        System.out.println("Resolved Path: " + resolvedPath);
        System.out.println("==================================================");

        // 1. CPU Only Test
        System.out.println("\n--- [1/2] Running CPU Only (n_gpu_layers = 0) ---");
        long startCpu = System.currentTimeMillis();
        double cpuTps = 0.0;
        long cpuGenTime = 0;

        try (FastAIModel cpuModel = new FastAIModel(resolvedPath, 2048, 0)) {
            long loadTime = System.currentTimeMillis() - startCpu;
            System.out.println("Model loaded in " + loadTime + " ms\n");

            long genStart = System.currentTimeMillis();
            final int[] tokens = {0};
            cpuModel.predict(prompt, maxTokens, token -> {
                tokens[0]++;
                System.out.print(token);
                System.out.flush();
            });
            cpuGenTime = System.currentTimeMillis() - genStart;
            cpuTps = (tokens[0] * 1000.0) / Math.max(cpuGenTime, 1);
            System.out.printf("\n\n-> CPU Result: %.2f Tokens/sec (%d tokens in %d ms)\n", cpuTps, tokens[0], cpuGenTime);
        } catch (Exception e) {
            System.out.println("CPU Test Error: " + e.getMessage());
        }

        // 2. Intel Iris GPU Test
        System.out.println("\n--- [2/2] Running Intel Iris GPU Accelerated (n_gpu_layers = 99) ---");
        long startGpu = System.currentTimeMillis();
        double gpuTps = 0.0;
        long gpuGenTime = 0;

        try (FastAIModel gpuModel = new FastAIModel(resolvedPath, 2048, 99)) {
            long loadTime = System.currentTimeMillis() - startGpu;
            System.out.println("Vulkan GPU Model loaded in " + loadTime + " ms\n");

            long genStart = System.currentTimeMillis();
            final int[] tokens = {0};
            gpuModel.predict(prompt, maxTokens, token -> {
                tokens[0]++;
                System.out.print(token);
                System.out.flush();
            });
            gpuGenTime = System.currentTimeMillis() - genStart;
            gpuTps = (tokens[0] * 1000.0) / Math.max(gpuGenTime, 1);
            System.out.printf("\n\n-> Intel Iris GPU Result: %.2f Tokens/sec (%d tokens in %d ms)\n", gpuTps, tokens[0], gpuGenTime);
        } catch (Exception e) {
            System.out.println("GPU Test Error: " + e.getMessage());
        }

        // Summary
        System.out.println("\n==================================================");
        System.out.println("📊 FINAL COMPARISON SUMMARY");
        System.out.println("==================================================");
        System.out.printf("  💻 CPU Only:         %.2f Tokens/sec (%d ms)\n", cpuTps, cpuGenTime);
        System.out.printf("  🎮 Intel Iris GPU:   %.2f Tokens/sec (%d ms)\n", gpuTps, gpuGenTime);
        if (cpuTps > 0 && gpuTps > 0) {
            double speedup = (gpuTps / cpuTps);
            System.out.printf("  🚀 GPU Speedup Factor: %.2fx faster!\n", speedup);
        }
        System.out.println("==================================================");
    }
}
