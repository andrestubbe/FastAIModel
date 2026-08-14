package fastaimodel.benchmark;

import fastaimodel.FastAIModel;

public class TestCpuVsGpu {
    public static void main(String[] args) {
        String modelPath = "c:/Users/andre/Documents/2026-06-14-Work-FastJava/FastAI/examples/Demo/models/qwen2.5-coder-0.5b-instruct-q8_0.gguf";
        String prompt = "Write a quicksort method in Java:";
        int maxTokens = 64;

        System.out.println("\n--- [Test 1/2] Running CPU Only (n_gpu_layers = 0) ---");
        long startCpu = System.currentTimeMillis();
        try (FastAIModel cpuModel = new FastAIModel(modelPath, 1024, 0)) {
            long loadTime = System.currentTimeMillis() - startCpu;
            System.out.println("Model loaded in " + loadTime + " ms");
            
            long genStart = System.currentTimeMillis();
            final int[] tokens = {0};
            cpuModel.predict(prompt, maxTokens, token -> {
                tokens[0]++;
                System.out.print(token);
                System.out.flush();
            });
            long genTime = System.currentTimeMillis() - genStart;
            double tps = (tokens[0] * 1000.0) / Math.max(genTime, 1);
            System.out.printf("\n\n[RESULT] CPU Generation Speed: %.2f Tokens/sec (%d tokens in %d ms)\n", tps, tokens[0], genTime);
        } catch (Exception e) {
            System.out.println("CPU Error: " + e.getMessage());
            e.printStackTrace();
        }

        System.out.println("\n--- [Test 2/2] Running Intel Iris GPU Accelerated (n_gpu_layers = 99) ---");
        long startGpu = System.currentTimeMillis();
        try (FastAIModel gpuModel = new FastAIModel(modelPath, 1024, 99)) {
            long loadTime = System.currentTimeMillis() - startGpu;
            System.out.println("Vulkan GPU Model loaded in " + loadTime + " ms");
            
            long genStart = System.currentTimeMillis();
            final int[] tokens = {0};
            gpuModel.predict(prompt, maxTokens, token -> {
                tokens[0]++;
                System.out.print(token);
                System.out.flush();
            });
            long genTime = System.currentTimeMillis() - genStart;
            double tps = (tokens[0] * 1000.0) / Math.max(genTime, 1);
            System.out.printf("\n\n[RESULT] Intel Iris GPU Speed: %.2f Tokens/sec (%d tokens in %d ms)\n", tps, tokens[0], genTime);
        } catch (Exception e) {
            System.out.println("GPU Error: " + e.getMessage());
            e.printStackTrace();
        }

        System.out.println("\n==================================================");
        System.out.println("🎉 Test Complete!");
        System.out.println("==================================================");
    }
}
