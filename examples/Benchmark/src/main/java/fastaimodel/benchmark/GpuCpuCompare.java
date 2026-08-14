package fastaimodel.benchmark;

import fastaimodel.FastAIModel;

public class GpuCpuCompare {
    public static void main(String[] args) {
        String modelPath = "models/qwen2.5-coder-1.5b.gguf";
        String prompt = "Write a quicksort function in Java:";
        int tokensToGenerate = 64;

        System.out.println("==================================================");
        System.out.println("⚡ FastAIModel Benchmark: CPU vs Intel Iris GPU (Vulkan)");
        System.out.println("==================================================");

        // 1. Run CPU Only (0 GPU layers)
        System.out.println("\n---> Testing CPU Only (n_gpu_layers = 0)...");
        try {
            long start = System.currentTimeMillis();
            try (FastAIModel cpuModel = new FastAIModel(modelPath, 2048, 0)) {
                long loadTime = System.currentTimeMillis() - start;
                System.out.println("  Model Loaded in " + loadTime + " ms");

                long genStart = System.currentTimeMillis();
                cpuModel.predict(prompt, tokensToGenerate, token -> {});
                long genTime = System.currentTimeMillis() - genStart;

                double tps = (tokensToGenerate * 1000.0) / genTime;
                System.out.printf("  CPU Speed: %.2f Tokens/sec (Time: %d ms)\n", tps, genTime);
            }
        } catch (Exception e) {
            System.out.println("  CPU Test Note: " + e.getMessage());
        }

        // 2. Run Intel Iris GPU Accelerated (Vulkan, 99 GPU layers)
        System.out.println("\n---> Testing Intel Iris GPU Accelerated (Vulkan, n_gpu_layers = 99)...");
        try {
            long start = System.currentTimeMillis();
            try (FastAIModel gpuModel = new FastAIModel(modelPath, 2048, 99)) {
                long loadTime = System.currentTimeMillis() - start;
                System.out.println("  Vulkan Model Loaded in " + loadTime + " ms");

                long genStart = System.currentTimeMillis();
                gpuModel.predict(prompt, tokensToGenerate, token -> {});
                long genTime = System.currentTimeMillis() - genStart;

                double tps = (tokensToGenerate * 1000.0) / genTime;
                System.out.printf("  Intel Iris GPU Speed: %.2f Tokens/sec (Time: %d ms)\n", tps, genTime);
            }
        } catch (Exception e) {
            System.out.println("  GPU Test Note: " + e.getMessage());
        }

        System.out.println("\n==================================================");
        System.out.println("🎉 Benchmark Complete!");
        System.out.println("==================================================");
    }
}
