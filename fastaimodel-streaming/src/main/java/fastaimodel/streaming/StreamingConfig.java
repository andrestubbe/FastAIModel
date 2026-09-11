package fastaimodel.streaming;

/**
 * Configuration for FastAIModel AIR-style Layer Streaming.
 */
public class StreamingConfig {

    private final int chunkBudgetMB;
    private final int contextLength;
    private final boolean overlapIO;
    private final boolean useGPU;
    private final float temperature;

    private StreamingConfig(Builder builder) {
        this.chunkBudgetMB = builder.chunkBudgetMB;
        this.contextLength = builder.contextLength;
        this.overlapIO = builder.overlapIO;
        this.useGPU = builder.useGPU;
        this.temperature = builder.temperature;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static StreamingConfig default1GB() {
        return builder().chunkBudgetMB(1024).useGPU(true).build();
    }

    public static StreamingConfig default2GB() {
        return builder().chunkBudgetMB(2048).useGPU(true).build();
    }

    public int getChunkBudgetMB() { return chunkBudgetMB; }
    public long getChunkBudgetBytes() { return (long) chunkBudgetMB * 1024 * 1024; }
    public int getContextLength() { return contextLength; }
    public boolean isOverlapIO() { return overlapIO; }
    public boolean isUseGPU() { return useGPU; }
    public float getTemperature() { return temperature; }

    public static class Builder {
        private int chunkBudgetMB = 1024; // Default 1 GB chunk budget
        private int contextLength = 2048;
        private boolean overlapIO = true;
        private boolean useGPU = true;
        private float temperature = 0.7f;

        public Builder chunkBudgetMB(int mb) {
            this.chunkBudgetMB = mb;
            return this;
        }

        public Builder contextLength(int length) {
            this.contextLength = length;
            return this;
        }

        public Builder overlapIO(boolean overlap) {
            this.overlapIO = overlap;
            return this;
        }

        public Builder useGPU(boolean gpu) {
            this.useGPU = gpu;
            return this;
        }

        public Builder temperature(float temp) {
            this.temperature = temp;
            return this;
        }

        public StreamingConfig build() {
            return new StreamingConfig(this);
        }
    }
}
