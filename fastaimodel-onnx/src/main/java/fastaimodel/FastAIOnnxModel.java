package fastaimodel;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtException;
import java.util.Map;

public class FastAIOnnxModel implements AutoCloseable {
    private final OrtEnvironment env;
    private final OrtSession session;

    public static FastAIOnnxModel open(String modelPath) {
        return new FastAIOnnxModel(modelPath);
    }

    public static Builder builder() {
        return new Builder();
    }

    public FastAIOnnxModel(String modelPath) {
        this(modelPath, new OrtSession.SessionOptions());
    }

    public FastAIOnnxModel(String modelPath, OrtSession.SessionOptions options) {
        try {
            this.env = OrtEnvironment.getEnvironment();
            this.session = env.createSession(modelPath, options != null ? options : new OrtSession.SessionOptions());
        } catch (OrtException e) {
            throw new RuntimeException("Failed to load ONNX model: " + modelPath, e);
        }
    }

    public static class Builder {
        private String modelPath;
        private int intraOpNumThreads = -1;
        private int interOpNumThreads = -1;

        public Builder model(String modelPath) {
            this.modelPath = modelPath;
            return this;
        }

        public Builder threads(int threads) {
            this.intraOpNumThreads = threads;
            return this;
        }

        public Builder intraOpThreads(int threads) {
            this.intraOpNumThreads = threads;
            return this;
        }

        public Builder interOpThreads(int threads) {
            this.interOpNumThreads = threads;
            return this;
        }

        public FastAIOnnxModel build() {
            if (modelPath == null || modelPath.isBlank()) {
                throw new IllegalArgumentException("modelPath must not be null or blank");
            }
            try {
                OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
                if (intraOpNumThreads > 0) {
                    opts.setIntraOpNumThreads(intraOpNumThreads);
                }
                if (interOpNumThreads > 0) {
                    opts.setInterOpNumThreads(interOpNumThreads);
                }
                return new FastAIOnnxModel(modelPath, opts);
            } catch (OrtException e) {
                throw new RuntimeException("Failed to configure ONNX session: " + modelPath, e);
            }
        }
    }

    public OrtSession getSession() {
        return session;
    }

    public OrtEnvironment getEnv() {
        return env;
    }

    public OrtSession.Result run(Map<String, OnnxTensor> inputs) {
        try {
            return session.run(inputs);
        } catch (OrtException e) {
            throw new RuntimeException("Failed to run ONNX inference", e);
        }
    }

    @Override
    public void close() {
        try {
            if (session != null) {
                session.close();
            }
            if (env != null) {
                env.close();
            }
        } catch (OrtException e) {
            System.err.println("Error closing ONNX resources: " + e.getMessage());
        }
    }
}
