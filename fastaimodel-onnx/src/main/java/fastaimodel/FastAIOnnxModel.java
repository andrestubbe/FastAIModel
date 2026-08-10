package fastaimodel;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtException;
import java.util.Map;

public class FastAIOnnxModel implements AutoCloseable {
    private final OrtEnvironment env;
    private final OrtSession session;

    public FastAIOnnxModel(String modelPath) {
        try {
            this.env = OrtEnvironment.getEnvironment();
            this.session = env.createSession(modelPath, new OrtSession.SessionOptions());
        } catch (OrtException e) {
            throw new RuntimeException("Failed to load ONNX model: " + modelPath, e);
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
