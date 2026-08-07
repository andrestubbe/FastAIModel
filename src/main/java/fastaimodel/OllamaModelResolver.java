package fastaimodel;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class OllamaModelResolver {

    private OllamaModelResolver() {}

    public static String resolve(String modelName) {
        if (modelName == null || modelName.isEmpty()) return null;
        if (modelName.startsWith("ollama:")) {
            modelName = modelName.substring("ollama:".length());
        }

        if (modelName.contains("?")) {
            modelName = modelName.split("\\?", 2)[0];
        }

        File directFile = new File(modelName);
        if (directFile.exists() && directFile.isFile()) {
            return directFile.getAbsolutePath();
        }

        String userHome = System.getProperty("user.home");
        Path ollamaPath = Paths.get(userHome, ".ollama", "models");
        if (!Files.exists(ollamaPath)) {
            return modelName;
        }

        String tag = "latest";
        String name = modelName;
        if (modelName.contains(":")) {
            String[] parts = modelName.split(":", 2);
            name = parts[0];
            tag = parts[1];
        }

        Path manifestPath = ollamaPath.resolve(Paths.get("manifests", "registry.ollama.ai", "library", name, tag));
        if (!Files.exists(manifestPath)) {
            // Check direct manifests subfolder
            manifestPath = ollamaPath.resolve(Paths.get("manifests", name, tag));
        }

        if (Files.exists(manifestPath)) {
            try {
                String json = Files.readString(manifestPath);
                Pattern pattern = Pattern.compile("\"digest\"\\s*:\\s*\"sha256:([a-f0-9]+)\"");
                Matcher matcher = pattern.matcher(json);
                while (matcher.find()) {
                    String hash = matcher.group(1);
                    Path blobPath = ollamaPath.resolve(Paths.get("blobs", "sha256-" + hash));
                    if (Files.exists(blobPath) && Files.size(blobPath) > 10 * 1024 * 1024) { // GGUF model layer is larger than 10MB
                        return blobPath.toAbsolutePath().toString();
                    }
                }
            } catch (Exception ignored) {}
        }

        return modelName;
    }
}
