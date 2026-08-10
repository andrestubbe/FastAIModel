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
                // Look specifically for model layer
                Pattern pattern = Pattern.compile("\"mediaType\"\\s*:\\s*\"application/vnd.ollama.image.model\"\\s*,\\s*\"digest\"\\s*:\\s*\"sha256:([a-f0-9]+)\"");
                Matcher matcher = pattern.matcher(json);
                if (matcher.find()) {
                    String hash = matcher.group(1);
                    Path blobPath = ollamaPath.resolve(Paths.get("blobs", "sha256-" + hash));
                    if (Files.exists(blobPath)) {
                        String rawPath = blobPath.toAbsolutePath().toString().replace('\\', '/');
                        try {
                            Path ggufAlias = Paths.get(System.getProperty("java.io.tmpdir"), "ollama_" + hash.substring(0, 12) + ".gguf");
                            if (!Files.exists(ggufAlias)) {
                                try {
                                    Files.createSymbolicLink(ggufAlias, blobPath);
                                } catch (Throwable t) {
                                    try {
                                        Files.createLink(ggufAlias, blobPath);
                                    } catch (Throwable t2) {
                                        return rawPath;
                                    }
                                }
                            }
                            return ggufAlias.toAbsolutePath().toString().replace('\\', '/');
                        } catch (Throwable ignored) {
                            return rawPath;
                        }
                    }
                }
                // Fallback to any blob > 10MB
                Pattern anyPattern = Pattern.compile("\"digest\"\\s*:\\s*\"sha256:([a-f0-9]+)\"");
                Matcher anyMatcher = anyPattern.matcher(json);
                while (anyMatcher.find()) {
                    String hash = anyMatcher.group(1);
                    Path blobPath = ollamaPath.resolve(Paths.get("blobs", "sha256-" + hash));
                    if (Files.exists(blobPath) && Files.size(blobPath) > 10 * 1024 * 1024) { // GGUF model layer > 10MB
                        String rawPath = blobPath.toAbsolutePath().toString().replace('\\', '/');
                        // llama.cpp requires .gguf extension on some platforms/versions
                        try {
                            Path ggufAlias = Paths.get(System.getProperty("java.io.tmpdir"), "ollama_" + hash.substring(0, 12) + ".gguf");
                            if (!Files.exists(ggufAlias)) {
                                try {
                                    Files.createSymbolicLink(ggufAlias, blobPath);
                                } catch (Throwable t) {
                                    // Fallback to hardlink or copy
                                    try {
                                        Files.createLink(ggufAlias, blobPath);
                                    } catch (Throwable t2) {
                                        return rawPath;
                                    }
                                }
                            }
                            return ggufAlias.toAbsolutePath().toString().replace('\\', '/');
                        } catch (Throwable ignored) {
                            return rawPath;
                        }
                    }
                }
            } catch (Exception ignored) {}
        }

        return modelName.replace('\\', '/');
    }
}
