package fastaimodel.streaming.io;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class ModelResolver {

    private ModelResolver() {}

    public static File resolve(String modelIdentifier) {
        if (modelIdentifier == null || modelIdentifier.isEmpty()) return null;

        File direct = new File(modelIdentifier);
        if (direct.exists() && direct.isFile()) {
            return direct;
        }

        String name = modelIdentifier;
        if (name.startsWith("ollama:")) {
            name = name.substring("ollama:".length());
        }

        String tag = "latest";
        if (name.contains(":")) {
            String[] parts = name.split(":", 2);
            name = parts[0];
            tag = parts[1];
        }

        String userHome = System.getProperty("user.home");
        Path ollamaPath = Paths.get(userHome, ".ollama", "models");
        if (Files.exists(ollamaPath)) {
            // Try standard library manifest path first
            Path manifestPath = ollamaPath.resolve(Paths.get("manifests", "registry.ollama.ai", "library", name, tag));
            if (!Files.exists(manifestPath)) {
                // Try direct registry manifest path (e.g. openbmb/minicpm5)
                manifestPath = ollamaPath.resolve(Paths.get("manifests", "registry.ollama.ai", name, tag));
            }
            if (!Files.exists(manifestPath)) {
                manifestPath = ollamaPath.resolve(Paths.get("manifests", name, tag));
            }

            if (Files.exists(manifestPath)) {
                try {
                    String json = Files.readString(manifestPath);
                    Pattern pattern = Pattern.compile("\"mediaType\"\\s*:\\s*\"application/vnd.ollama.image.model\"\\s*,\\s*\"digest\"\\s*:\\s*\"sha256:([a-f0-9]+)\"");
                    Matcher matcher = pattern.matcher(json);
                    if (matcher.find()) {
                        String hash = matcher.group(1);
                        Path blobPath = ollamaPath.resolve(Paths.get("blobs", "sha256-" + hash));
                        if (Files.exists(blobPath)) {
                            return blobPath.toFile();
                        }
                    }
                } catch (Throwable ignored) {}
            }
        }

        return direct;
    }

    public static java.util.List<String> listInstalledModels() {
        java.util.List<String> list = new java.util.ArrayList<>();
        String userHome = System.getProperty("user.home");
        Path manifestsBase = Paths.get(userHome, ".ollama", "models", "manifests");
        if (!Files.exists(manifestsBase)) return list;

        try (java.util.stream.Stream<Path> stream = Files.walk(manifestsBase)) {
            stream.filter(Files::isRegularFile).forEach(p -> {
                Path rel = manifestsBase.relativize(p);
                String s = rel.toString().replace('\\', '/');
                if (s.startsWith("registry.ollama.ai/")) {
                    s = s.substring("registry.ollama.ai/".length());
                }
                if (s.startsWith("library/")) {
                    s = s.substring("library/".length());
                }
                int lastSlash = s.lastIndexOf('/');
                if (lastSlash != -1) {
                    list.add(s.substring(0, lastSlash) + ":" + s.substring(lastSlash + 1));
                } else {
                    list.add(s);
                }
            });
        } catch (Throwable ignored) {}
        java.util.Collections.sort(list);
        return list;
    }
}
