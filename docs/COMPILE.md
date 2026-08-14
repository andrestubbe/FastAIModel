# FastAIModel Compilation Guide

## Requirements

- **Windows**: Visual Studio 2022/2026 Developer Command Prompt (`x64`), Vulkan SDK.
- **macOS**: Xcode Command Line Tools (`xcode-select --install`).
- **Java**: JDK 17+ (Java 21 recommended).

---

## Build Steps (Windows)

```cmd
compile.bat
mvn install -DskipTests
```

Produces `build/fastaimodel.dll` and installs Maven artifacts to local repository.

---

## Build Steps (macOS)

```bash
./compile.sh
mvn install -DskipTests
```

Produces `build/libfastaimodel.dylib` with Metal GPU acceleration.
