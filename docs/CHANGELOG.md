# FastAIModel Changelog

## [0.1.2] - 2026-08-10
### Changed
- **Multi-Module Refactoring**: Split project into modular artifacts:
  - `fastaimodel-onnx`: Pure ONNX Runtime execution engine without C++ Llama DLL dependencies.
  - `fastaimodel-llama`: Native `llama.cpp` C++ GGUF inference engine backed by `FastCore`.
- Updated parent POM to aggregate both sub-modules (`<packaging>pom</packaging>`).

## [0.1.1] - 2026-06-14
### Added
- Native `llama.cpp` JNI bindings for local GGUF model execution.
- `FastAIOnnxModel` wrapper for ONNX sessions.

## [0.1.0-ALPHA] - 2026-06-13
### Added
- Initial JNI interface structure for local model loading and token generation.
