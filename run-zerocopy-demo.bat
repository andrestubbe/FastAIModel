@echo off
echo ==================================================
echo ⚡ FastAIModel & FastSharedMemory Zero-Copy Demo
echo ==================================================
set /p MODEL="Enter GGUF model path or Ollama model name (default: qwen2.5:0.5b): "
if "%MODEL%"=="" set MODEL=qwen2.5:0.5b

echo.
echo Launching Zero-Copy Shared Memory GPU Inference for model "%MODEL%"...
echo.

java --enable-native-access=ALL-UNNAMED -cp "fastaimodel-llama/target/test-classes;fastaimodel-llama/target/classes;lib/*;%USERPROFILE%/.m2/repository/com/github/andrestubbe/FastSharedMemory/0.1.2/FastSharedMemory-0.1.2.jar;%USERPROFILE%/.m2/repository/com/github/andrestubbe/FastPointer/0.1.1/FastPointer-0.1.1.jar;%USERPROFILE%/.m2/repository/com/github/andrestubbe/FastCore/0.1.0/FastCore-0.1.0.jar" fastaimodel.benchmark.ZeroCopyIpcDemo "%MODEL%"

pause
