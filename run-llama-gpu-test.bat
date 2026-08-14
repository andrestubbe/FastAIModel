@echo off
setlocal EnableDelayedExpansion

cd /d "%~dp0"

echo ==================================================
echo FastAIModel Llama CPU vs Intel Iris GPU Comparison
echo ==================================================

set "JAVA_HOME=C:\Program Files\Java\jdk-25.0.3"
if not defined JAVA_HOME (
    set "JAVA_HOME=C:\Program Files\Java\jdk-17"
)

set "DEFAULT_MODEL=c:/Users/andre/Documents/2026-06-14-Work-FastJava/FastAI/examples/Demo/models/Llama-3.2-1B-Instruct-Q8_0.gguf"

set /p MODEL_NAME="Enter GGUF model path or Ollama model name (default: Llama-3.2-1B): "
if "%MODEL_NAME%"=="" (
    set "MODEL_NAME=%DEFAULT_MODEL%"
)

echo.
echo Running CPU vs Intel Iris GPU comparison for model...
echo.

set "CLASSPATH=target/test-classes;fastaimodel-llama/target/classes;lib/fastaimodel-llama.jar;lib/fastcore-0.1.0.jar;lib/fastgpu-0.1.1.jar"

"%JAVA_HOME%\bin\java.exe" --enable-native-access=ALL-UNNAMED -cp "%CLASSPATH%" -Djava.library.path="lib" fastaimodel.benchmark.InteractiveOllamaTest "%MODEL_NAME%"

pause
