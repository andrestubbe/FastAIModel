@echo off
cd /d "%~dp0"
set PATH=%~dp0lib;%PATH%
echo Running FastAIModel ONNX Demo...
mvn test-compile exec:exec "-Dexec.executable=java" "-Dexec.workingdir=lib" "-Dexec.classpathScope=test" "-Dexec.args=-cp %%classpath fastaimodel.OnnxDemo"
pause
