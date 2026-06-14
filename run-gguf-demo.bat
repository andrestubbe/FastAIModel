@echo off
cd /d "%~dp0"
set PATH=%~dp0lib;%PATH%
echo Running FastAIModel GGUF Demo...
mvn test-compile exec:java "-Dexec.mainClass=fastaimodel.Demo" "-Dexec.classpathScope=test" "-Dexec.vmArgs=-Djava.library.path=lib"
pause
