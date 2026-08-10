@echo off
cd /d "%~dp0"
set PATH=%~dp0lib;%PATH%
echo Running FastAIModel GGUF Silent Demo...
set "MODEL_ARG=%~1"
if not "%MODEL_ARG%"=="" (
    for %%i in ("%MODEL_ARG%") do set "MODEL_ARG=%%~fi"
)
chcp 65001 >nul
cd fastaimodel-llama
call mvn test-compile >nul 2>&1
mvn -q exec:exec "-Dexec.executable=java" "-Dexec.workingdir=../lib" "-Dexec.classpathScope=test" "-Dexec.args=-Dmodel.path=\"%MODEL_ARG%\" -Dfile.encoding=UTF-8 -cp %%classpath fastaimodel.GgufDemo"
cd ..
pause
