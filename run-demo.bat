@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo [1/3] Building FastAIModel modules...
call mvn clean install -DskipTests -q
if %errorlevel% neq 0 ( echo [ERROR] Build failed! & pause & exit /b 1 )

echo [2/3] Compiling Streaming Demo...
cd examples\StreamingDemo
call mvn compile -q
if %errorlevel% neq 0 ( echo [ERROR] Demo compile failed! & pause & exit /b 1 )

echo [3/3] Running Streaming Demo...
call mvn exec:java -Dexec.mainClass=fastaimodel.streaming.StreamingDemo -q %*

cd ..\..
pause
