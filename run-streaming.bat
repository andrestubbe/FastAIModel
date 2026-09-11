@echo off
setlocal enabledelayedexpansion
title FastAIModel - AIR-Style Layer Streaming Demo

echo [*] Building fastaimodel-streaming and StreamingDemo...
call mvn clean package -pl fastaimodel-streaming,examples/StreamingDemo -DskipTests -q
if %errorlevel% neq 0 (
    echo [-] Build failed!
    pause
    exit /b 1
)

echo [*] Starting StreamingDemo with strict 2 GB memory cap (-Xmx2g)...
set CP=fastaimodel-streaming\target\fastaimodel-streaming-0.1.4.jar;examples\StreamingDemo\target\StreamingDemo-0.1.0.jar;%USERPROFILE%\.m2\repository\com\github\andrestubbe\FastPointer\0.1.1\FastPointer-0.1.1.jar;%USERPROFILE%\.m2\repository\com\github\andrestubbe\FastMemory\0.1.1\FastMemory-0.1.1.jar;%USERPROFILE%\.m2\repository\com\github\andrestubbe\FastSIMD\0.1.3\FastSIMD-0.1.3.jar;%USERPROFILE%\.m2\repository\com\github\andrestubbe\FastGPU\0.1.1\fastgpu-0.1.1.jar;%USERPROFILE%\.m2\repository\com\github\andrestubbe\FastSharedMemory\0.1.2\FastSharedMemory-0.1.2.jar

java -Xmx2g -cp "%CP%" fastaimodel.demo.StreamingDemo

echo.
pause
