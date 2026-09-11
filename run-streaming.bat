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
echo.
java -Xmx2g -cp "fastaimodel-streaming/target/*;examples/StreamingDemo/target/*;examples/StreamingDemo/target/lib/*" fastaimodel.demo.StreamingDemo

echo.
pause
