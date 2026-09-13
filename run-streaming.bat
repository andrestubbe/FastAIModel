@echo off
setlocal enabledelayedexpansion
title FastAIModel - Zero-Copy Streaming

chcp 65001 >nul
set CP=fastaimodel-streaming\target\fastaimodel-streaming-0.1.7.jar;fastaimodel-llama\target\fastaimodel-llama-0.1.7.jar;..\FastCore\target\FastCore-0.1.0.jar;examples\StreamingDemo\target\StreamingDemo-0.1.0.jar;%USERPROFILE%\.m2\repository\com\github\andrestubbe\FastPointer\0.1.1\FastPointer-0.1.1.jar;%USERPROFILE%\.m2\repository\com\github\andrestubbe\FastMemory\0.1.1\FastMemory-0.1.1.jar;%USERPROFILE%\.m2\repository\com\github\andrestubbe\FastSIMD\0.1.3\FastSIMD-0.1.3.jar;%USERPROFILE%\.m2\repository\com\github\andrestubbe\FastGPU\0.1.1\fastgpu-0.1.1.jar;%USERPROFILE%\.m2\repository\com\github\andrestubbe\FastSharedMemory\0.1.2\FastSharedMemory-0.1.2.jar

java --add-modules jdk.incubator.vector --add-opens java.base/java.nio=ALL-UNNAMED --add-opens java.base/sun.misc=ALL-UNNAMED -Dfile.encoding=UTF-8 -Xmx2g -cp "%CP%" fastaimodel.demo.StreamingDemo %*

echo.
pause
