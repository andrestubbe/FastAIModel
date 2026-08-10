@echo off
setlocal EnableDelayedExpansion

cd /d "%~dp0"

echo ===========================================
set "JAVA_HOME=C:\Program Files\Java\jdk-25.0.3"
if not defined JAVA_HOME (
    set "JAVA_HOME=C:\Program Files\Java\jdk-17"
)

:: Find VS
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
    set "VS_INSTALL=%%i"
)
set "VCVARS=%VS_INSTALL%\VC\Auxiliary\Build\vcvars64.bat"
call "%VCVARS%"

if not exist build mkdir build

echo.
echo Generating llama.lib, ggml.lib, ggml-base.lib...
dumpbin /exports lib\llama.dll > build\llama.exports
echo EXPORTS > build\llama.def
for /f "tokens=4" %%a in ('findstr /R /C:"^[ ]*[0-9]" build\llama.exports') do (
    echo %%a >> build\llama.def
)
lib /def:build\llama.def /out:build\llama.lib /machine:x64

dumpbin /exports lib\ggml.dll > build\ggml.exports
echo EXPORTS > build\ggml.def
for /f "tokens=4" %%a in ('findstr /R /C:"^[ ]*[0-9]" build\ggml.exports') do (
    echo %%a >> build\ggml.def
)
lib /def:build\ggml.def /out:build\ggml.lib /machine:x64

dumpbin /exports lib\ggml-base.dll > build\ggml-base.exports
echo EXPORTS > build\ggml-base.def
for /f "tokens=4" %%a in ('findstr /R /C:"^[ ]*[0-9]" build\ggml-base.exports') do (
    echo %%a >> build\ggml-base.def
)
lib /def:build\ggml-base.def /out:build\ggml-base.lib /machine:x64

echo.
echo Compiling FastAIModel JNI Bridge (C++)...
cl /LD /Fe:build\fastaimodel.dll /Fo:build\ ^
    native\fastaimodel.cpp ^
    /I"%JAVA_HOME%\include" ^
    /I"%JAVA_HOME%\include\win32" ^
    /I"native\llama" ^
    /EHsc /std:c++17 /O2 /W3 ^
    /link /DEF:native\fastaimodel.def build\llama.lib build\ggml.lib build\ggml-base.lib

if %errorlevel% neq 0 (
    echo C++ COMPILATION FAILED
    exit /b 1
)

echo.
echo Copying DLL to lib and target folders...
copy /Y build\fastaimodel.dll lib\fastaimodel.dll
if exist fastaimodel-llama\target\classes (
    if not exist fastaimodel-llama\target\classes\native mkdir fastaimodel-llama\target\classes\native
    copy /Y build\fastaimodel.dll fastaimodel-llama\target\classes\native\fastaimodel.dll
)

del /Q build\*.obj
del /Q build\*.exp

echo Build successful!
