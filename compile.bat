@echo off
setlocal EnableDelayedExpansion

cd /d "%~dp0"

echo ===========================================
set "JAVA_HOME=C:\Program Files\Java\jdk-25"
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
echo Generating llama.lib from llama.def...
lib /def:llama.def /out:build\llama.lib /machine:x64

echo.
echo Compiling FastAIModel JNI Bridge (C++)...
cl /LD /Fe:build\fastaimodel.dll /Fo:build\ ^
    native\fastaimodel.cpp ^
    /I"%JAVA_HOME%\include" ^
    /I"%JAVA_HOME%\include\win32" ^
    /I"native\llama" ^
    /EHsc /std:c++17 /O2 /W3 ^
    /link /DEF:native\fastaimodel.def build\llama.lib

if %errorlevel% neq 0 (
    echo C++ COMPILATION FAILED
    exit /b 1
)

echo.
echo Copying DLL to build folder...
copy /Y build\fastaimodel.dll build\fastaimodel.dll

del /Q build\*.obj
del /Q build\*.exp

echo Build successful!
