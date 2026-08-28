@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Navigate to the project directory
cd /d "%~dp0"

set "PYTHON_CMD=py -3"
%PYTHON_CMD% --version >nul 2>nul
if errorlevel 1 (
    set "PYTHON_CMD=python"
    %PYTHON_CMD% --version >nul 2>nul
    if errorlevel 1 (
        set "PYTHON_CMD="
    )
)

set "DOCS_ENABLED=1"
call :check_command doxygen Doxygen
call :check_command pandoc Pandoc
if "%PYTHON_CMD%"=="" (
    echo Warning: Python 3 not found. HTML documentation will be skipped.
    set "DOCS_ENABLED=0"
) else (
    call :check_python_module sphinx
    call :check_python_module sphinx_rtd_theme
    call :check_python_module nbsphinx
    call :check_python_module breathe
)

rem Create a new build directory
if not exist build mkdir build

rem Navigate to the build directory
cd build

set CONFIG=Release
if /I "%~1"=="--debug" set CONFIG=Debug

call :detect_vs_generator
if not defined CMAKE_GENERATOR (
    echo Error: Could not detect a supported Visual Studio installation for CMake.
    exit /b 1
)

echo Using CMake generator: %CMAKE_GENERATOR%
set "CURRENT_GENERATOR="
set "CURRENT_GENERATOR_PLATFORM="
for /f "tokens=2 delims==" %%G in ('findstr /b "CMAKE_GENERATOR:INTERNAL=" CMakeCache.txt 2^>nul') do (
    set "CURRENT_GENERATOR=%%G"
)
for /f "tokens=2 delims==" %%G in ('findstr /b "CMAKE_GENERATOR_PLATFORM:INTERNAL=" CMakeCache.txt 2^>nul') do (
    set "CURRENT_GENERATOR_PLATFORM=%%G"
)
if defined CURRENT_GENERATOR if /I not "!CURRENT_GENERATOR!"=="%CMAKE_GENERATOR%" (
    echo Existing build directory uses "!CURRENT_GENERATOR!". Resetting build directory...
    call :reset_build_dir
)
if defined CURRENT_GENERATOR if /I "!CURRENT_GENERATOR!"=="%CMAKE_GENERATOR%" if /I not "!CURRENT_GENERATOR_PLATFORM!"=="x64" (
    echo Existing build directory uses platform "!CURRENT_GENERATOR_PLATFORM!". Resetting build directory...
    call :reset_build_dir
)
call :reset_fetchcontent_subbuilds

cmake -G "%CMAKE_GENERATOR%" -A x64 ^
  -DMINION_BUILD_BENCHMARK=ON ^
  -DMINION_BUILD_EXAMPLES=ON ^
  -DMINION_BUILD_PYTHON=ON ^
  ..
if errorlevel 1 exit /b 1

rem Build the project using MSBuild
cmake --build . --config %CONFIG% -j 8
if errorlevel 1 exit /b 1

rem Move back to repository root
cd ..

rem Run Doxygen if documentation dependencies are installed
if "%DOCS_ENABLED%"=="1" (
    echo Generating documentation with Doxygen...
    doxygen Doxyfile
)

rem Build HTML documentation
if "%DOCS_ENABLED%"=="1" (
if exist docs (
    echo Building HTML documentation...
    pushd docs
    if exist make.bat (
        call make.bat clean
        call make.bat html
    ) else (
        echo Warning: docs\\make.bat not found. Skipping HTML documentation.
    )
    popd
) else (
    echo Warning: 'docs' folder not found. Skipping HTML documentation.
)
) else (
    echo Skipping documentation because one or more documentation dependencies are missing.
)

echo Build complete!
exit /b 0

:detect_vs_generator
set "CMAKE_GENERATOR="
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" set "VSWHERE=%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo Error: vswhere.exe not found. Install Visual Studio Installer components.
    exit /b 1
)

set "VS_MAJOR="
for /f "usebackq delims=" %%G in (`"%VSWHERE%" -latest -products * -requires Microsoft.Component.MSBuild -property installationVersion`) do (
    set "VS_VERSION=%%G"
)
for /f "tokens=1 delims=." %%G in ("%VS_VERSION%") do (
    set "VS_MAJOR=%%G"
)

if "%VS_MAJOR%"=="18" set "CMAKE_GENERATOR=Visual Studio 18 2026"
if "%VS_MAJOR%"=="17" set "CMAKE_GENERATOR=Visual Studio 17 2022"
if "%VS_MAJOR%"=="16" set "CMAKE_GENERATOR=Visual Studio 16 2019"
if "%VS_MAJOR%"=="15" set "CMAKE_GENERATOR=Visual Studio 15 2017"
exit /b 0

:reset_build_dir
for /f "delims=" %%G in ('dir /b /a') do (
    if /I not "%%G"=="." if /I not "%%G"==".." (
        attrib -r -s -h "%%G" /s /d >nul 2>nul
        if exist "%%G\" (
            rmdir /s /q "%%G"
        ) else (
            del /f /q "%%G"
        )
    )
)
exit /b 0

:reset_fetchcontent_subbuilds
if not exist "_deps" exit /b 0
for /d %%G in ("_deps\*-subbuild") do (
    if exist "%%~G\CMakeCache.txt" (
        echo Resetting FetchContent sub-build cache: %%~nxG
        rmdir /s /q "%%~G"
    )
)
exit /b 0

:check_command
where %~1 >nul 2>nul
if not errorlevel 1 exit /b 0

echo Warning: %~2 not found. Documentation generation will be skipped.
set "DOCS_ENABLED=0"
exit /b 0

:check_python_module
%PYTHON_CMD% -c "import %~1" >nul 2>nul
if not errorlevel 1 exit /b 0

echo Warning: Python module %~1 not found. Documentation generation will be skipped.
set "DOCS_ENABLED=0"
exit /b 0
