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

rem Run CMake to configure the project with Visual Studio 2022 generator
cmake -G "Visual Studio 17 2022" ^
  -DMINION_BUILD_CEC=ON ^
  -DMINION_BUILD_EXAMPLES=ON ^
  -DMINION_BUILD_PYTHON=ON ^
  ..

rem Build the project using MSBuild
set CONFIG=Release
if /I "%~1"=="--debug" set CONFIG=Debug
cmake --build . --config %CONFIG%

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
