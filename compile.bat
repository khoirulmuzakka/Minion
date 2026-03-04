@echo off

rem Navigate to the project directory
cd /d "%~dp0"


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

rem Run Doxygen if installed
where doxygen >nul 2>nul
if %ERRORLEVEL%==0 (
    echo Generating documentation with Doxygen...
    doxygen Doxyfile
) else (
    echo Warning: Doxygen not found. Skipping documentation generation.
)

rem Build HTML documentation
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

echo Build complete!
