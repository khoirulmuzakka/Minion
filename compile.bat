@echo off

rem Navigate to the project directory
cd /d "%~dp0"

rem Remove the existing directories and files
rd /s /q dist
rd /s /q build
rd /s /q *.egg-info
rd /s /q minionpy\lib
rd /s /q lib

rem Create a new build directory
if not exist build mkdir build

rem Navigate to the build directory
cd build

rem Run CMake to configure the project with Visual Studio 2022 generator
cmake -G "Visual Studio 17 2022" ..

rem Build the project using MSBuild
cmake --build . --clean-first --config Release

rem Pause to see build output (optional)
pause
