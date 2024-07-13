@echo off
setlocal enabledelayedexpansion

rem Change to the AutoNRA directory
cd %~dp0

echo Setting up python environment ...
cd .\python_env 
cd Scripts 
call Activate 
echo done.

cd ..\..

echo compiling ..
python Setup.py build_ext --inplace
echo compilation done 

pause
