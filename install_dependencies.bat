@echo off
setlocal enabledelayedexpansion

echo setting up python environment ...
python -m venv python_env
cd .\python_env 
cd Scripts 
call Activate 
echo current vev : 
echo %VIRTUAL_ENV%
cd ..\..
echo done


echo Installing dependencies ...    	
pip install -r Requirements.txt
echo done 
pause

