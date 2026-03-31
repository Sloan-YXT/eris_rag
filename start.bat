@echo off
cd /d "%~dp0"
:loop
echo [%date% %time%] Server starting... >> server.log
python -m src.server >> server.log 2>&1
echo [%date% %time%] Server exited (code=%ERRORLEVEL%), restarting in 5s... >> server.log
timeout /t 5 /nobreak >nul
goto loop
