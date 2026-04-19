@echo off
REM ForgeShield — start both services

echo [ForgeShield] Starting Python detector API on port 8000...
start "ForgeShield-Detector" cmd /k "cd /d %~dp0detector && python detector_api.py"

echo [ForgeShield] Waiting 5 seconds for Python API to initialise...
timeout /t 5 /nobreak >nul

echo [ForgeShield] Starting Node.js backend on port 5000...
start "ForgeShield-Backend" cmd /k "cd /d %~dp0backend && npm start"

echo.
echo [ForgeShield] Both services started.
echo   Backend  : http://localhost:5000/api/health
echo   Detector : http://localhost:8000/health
echo.
pause
