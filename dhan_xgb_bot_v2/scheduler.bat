@echo off
REM ============================================================
REM  scheduler.bat  —  Windows Task Scheduler setup
REM  Bot folder : E:\TradingBot\dhan_xgb_bot_v2\dhan_xgb_bot_v2
REM  Venv       : E:\TradingBot\dhan_xgb_bot_v2\dhan_xgb_bot_v2\venv
REM ============================================================
REM Right-click this file -> Run as administrator
REM ============================================================

SET BOT_DIR=E:\TradingBot\dhan_xgb_bot_v2\dhan_xgb_bot_v2
SET PYTHON=%BOT_DIR%\venv\Scripts\python.exe

cls
echo ============================================================
echo  Trading Bot -- Task Scheduler Setup
echo ============================================================
echo  Bot folder : %BOT_DIR%
echo  Python     : %PYTHON%
echo ============================================================
echo.

REM Verify python exists
IF NOT EXIST "%PYTHON%" (
    echo [ERROR] Python not found at: %PYTHON%
    echo         Run: cd %BOT_DIR% then python -m venv venv
    pause
    exit /b 1
)
echo [OK] Python found.
echo.

REM Delete ALL old tasks cleanly
echo Removing old tasks...
schtasks /delete /tn "TradingBot_TokenRefresh"      /f 2>nul
schtasks /delete /tn "TradingBot_HealthCheck"       /f 2>nul
schtasks /delete /tn "TradingBot_LiveBot"            /f 2>nul
schtasks /delete /tn "TradingBot_InstrumentRefresh"  /f 2>nul
schtasks /delete /tn "TradingBot_WeeklyRetrain"      /f 2>nul
echo Done.
echo.

REM ============================================================
REM  KEY FIX: Use cmd.exe as the program, pass python -m as arg
REM  This ensures working directory is set before python runs
REM  /d flag on cd ensures it works across drives (E: drive)
REM ============================================================

REM Task 1: Token refresh 8:55 AM daily
schtasks /create /tn "TradingBot_TokenRefresh" ^
  /tr "cmd.exe /c cd /d \"%BOT_DIR%\" && \"%PYTHON%\" -m bot.token_refresh >> logs\token_refresh.log 2>&1" ^
  /sc daily /st 08:55 /ru "%USERNAME%" /f
IF %ERRORLEVEL%==0 (echo [OK] Task 1 -- Token refresh 8:55 AM daily) ELSE (echo [FAIL] Task 1)

REM Task 2: Health check 9:00 AM daily
schtasks /create /tn "TradingBot_HealthCheck" ^
  /tr "cmd.exe /c cd /d \"%BOT_DIR%\" && \"%PYTHON%\" -m bot.health_check >> logs\health_check.log 2>&1" ^
  /sc daily /st 09:00 /ru "%USERNAME%" /f
IF %ERRORLEVEL%==0 (echo [OK] Task 2 -- Health check 9:00 AM daily) ELSE (echo [FAIL] Task 2)

REM Task 3: Live bot 9:10 AM daily
schtasks /create /tn "TradingBot_LiveBot" ^
  /tr "cmd.exe /c cd /d \"%BOT_DIR%\" && \"%PYTHON%\" -m bot.live_bot >> logs\live_bot.log 2>&1" ^
  /sc daily /st 09:10 /ru "%USERNAME%" /f
IF %ERRORLEVEL%==0 (echo [OK] Task 3 -- Live bot 9:10 AM daily) ELSE (echo [FAIL] Task 3)

REM Task 4: Instrument refresh Sunday 7:30 PM
schtasks /create /tn "TradingBot_InstrumentRefresh" ^
  /tr "cmd.exe /c cd /d \"%BOT_DIR%\" && \"%PYTHON%\" -m data.load_instruments >> logs\instruments.log 2>&1" ^
  /sc weekly /d SUN /st 19:30 /ru "%USERNAME%" /f
IF %ERRORLEVEL%==0 (echo [OK] Task 4 -- Instrument refresh Sunday 7:30 PM) ELSE (echo [FAIL] Task 4)

REM Task 5: Weekly retrain Sunday 8:00 PM
schtasks /create /tn "TradingBot_WeeklyRetrain" ^
  /tr "cmd.exe /c cd /d \"%BOT_DIR%\" && \"%PYTHON%\" -m bot.auto_retrain >> logs\retrain.log 2>&1" ^
  /sc weekly /d SUN /st 20:00 /ru "%USERNAME%" /f
IF %ERRORLEVEL%==0 (echo [OK] Task 5 -- Weekly retrain Sunday 8:00 PM) ELSE (echo [FAIL] Task 5)

echo.
echo ============================================================
echo  Done. Verify in Task Scheduler (search in Start Menu)
echo  All 5 tasks should show status: Ready
echo ============================================================
echo.
echo  To test immediately (right-click task, Run):
echo    schtasks /run /tn "TradingBot_HealthCheck"
echo.
echo  To check logs after running:
echo    type %BOT_DIR%\logs\health_check.log
echo    type %BOT_DIR%\logs\live_bot.log
echo.
echo  IMPORTANT: Windows must NOT sleep during market hours
echo  Settings - Power - Sleep - Never (plugged in)
echo ============================================================
echo.
pause