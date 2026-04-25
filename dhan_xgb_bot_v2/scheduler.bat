@echo off
REM ============================================================
REM  scheduler.bat  —  Windows Task Scheduler setup
REM  Bot folder : E:\TradingBot\dhan_xgb_bot_v2\dhan_xgb_bot_v2
REM  Venv       : E:\TradingBot\dhan_xgb_bot_v2\dhan_xgb_bot_v2\venv
REM ============================================================
REM Right-click -> Run as administrator
REM ============================================================

SET BOT_DIR=E:\TradingBot\dhan_xgb_bot_v2\dhan_xgb_bot_v2
SET PYTHON=%BOT_DIR%\venv\Scripts\python.exe
SET LOGS=%BOT_DIR%\logs

cls
echo ============================================================
echo  Trading Bot -- Task Scheduler Setup
echo ============================================================
echo  Bot folder : %BOT_DIR%
echo  Python     : %PYTHON%
echo  Logs       : %LOGS%
echo ============================================================
echo.

REM Verify python exists
IF NOT EXIST "%PYTHON%" (
    echo [ERROR] Python not found at: %PYTHON%
    pause
    exit /b 1
)
echo [OK] Python found.

REM Create logs folder if missing
IF NOT EXIST "%LOGS%" mkdir "%LOGS%"
echo [OK] Logs folder ready.
echo.

REM Delete old tasks
echo Removing old tasks...
schtasks /delete /tn "TradingBot_TokenRefresh"      /f 2>nul
schtasks /delete /tn "TradingBot_HealthCheck"       /f 2>nul
schtasks /delete /tn "TradingBot_LiveBot"            /f 2>nul
schtasks /delete /tn "TradingBot_InstrumentRefresh"  /f 2>nul
schtasks /delete /tn "TradingBot_WeeklyRetrain"      /f 2>nul
echo Done.
echo.

REM ============================================================
REM  FIX: Use full absolute path for log files
REM  FIX: /rl HIGHEST ensures admin-level execution
REM  FIX: chcp 65001 sets UTF-8 so Rs symbol doesn't crash
REM ============================================================

REM Task 1: Token refresh 8:55 AM daily
schtasks /create /tn "TradingBot_TokenRefresh" ^
  /tr "cmd.exe /c chcp 65001 && cd /d \"%BOT_DIR%\" && \"%PYTHON%\" -m bot.token_refresh >> \"%LOGS%\token_refresh.log\" 2>&1" ^
  /sc daily /st 08:55 /ru "%USERNAME%" /rl HIGHEST /f
IF %ERRORLEVEL%==0 (echo [OK] Task 1 -- Token refresh 8:55 AM) ELSE (echo [FAIL] Task 1)

REM Task 2: Health check 9:00 AM daily
schtasks /create /tn "TradingBot_HealthCheck" ^
  /tr "cmd.exe /c chcp 65001 && cd /d \"%BOT_DIR%\" && \"%PYTHON%\" -m bot.health_check >> \"%LOGS%\health_check.log\" 2>&1" ^
  /sc daily /st 09:00 /ru "%USERNAME%" /rl HIGHEST /f
IF %ERRORLEVEL%==0 (echo [OK] Task 2 -- Health check 9:00 AM) ELSE (echo [FAIL] Task 2)

REM Task 3: Live bot 9:10 AM daily
schtasks /create /tn "TradingBot_LiveBot" ^
  /tr "cmd.exe /c chcp 65001 && cd /d \"%BOT_DIR%\" && \"%PYTHON%\" -m bot.live_bot >> \"%LOGS%\live_bot.log\" 2>&1" ^
  /sc daily /st 09:10 /ru "%USERNAME%" /rl HIGHEST /f
IF %ERRORLEVEL%==0 (echo [OK] Task 3 -- Live bot 9:10 AM) ELSE (echo [FAIL] Task 3)

REM Task 4: Instrument refresh Sunday 7:30 PM
schtasks /create /tn "TradingBot_InstrumentRefresh" ^
  /tr "cmd.exe /c chcp 65001 && cd /d \"%BOT_DIR%\" && \"%PYTHON%\" -m data.load_instruments >> \"%LOGS%\instruments.log\" 2>&1" ^
  /sc weekly /d SUN /st 19:30 /ru "%USERNAME%" /rl HIGHEST /f
IF %ERRORLEVEL%==0 (echo [OK] Task 4 -- Instrument refresh Sunday 7:30 PM) ELSE (echo [FAIL] Task 4)

REM Task 5: Weekly retrain Sunday 8:00 PM
schtasks /create /tn "TradingBot_WeeklyRetrain" ^
  /tr "cmd.exe /c chcp 65001 && cd /d \"%BOT_DIR%\" && \"%PYTHON%\" -m bot.auto_retrain >> \"%LOGS%\retrain.log\" 2>&1" ^
  /sc weekly /d SUN /st 20:00 /ru "%USERNAME%" /rl HIGHEST /f
IF %ERRORLEVEL%==0 (echo [OK] Task 5 -- Weekly retrain Sunday 8:00 PM) ELSE (echo [FAIL] Task 5)

echo.
echo ============================================================
echo  Done. Now TEST each task:
echo ============================================================
echo.
echo  Step 1: Open Task Scheduler (Win key, search Task Scheduler)
echo  Step 2: Find TradingBot_HealthCheck
echo  Step 3: Right-click - Run
echo  Step 4: Wait 30 seconds
echo  Step 5: Check result:
echo    schtasks /query /tn "TradingBot_HealthCheck" /fo LIST
echo    Look for: Last Run Result = 0x0 (success)
echo.
echo  Step 6: Check the log file:
echo    type "%LOGS%\health_check.log"
echo.
echo  If still (0x1), the log file will show the exact error.
echo.
echo  IMPORTANT: PC must NOT sleep during market hours
echo  Settings - Power - Sleep - Never (plugged in)
echo ============================================================
echo.
pause