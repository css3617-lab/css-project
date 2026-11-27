@echo off
setlocal
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

rem ---- find Python 3 ------------------------------------------------------
set PY=
where py >nul 2>nul && set PY=py -3
if "%PY%"=="" ( where python >nul 2>nul && set PY=python )
if "%PY%"=="" (
  echo [ERROR] Python 3 was not found. Install Python 3.10+ from https://www.python.org/downloads/
  echo         and tick "Add python.exe to PATH" during installation, then run this file again.
  pause
  exit /b 1
)
%PY% --version

rem ---- packages -----------------------------------------------------------
%PY% -m pip install --upgrade pip >nul
%PY% -m pip install -r requirements.txt
if errorlevel 1 ( echo [ERROR] package installation failed & pause & exit /b 1 )

rem ---- data (WS-DREAM #1 is bundled; #2 is downloaded from Zenodo) -------
%PY% download_data.py

rem ---- experiments (each one resumes where it stopped if interrupted) -----
if not exist results mkdir results
echo. > results\run_log.txt
for %%E in (e1_e2_benchmark e3_exact e4_structured e5_sensitivity e6_largescale e7_dynamic e8_negotiation) do (
  echo ================= %%E =================
  echo ================= %%E ================= >> results\run_log.txt
  %PY% %%E.py >> results\run_log.txt 2>&1
  if errorlevel 1 ( echo [WARN] %%E reported an error, see results\run_log.txt )
)

%PY% summarize.py > results\summary.txt 2>&1
%PY% -c "import platform,os;open('results/machine.txt','w').write(platform.platform()+'\n'+platform.processor()+'\ncpus='+str(os.cpu_count())+'\npython='+platform.python_version()+'\n')"

rem ---- pack results -------------------------------------------------------
if exist results.zip del results.zip
powershell -NoProfile -Command "Compress-Archive -Path 'results\*' -DestinationPath 'results.zip' -Force"
echo.
echo Done. Please attach results.zip to the chat.
pause
