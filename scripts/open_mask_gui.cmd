@echo off
setlocal
cd /d "%~dp0.."

set "ASTROSEG_PYTHON=.python311\python.exe"
if not exist "%ASTROSEG_PYTHON%" (
  echo Could not find %ASTROSEG_PYTHON%.
  echo Install the project environment first, or run launch_mask_review_gui.py manually.
  pause
  exit /b 1
)

"%ASTROSEG_PYTHON%" scripts\launch_mask_review_gui.py ^
  --images .astroseg_gui\comparison\images ^
  --masks "Cyto2=.astroseg_gui\comparison\masks\cyto2" ^
  --masks "Cyto3=.astroseg_gui\comparison\masks\cyto3" ^
  --masks "New 3-channel=.astroseg_gui\comparison\masks\three_channel" ^
  --masks "AstroSeg v2=.astroseg_gui\comparison\masks\astroseg_v2" ^
  --ground-truth .astroseg_gui\comparison\ground_truth ^
  --corrections .astroseg_gui\comparison\corrections

if errorlevel 1 pause
