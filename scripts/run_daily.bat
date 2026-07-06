@echo off
setlocal

REM 激活虚拟环境（如需要可取消注释）
REM call .venv\Scripts\activate.bat

cd /d "%~dp0\.."
python -m src.pipeline.train_pipeline

endlocal
