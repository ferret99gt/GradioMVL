@echo off

set "TORCH_BACKEND=rocm"
if not defined VENV_DIR (set "VENV_DIR=%~dp0venv-rocm")

call "%~dp0launch.bat" %*
