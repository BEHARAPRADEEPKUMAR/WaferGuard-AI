@echo off
setlocal
call .venv\Scripts\activate.bat
pytest
python -m py_compile app.py utils\*.py scripts\*.py
endlocal
