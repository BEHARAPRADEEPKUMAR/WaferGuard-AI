@echo off
setlocal
if not exist .venv\Scripts\python.exe (
    py -3.11 -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if not exist .env copy .env.example .env
streamlit run app.py
endlocal
