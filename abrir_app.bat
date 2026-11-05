@echo off
rem Abre la app de detección de malezas (doble click).
rem Usa el Python del venv directo: funciona aunque haya un entrenamiento corriendo.
rem pythonw = sin ventana de consola; los mensajes quedan en outputs\app.log.
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo No existe el entorno .venv. Corre primero: uv sync
    pause
    exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" -m maleza app
