@echo off
REM Helper script to run the perception pipeline in Docker (Windows)

set IMAGE_PATH=%~1
set OUTPUT_PATH=%~2

if "%IMAGE_PATH%"=="" (
    echo Usage: run_docker.bat ^<image_path^> [output_path]
    echo Example: run_docker.bat input_images\test.jpg outputs\result.json
    echo.
    echo Note: Paths are relative to the project root
    exit /b 1
)

if "%OUTPUT_PATH%"=="" (
    set OUTPUT_PATH=outputs/scene_json/output.json
)

REM Convert backslashes to forward slashes for Docker
set IMAGE_PATH=%IMAGE_PATH:\=/%
set OUTPUT_PATH=%OUTPUT_PATH:\=/%

echo Processing image: %IMAGE_PATH%
echo Output will be saved to: %OUTPUT_PATH%
echo.

REM Run the pipeline
docker-compose run --rm perception python main.py /app/%IMAGE_PATH% --output /app/%OUTPUT_PATH%

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [SUCCESS] Processing complete! Output saved to: %OUTPUT_PATH%
) else (
    echo.
    echo [ERROR] Processing failed. Check logs above for errors.
    exit /b 1
)
