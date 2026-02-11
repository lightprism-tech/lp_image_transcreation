@echo off
REM Open an interactive bash terminal in the Docker container

echo Opening bash terminal in Stage-1 Perception container...
echo Type 'exit' to leave the container shell
echo.

docker-compose run --rm perception /bin/bash
