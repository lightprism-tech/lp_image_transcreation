#!/bin/bash
# Helper script to run the perception pipeline in Docker

IMAGE_PATH=$1
OUTPUT_PATH=${2:-"outputs/scene_json/output.json"}

if [ -z "$IMAGE_PATH" ]; then
    echo "Usage: ./run_docker.sh <image_path> [output_path]"
    echo "Example: ./run_docker.sh input_images/test.jpg outputs/result.json"
    echo ""
    echo "Note: Paths are relative to the project root"
    exit 1
fi

# Convert Windows paths to Unix if needed (for Git Bash on Windows)
IMAGE_PATH="${IMAGE_PATH//\\//}"
OUTPUT_PATH="${OUTPUT_PATH//\\//}"

echo "Processing image: $IMAGE_PATH"
echo "Output will be saved to: $OUTPUT_PATH"
echo ""

# Run the pipeline
docker-compose run --rm perception python main.py "/app/$IMAGE_PATH" --output "/app/$OUTPUT_PATH"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Processing complete! Output saved to: $OUTPUT_PATH"
else
    echo ""
    echo "✗ Processing failed. Check logs above for errors."
    exit 1
fi
