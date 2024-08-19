#!/bin/bash
set -e

echo "Current working directory: $(pwd)"
echo "Contents of /app:"
ls -la /app

echo "Contents of /app/models:"
ls -la /app/models

echo "Running models..."
python /app/models/run_models.py

echo "Script execution completed."