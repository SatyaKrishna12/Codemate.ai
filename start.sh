#!/bin/bash

# Quick Start Script for Deep Researcher Agent
# This script activates the virtual environment and starts the server

echo "🚀 Starting Deep Researcher Agent..."
echo "📍 Activating virtual environment..."

# Activate virtual environment (Windows)
if [ -f "./researcher_env/Scripts/activate" ]; then
    source ./researcher_env/Scripts/activate
elif [ -f "./researcher_env/bin/activate" ]; then
    source ./researcher_env/bin/activate
else
    echo "❌ Virtual environment not found!"
    exit 1
fi

echo "✅ Virtual environment activated"

# Install dependencies if needed
echo "📦 Checking dependencies..."
pip install -r requirements.txt --quiet

echo "🌐 Starting server on http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo "💡 Health Check: http://localhost:8000/api/v1/researcher/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=========================="

# Start the server
python -m uvicorn main_simplified:app --host 127.0.0.1 --port 8000 --reload
