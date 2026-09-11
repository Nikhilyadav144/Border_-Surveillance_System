#!/bin/bash

PROJECT="$HOME/Desktop/AI-Border-Surveillance"

cd "$PROJECT"

source "$PROJECT/.venv/bin/activate"

echo "Starting Border Guard AI Backend..."
uvicorn api.server:app --host 127.0.0.1 --port 8000 > backend.log 2>&1 &

sleep 3

echo "Starting React Dashboard..."
cd "$PROJECT/frontend"

npm run dev -- --host 127.0.0.1 > frontend.log 2>&1 &

sleep 3

echo "Opening Border Guard AI Dashboard..."
open "http://localhost:5173"

wait