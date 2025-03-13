#!/bin/bash

# Kill any existing processes
pkill -f langgraph
pkill -f streamlit

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Activate virtual environment
source .venv/bin/activate

# Install the package in development mode
uv pip install -e .

# Install necessary packages
uv pip install "langgraph-cli[inmem]"
uv pip install arcade-ai

# Start the LangGraph server
echo "Starting LangGraph server..."
langgraph dev &
LANGGRAPH_PID=$!

# Wait for the server to start
sleep 5

# Start the Streamlit app
echo "Starting Streamlit app..."
uv run streamlit run streamlit_app.py --server.port 8502 &
STREAMLIT_PID=$!

# Wait for both processes to finish
wait $LANGGRAPH_PID $STREAMLIT_PID 