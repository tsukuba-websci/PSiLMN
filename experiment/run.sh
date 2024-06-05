#!/bin/bash

# Directory for logs
LOG_DIR="output/logs"

# Create the logs directory if it doesn't exist
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p $LOG_DIR
    echo "Created directory $LOG_DIR for logs."
fi

# Check if there are files in /input (or deeper)
if [ -z "$(find input -type f)" ]; then
    echo "No files found in /input directory. Running generate_networks.py."
    poetry run python generate_networks.py
else
    echo "Files found in /input directory. Skipping generate_networks.py."
fi

# Run the main.py script in the background
nohup poetry run python main.py > $LOG_DIR/main.log 2>&1 &
MAIN_PID=$!
echo "main.py is running in the background. Output is being written to $LOG_DIR/main.log"

# Wait for main.py to finish
wait $MAIN_PID

# Run the analysis.py script in the background
nohup poetry run python analysis.py > $LOG_DIR/analysis.log 2>&1 &
ANALYSIS_PID=$!
echo "analysis.py is running in the background. Output is being written to $LOG_DIR/analysis.log"

# Wait for analysis.py to finish
wait $ANALYSIS_PID
