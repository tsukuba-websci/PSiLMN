#!/bin/bash

# Function to kill a process by name
kill_process_by_name() {
    local process_name=$1
    local pids=$(pgrep -f $process_name)

    if [ -n "$pids" ]; then
        echo "Killing $process_name processes with PIDs: $pids"
        kill -9 $pids
    else
        echo "No $process_name process found."
    fi
}

# Kill the main.py process
kill_process_by_name "main.py"

# Kill the analysis.py process
kill_process_by_name "analysis.py"

# Kill the run.sh script process
kill_process_by_name "run.sh"

echo "All specified processes have been checked and terminated if they were running."
