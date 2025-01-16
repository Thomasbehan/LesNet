#!/bin/bash

# Activate your Python environment if necessary
# source /path/to/your/venv/bin/activate

# Set the log directory where TensorBoard will look for event files
LOG_DIR="./logs"

# Ensure the log directory exists
mkdir -p $LOG_DIR

# Start TensorBoard
tensorboard --logdir=$LOG_DIR --host=0.0.0.0 --port=6006 &

# Print the URL to access TensorBoard
echo "TensorBoard is running at http://localhost:6006"
