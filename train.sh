#!/bin/sh

# Define color codes for output
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to output messages with color
log_message() {
    local color="$1"
    shift
    echo -e "${color}$@${NC}"
}

# Start the training process
log_message "${CYAN}" "Starting the training process for LesNet..."

# Pull latest updates from Git repository
log_message "${GREEN}" "Pulling latest updates from Git repository..."
git pull
if [ $? -eq 0 ]; then
    log_message "${GREEN}" "Updates pulled successfully."
else
    log_message "${GREEN}" "Failed to pull updates from Git repository." >&2
    exit 1
fi

# Install required Python packages
log_message "${GREEN}" "Installing required Python packages..."
python -m pip install -e .[testing]
if [ $? -eq 0 ]; then
    log_message "${GREEN}" "Dependencies installed successfully."
else
    log_message "${GREEN}" "Failed to install dependencies." >&2
    exit 1
fi

# Run the training model script
log_message "${GREEN}" "Running training model script..."
python commands/run_train_model.py
if [ $? -eq 0 ]; then
    log_message "${CYAN}" "Training process completed successfully. Check above logs for details."
else
    log_message "${GREEN}" "Training process encountered an error." >&2
    exit 1
fi
