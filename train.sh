#!/bin/bash

# Start TensorBoard in the background
tensorboard --logdir logs --bind_all &

# Give TensorBoard a few seconds to initialize
sleep 5

# Start the training command
python3.10 skinvestigatorai/core/ai/train.py &

# Wait for all background processes to complete
wait
