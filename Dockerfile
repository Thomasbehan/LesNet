# Official Ubuntu base image with Python 3.12
FROM ubuntu:24.04

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app/

# Install any needed packages and run setup.py
RUN apt-get update
RUN apt-get install -y --no-install-recommends python3.12 python3-pip python3.12-dev curl
RUN python3.12 -m pip install --break-system-packages --upgrade pip
RUN python3.12 -m pip install --break-system-packages -e .
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

# Fetch the released M-4s triage model + artifact bundle into the dir the app loads
# (LESNET_TRIAGE_ARTIFACTS, default models/triage) so the live demo serves the new model.
RUN mkdir -p models/triage && \
    curl -L -o models/triage/triage_model.keras https://github.com/Thomasbehan/LesNet/releases/download/4.1.0/LesNet.M-4s.keras && \
    curl -L -o models/triage/artifacts.json https://github.com/Thomasbehan/LesNet/releases/download/4.1.0/LesNet.M-4s.artifacts.json

# Add a new user to avoid running the application as root
RUN useradd -ms /bin/bash appuser
USER appuser

# Make port 6543 available to the world outside this container and 6006 for TensorBoard
EXPOSE 6543 6006

# Define environment variable
ENV NAME World

# Ensure the pserve command is in the PATH
ENV PATH="/app/.local/bin:${PATH}"

# Run command when the container launches (production config — no debug toolbar, no autoreload)
CMD ["pserve", "production.ini"]