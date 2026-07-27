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

# Bake in the JEPA medium model the demo serves by default (LESNET_JEPA_HOME, default
# models/jepa). The app can self-heal by fetching it at runtime, but that would make the first
# request after a cold start pull 318 MB and very likely time out — so fetch it at build time.
RUN mkdir -p models/jepa && \
    curl -fsSL https://github.com/Thomasbehan/LesNet/releases/download/v5.0.0/lesnet-jepa-medium.tar.gz \
    | tar xz -C models/jepa && \
    test -f models/jepa/medium/jepa_config.json

# Add a new user to avoid running the application as root; let it write the model dir
# (so the app's self-healing model fetch works at runtime).
RUN useradd -ms /bin/bash appuser && chown -R appuser /app
USER appuser

# Make port 6543 available to the world outside this container and 6006 for TensorBoard
EXPOSE 6543 6006

# Define environment variable
ENV NAME World

# Ensure the pserve command is in the PATH
ENV PATH="/app/.local/bin:${PATH}"

# Run command when the container launches (production config — no debug toolbar, no autoreload)
CMD ["pserve", "production.ini"]