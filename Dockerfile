# Official Ubuntu base image with Python 3.10
FROM ubuntu:22.04

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app/

# Install any needed packages and run setup.py
RUN apt-get update
RUN apt-get install -y --no-install-recommends python3.10 python3-pip python3.10-dev curl
RUN python3.10 -m pip install --upgrade pip
RUN python3.10 -m pip install -e .
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

# Get the latest available model
RUN curl -L -o models/LesNet.keras https://github.com/Thomasbehan/LesNet/releases/download/3.1.0/LesNet.keras

# Add a new user to avoid running the application as root
RUN useradd -ms /bin/bash appuser
USER appuser

# Make port 6543 available to the world outside this container and 6006 for TensorBoard
EXPOSE 6543 6006

# Define environment variable
ENV NAME World

# Ensure the pserve command is in the PATH
ENV PATH="/app/.local/bin:${PATH}"

# Run command when the container launches
CMD ["pserve", "development.ini", "--reload"]