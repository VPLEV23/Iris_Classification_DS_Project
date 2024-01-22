# Dockerfile for training
# Start from a base Python image with version 3.11
FROM python:3.10

# Initialize settings_name variable with the default value
ARG settings_name=settings.json
ENV CONF_PATH=${settings_name}

# Changing the path to the working directory in the container
WORKDIR /app

# Copy data folder, training code, utils.py, settings.json, and requirements.txt to Docker
COPY data/ /app/data
COPY training/ /app/training
COPY utils.py /app
COPY ${settings_name} /app
COPY requirements.txt /app

# Install any necessary packages listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Train the model
RUN python3 /app/training/train.py