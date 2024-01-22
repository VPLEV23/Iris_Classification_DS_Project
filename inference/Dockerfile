# Dockerfile for inference
# Start from a base Python image with Python 3.10
FROM python:3.10

# Set the model name and settings file name as arguments
ARG model_name=iris_pytorch_model.pth
ARG settings_name=settings.json
ENV CONF_PATH=${settings_name}

WORKDIR /app

# Copy data folder to Docker
COPY data/ /app/data

# Copy models folder to Docker
COPY models/${model_name} /app/models/${model_name}

# Copy the inference code
COPY inference /app/inference

# Copy utility scripts and configuration files
COPY utils.py /app
COPY ${CONF_PATH} /app

# Install any necessary packages listed in requirements.txt
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Command to run the inference script
CMD ["python3", "inference/run.py"]
