# Base image with Python
FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    libgl1 libglib2.0-0 libxcb-xinerama0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory in container
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY model.onnx .
COPY real_time.py .
COPY emojis/ emojis/
COPY mtcnn_ort/ mtcnn_ort/

# Command to run your script or application
CMD ["python3", "real_time.py"]
