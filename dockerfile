# Base image with Python
FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    libgl1 libglib2.0-0 libxcb-xinerama0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# RUN apt-get update && apt-get install -y \
#     libgl1 libglib2.0-0 \
#     libx11-xcb1 libxcb1 libxcb-render0 libxcb-shm0 \
#     libxext6 libxrender1 \
#     libqt5gui5 libqt5widgets5 libqt5core5a qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools \
#     x11-xserver-utils && \
#     apt-get clean && rm -rf /var/lib/apt/lists/*



# Set working directory in container
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY model.onnx .
COPY real_time.py .
COPY emojis/ emojis/

EXPOSE 5000

# Command to run your script or application
CMD ["python3", "real_time.py"]
