# ----------------------------#
#  HVAC YOLO - Inference API  #
# ----------------------------#
FROM python:3.10-slim

# Avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Working directory inside the container
WORKDIR /app

# Install system dependencies (no GUI | no tk)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    roboflow==1.1.27 \
    ultralytics==8.3.3 \
    opencv-python-headless==4.8.0.74 \
    fastapi==0.115.2 \
    uvicorn==0.30.6 \
    python-multipart==0.0.9 \
    openvino==2024.3.0

COPY .. .
EXPOSE 8000
ENV RUNNING_IN_DOCKER=1

# Run the FastAPI server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]