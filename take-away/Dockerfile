FROM intel/dlstreamer:2025.2.0-ubuntu22

USER root

# -------------------------------
# Proxy support
# -------------------------------
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}
ENV NO_PROXY=${NO_PROXY}
ENV http_proxy=${HTTP_PROXY}
ENV https_proxy=${HTTPS_PROXY}
ENV no_proxy=${NO_PROXY}

# -------------------------------
# Python path
# -------------------------------
# Include DL Streamer Python paths for gvapython plugin
ENV PYTHONPATH="/app:/opt/intel/dlstreamer/python:/opt/intel/dlstreamer/gstreamer/lib/python3/dist-packages:/home/dlstreamer/dlstreamer/python:${PYTHONPATH}"

# -------------------------------
# Prevent interactive tzdata prompt
# -------------------------------
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# -------------------------------
# Install system dependencies
# -------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tzdata \
        git \
        curl \
        ca-certificates \
        jq \
    && rm -rf /var/lib/apt/lists/*
# Note: python3-gi and python3-gst-1.0 are already provided by the
# intel/dlstreamer base image. Installing them here conflicts with
# Intel's libglvnd0 (1.7.0-2101~22.04) which satisfies a different
# version than libegl1 expects (= 1.4.0-1).

# -------------------------------
# Install Python dependencies
# -------------------------------
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

RUN python3 -m pip install --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir -r requirements.txt

# -------------------------------
# Copy application code
# -------------------------------
# Copy all modules
COPY scripts /scripts

ENV PYTHONPATH="/app:${PYTHONPATH}"
COPY src/core/ /app/core/
COPY src/api/ /app/api/
COPY src/parallel/ /app/parallel/
COPY config/ /app/config/
COPY src/main.py src/frame_pipeline.py config_loader.py /app/

# Make main.py executable
RUN chmod +x /app/main.py

# -------------------------------
# Pre-download EasyOCR models to avoid race conditions
# -------------------------------
RUN mkdir -p /models/easyocr && \
    python3 -c "import easyocr; easyocr.Reader(['en'], gpu=False, verbose=False, model_storage_directory='/models/easyocr')" && \
    echo "EasyOCR models pre-downloaded to /models/easyocr"

# -------------------------------
# GStreamer configuration
# -------------------------------
# Set GStreamer plugin path for DL Streamer
ENV GST_PLUGIN_PATH=/opt/intel/dlstreamer/lib:/opt/intel/dlstreamer/gstreamer/lib/gstreamer-1.0:/opt/intel/dlstreamer/gstreamer/lib/
ENV GST_VA_ALL_DRIVERS=1

# -------------------------------
# Service configuration
# -------------------------------
# Default to single mode (can be overridden in docker-compose)
ENV SERVICE_MODE=single
ENV WORKERS=1
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# -------------------------------
# Expose ports
# -------------------------------
EXPOSE 8000

# -------------------------------
# Health check
# -------------------------------
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# -------------------------------
# Entry point
# -------------------------------
CMD ["python3", "/app/main.py"]
