FROM python:3.10-slim

# Proxy configuration
ARG http_proxy
ARG https_proxy
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG no_proxy
ARG NO_PROXY

#ENV http_proxy=${http_proxy} \
#    https_proxy=${https_proxy} \
#    HTTP_PROXY=${HTTP_PROXY} \
#    HTTPS_PROXY=${HTTPS_PROXY} \
#    no_proxy=${no_proxy} \
#    NO_PROXY=${NO_PROXY}

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    git && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create non-root user for security
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} appuser && \
    useradd -u ${UID} -g ${GID} -m appuser

# Copy application files from src
COPY src/ ./src/
COPY configs/ ./configs/

# Create results directory and set permissions
RUN mkdir -p /app/results && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 7860
EXPOSE 8080

# Run the combined application from src directory
CMD ["python", "src/main.py"]
