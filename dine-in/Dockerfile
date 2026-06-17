FROM intel/dlstreamer:2026.1.0-ubuntu24

USER root

# Set working directory
WORKDIR /app

# Install system dependencies
RUN rm -f /etc/apt/sources.list.d/intel-gpu*.list 2>/dev/null; \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    git \
    gosu && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN python3 -m pip install --break-system-packages --ignore-installed --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --break-system-packages --no-cache-dir -r requirements.txt

# Copy application files from src
COPY src/ ./src/
COPY configs/ ./configs/

# Create results directory and set permissions (dlstreamer is UID 1000 in base image)
RUN mkdir -p /app/results && chown -R dlstreamer:dlstreamer /app
# Copy entrypoint script (runs as root to fix volume permissions, then drops to appuser)
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose ports
EXPOSE 7860
EXPOSE 8080

# Entrypoint fixes permissions and drops to appuser
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python3", "src/main.py"]
