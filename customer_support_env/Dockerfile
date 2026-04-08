# Use Python 3.10 slim as the base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app"

# Set working directory
WORKDIR /app

# Install only essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for extremely fast dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv

# Copy only the dependency files first to leverage Docker cache
COPY pyproject.toml uv.lock ./

# Install dependencies WITHOUT the project itself first
# This layer will be cached unless pyproject.toml or uv.lock changes
RUN uv sync --frozen --no-install-project --no-editable

# Now copy the rest of the application code
COPY . .

# Final sync to install the project
RUN uv sync --frozen --no-editable

# Expose the default HF Space port
EXPOSE 7860

# Simple health check using the HF default port
HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start uvicorn directly on port 7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
