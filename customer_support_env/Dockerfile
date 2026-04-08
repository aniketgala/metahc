# Use Python 3.11 slim as the base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx

# Copy the environment code
COPY . /app/

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1

# Sync dependencies using uv
# This will use the uv.lock file we just generated and included in the build context
RUN uv sync --frozen --no-editable

# Expose the FastAPI port
EXPOSE 8000

# Health check to ensure the space starts correctly
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server directly
# Using --host 0.0.0.0 and --port 8000 as required by HF Spaces
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
