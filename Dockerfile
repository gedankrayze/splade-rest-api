FROM python:3.15-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SPLADE_MODEL_DIR=/app/models/{model_name} \
    SPLADE_DATA_DIR=/app/app/data \
    SPLADE_AUTO_DOWNLOAD_MODEL=true \
    PATH="/app/.venv/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency metadata first for better caching
COPY pyproject.toml uv.lock README.md ./

# Install Python dependencies into .venv
RUN uv sync --frozen --no-dev

# Copy application code
COPY . .

# Create necessary directories if they don't exist
RUN mkdir -p /app/models /app/app/data

# Expose the port the app runs on
EXPOSE 3000

# Command to run the application
CMD ["uvicorn", "app.api.server:app", "--host", "0.0.0.0", "--port", "3000"]
