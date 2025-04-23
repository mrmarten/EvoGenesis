FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the web UI port
EXPOSE 5000

# Create data and output directories
RUN mkdir -p /app/data /app/output

# Set environment variables
ENV PYTHONPATH=/app

# Command to run the application in service mode
CMD ["python", "main.py", "--mode", "service", "--config", "config/default.json"]
