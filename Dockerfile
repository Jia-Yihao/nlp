FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (no git-lfs needed because model is pre‑downloaded)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (leverage Docker cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model directory (provided by CI, contains full model weights)
COPY model ./model

# Copy prediction script
COPY predict.py .

# Make script executable
RUN chmod +x predict.py

# Entrypoint – TIRA will pass arguments
ENTRYPOINT ["python", "-u", "predict.py"]
