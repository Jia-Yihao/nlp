FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# HF cache (important)
ENV HF_HOME=/tmp/hf_home
ENV TRANSFORMERS_CACHE=/tmp/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x predict.py

ENTRYPOINT ["python", "-u", "predict.py"]
