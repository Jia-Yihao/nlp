FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy code
COPY predict.py .

# copy model directory (由 CI 提前下载到 repo/model)
COPY model ./model

RUN chmod +x predict.py

ENTRYPOINT ["python", "-u", "predict.py"]
