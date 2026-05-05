FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
# 加速 Hugging Face 下载（注释单独一行）
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# 安装 git 和 git-lfs（Hugging Face 下载需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    git-lfs \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x predict.py

ENTRYPOINT ["python", "-u", "predict.py"]
