FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HUB_ENABLE_HF_TRANSFER=1  # 加速 Hugging Face 下载

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

# 注意：不能加任何参数，让 TIRA 在运行时传递
ENTRYPOINT ["python", "-u", "predict.py"]
