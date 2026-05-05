FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖（包括 git-lfs，部分模型下载可能需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    git-lfs \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 在构建阶段下载模型到 /app/model
RUN python -c "\
from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
model_name = 'Yihao-Jia/eist'; \
print(f'Downloading {model_name}...'); \
tokenizer = AutoTokenizer.from_pretrained(model_name); \
tokenizer.save_pretrained('/app/model'); \
model = AutoModelForSequenceClassification.from_pretrained(model_name); \
model.save_pretrained('/app/model'); \
print('Model saved to /app/model') \
"

COPY predict.py .

RUN chmod +x predict.py

ENTRYPOINT ["python", "-u", "predict.py"]
