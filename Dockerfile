# 使用 Python 3.12 slim 版本（与你本地环境一致，体积小）
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 避免交互式提示
ENV DEBIAN_FRONTEND=noninteractive

# 安装系统基础依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖文件并安装（利用 Docker 缓存加速）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 将当前目录所有内容复制到容器的 /app 目录中
# 此时容器里的结构就是 /app/predict.py 和 /app/model/...
COPY . .

# 设置入口点：Tira 运行时会自动在后面加上 [输入文件路径] [输出目录路径]
ENTRYPOINT ["python", "predict.py"]