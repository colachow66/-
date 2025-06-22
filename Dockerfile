# 使用NVIDIA基础镜像
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    wget \
    python3 \
    python3-pip \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 创建工作目录
WORKDIR /app

# 安装Python依赖
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 复制项目代码
COPY . .

# 验证安装
RUN python -c "import torch; print('PyTorch version:', torch.__version__); \
               import onnxruntime; print('ONNX Runtime version:', onnxruntime.__version__)"

# 设置默认命令
CMD ["bash"]

# 构建镜像
docker build -t thyroid-segmentation .

# 运行容器(映射数据目录)
docker run --gpus all -it -v /path/to/your/data:/app/data thyroid-segmentation