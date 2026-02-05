# 使用 NVIDIA 官方提供的基础镜像（包含 CUDA 开发环境）
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

# 1. 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1

# 2. 安装系统基础依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential python3-dev python3-pip && \
    rm -rf /var/lib/apt/lists/*

# 3. 安装 uv (极速包管理神器)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 4. 设置工作目录
WORKDIR /workspace

# 5. 复制依赖文件 (假设你已经有了 pyproject.toml)
# 如果还没有，这一步可以先注释掉，我们先跑通镜像
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# 默认进入 bash
CMD ["/bin/bash"]