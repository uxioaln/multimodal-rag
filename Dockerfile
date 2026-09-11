# 多模态数据处理 RAG - Docker 镜像
#
# 构建：
#   docker build -t disney-rag .
#
# 运行（密钥通过 --env-file 注入，数据目录挂载持久化）：
#   docker run --rm -p 5050:5050 --env-file .env -v "$(pwd)/data:/app/data" disney-rag
#
# 首次使用需构建向量索引（调用云端 API，产生费用，故不随容器自动执行）：
#   docker run --rm --env-file .env -v "$(pwd)/data:/app/data" disney-rag \
#       python scripts/build_index.py
#
# 模型说明：本项目使用云端模型服务（DashScope 多模态 embedding / AGICTO Chat LLM），
# 构建镜像时无需下载模型权重，仅需安装 Python 依赖（已配置国内 PyPI 镜像加速）。

FROM python:3.10-slim

# PyPI 镜像源：默认阿里云源（本机 Docker Desktop 内置代理对清华源返回 403）
# 如需换回清华源：docker build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple -t disney-rag .
ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/

# PYTHONUNBUFFERED: 让日志实时输出到 docker logs
# PIP_INDEX_URL:   国内 PyPI 镜像，加速依赖安装（可通过 --build-arg 覆盖）
# TZ:              容器时区（SQLite 中 datetime('now','localtime') 依赖系统时区数据）
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Asia/Shanghai

# tzdata 提供时区数据；libgomp1 为 numpy 运行时依赖
RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先安装依赖再拷贝源码：requirements.txt 未变化时可直接命中构建缓存
# --retries/--timeout: 镜像源偶发抖动时自动重试，避免构建因瞬时网络故障失败
COPY requirements.txt ./
RUN pip install --no-cache-dir --retries 5 --timeout 60 -r requirements.txt

COPY . .

# 补齐运行时数据目录（本地运行数据已被 .dockerignore 排除，源文档 data/knowledge_base 随镜像携带）
RUN mkdir -p data/db data/qdrant data/indexes data/stats

EXPOSE 5050

# 启动前初始化数据库（scripts/init_db.py 幂等：建表 IF NOT EXISTS + 管理员去重）
CMD ["sh", "-c", "python scripts/init_db.py && exec python run.py"]

# 健康检查：首页可访问即视为存活
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5050/', timeout=4)" || exit 1
