# ============================================================================
# ETAPA 1: Builder (Construcción)
# ============================================================================
FROM python:3.10-slim AS builder

WORKDIR /app

# Instalar herramientas esenciales de compilación
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Crear y activar entorno virtual
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 1. Actualizar pip
RUN pip install --no-cache-dir --upgrade pip

# 2. Instalar PyTorch optimizado para CUDA 12.1 (Más ligero que 11.8)
# Sigue siendo compatible con Tesla T4 y drivers modernos
RUN pip install --no-cache-dir \
    torch==2.1.0+cu121 \
    torchvision==0.16.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# 3. Instalar dependencias del proyecto (Asegúrate que tu requirements.txt NO tenga 'torch')
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. LIMPIEZA QUIRÚRGICA (Aquí es donde bajamos de los 8GB)
# Eliminamos archivos de objetos, cache y librerías estáticas de NVIDIA que no se usan en ejecución
RUN find /opt/venv -name "*.a" -delete && \
    find /opt/venv -name "*.o" -delete && \
    find /opt/venv -name "__pycache__" -type d -exec rm -rf {} + && \
    rm -rf /opt/venv/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn_static.a && \
    rm -rf /opt/venv/lib/python3.10/site-packages/nvidia/cublas/lib/libcublas_static.a

# ============================================================================
# ETAPA 2: Runtime (Imagen Final)
# ============================================================================
FROM python:3.10-slim

# Variables de entorno para Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Instalar solo las librerías compartidas necesarias para OpenCV y ejecución de GPU
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar el entorno virtual optimizado desde la etapa anterior
COPY --from=builder /opt/venv /opt/venv

# Copiar el código fuente
# IMPORTANTE: Asegúrate de tener un .dockerignore para no copiar 'data/' o 'runs/'
COPY . .

# Limpieza preventiva de archivos del host que se hayan podido colar
RUN rm -rf .git .github .gitignore venv/ env/ .ipynb_checkpoints/

# Configuración de seguridad y directorios
RUN mkdir -p /app/data /app/output /app/models && \
    useradd --create-home --shell /bin/bash trainer && \
    chown -R trainer:trainer /app

USER trainer

# Configuración por defecto de YOLO
ENV EPOCHS=50 \
    BATCH_SIZE=16 \
    IMG_SIZE=640

ENTRYPOINT ["python", "train.py"]