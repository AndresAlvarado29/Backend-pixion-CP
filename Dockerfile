

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04


# Establece variables de entorno
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Actualiza los paquetes y instala dependencias del sistema
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libssl-dev \
    libffi-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Configura el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia los archivos necesarios al contenedor
COPY requirements.txt /app/requirements.txt
COPY . /app/

EXPOSE 8000
# Instala las dependencias de Python
RUN pip install --upgrade pip && pip install -r requirements.txt

# Configura Gunicorn como punto de entrada
CMD ["gunicorn", "--workers=1", "--threads=4", "--bind=0.0.0.0:8000", "backend_pixion.wsgi"]
