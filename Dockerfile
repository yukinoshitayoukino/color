# Используем Debian Bullseye для совместимости
FROM python:3.9-slim-bullseye

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY . .

# Создаем директории
RUN mkdir -p static/uploads static/modified static/histograms templates

# Копируем шаблон если его нет
COPY templates/index.html templates/

EXPOSE 8000

CMD ["python", "main.py"]