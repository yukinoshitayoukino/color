# syntax=docker/dockerfile:1.4
FROM python:3.9-slim
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
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем requirements.txt первым для лучшего кэширования
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальные файлы
COPY . .

# Создаем необходимые директории
RUN mkdir -p static/uploads static/modified static/histograms templates

# Открываем порт
EXPOSE 8000

# Запускаем приложение
CMD ["python", "main.py"]