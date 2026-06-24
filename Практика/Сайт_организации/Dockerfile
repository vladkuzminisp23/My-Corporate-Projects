# Используем легкий образ с Python
FROM python:3.9-slim

# Устанавливаем системные зависимости для OpenCV/Pillow/Torch
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем зависимости и устанавливаем
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY . .

# Открываем порт
EXPOSE 8000

# Запускаем сервер (многопоточный uvicorn)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]