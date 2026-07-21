FROM python:3.10-slim

# ffmpeg ve gerekli sistem araçlarını kur
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Önce requirements.txt'i kopyala ve bağımlılıkları yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Projenin geri kalanını kopyala
COPY . .

# Botu başlat
CMD ["python", "bot/main.py"]
