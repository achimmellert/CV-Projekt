FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 nginx \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Nginx-Konfiguration
RUN echo "server { listen 80; location / { proxy_pass http://localhost:8000; } location /predict { proxy_pass http://localhost:8000; } }" > /etc/nginx/sites-available/default

EXPOSE 80

CMD sh -c "nginx & uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1"