FROM python:3.11-slim

WORKDIR /app

# Requirements für Flet installieren
RUN pip install --no-cache-dir flet httpx

COPY . .

# Port für Flet Web
EXPOSE 8550

CMD ["flet", "run", "--web", "--host", "0.0.0.0", "--port", "8550", "src/flet_app.py"]