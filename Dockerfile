FROM python:3.12-slim

WORKDIR /app

# System deps for TF and matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/
COPY app/ app/
COPY models/ models/

RUN pip install --no-cache-dir ".[app]"

EXPOSE 7860

CMD ["python", "app/app.py"]
