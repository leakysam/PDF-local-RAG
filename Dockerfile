# Use lightweight Python base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libheif-dev \
    && rm -rf /var/lib/apt/lists/*

# Set env vars
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app
COPY . .

# Default run command
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]

