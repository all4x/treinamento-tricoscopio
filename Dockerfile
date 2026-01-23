FROM python:3.9-slim

WORKDIR /app

# Install necessary system dependencies for CPU usage
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python packages with increased timeout
RUN pip install --no-cache-dir numpy==1.24.3 && \
    pip install --no-cache-dir --timeout=300 -r requirements.txt && \
    pip cache purge

# Copy the application code first
COPY api_tricoscopio.py nginx.conf ./

# Download the model file from the file server with a different name
RUN wget -O model_yolo_new.pt "https://projetos-filebrowser.6lzljz.easypanel.host/api/public/dl/inxugQkS/arquivos/bestv2.pt" || \
    curl -L -o model_yolo_new.pt "https://projetos-filebrowser.6lzljz.easypanel.host/api/public/dl/inxugQkS/arquivos/bestv2.pt"

# Verify model file exists and is valid
RUN python -c "import os; assert os.path.exists('model_yolo_new.pt') and os.path.getsize('model_yolo_new.pt') > 1000000, 'Model file missing or too small'"

# Rename the model to what the application expects
RUN mv model_yolo_new.pt bestv2.pt

# Copy any remaining files EXCEPT the model
COPY --chown=root:root \
     .dockerignore \
     docker-compose.yml \
     nginx.conf \
     requirements.txt \
     tricoscopio.yaml \
     ./

# Expose the port the app runs on
EXPOSE 7000

# Set environment variable to use CPU
ENV CUDA_VISIBLE_DEVICES=""

# Command to run the application
CMD ["python", "api_tricoscopio.py"] 