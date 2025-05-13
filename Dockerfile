FROM python:3.9-slim

WORKDIR /app

# Install only necessary system dependencies for CPU usage
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python packages with specific order and clean cache
RUN pip install --no-cache-dir numpy==1.24.3 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# Copy the rest of the application
COPY . .

# Expose the port the app runs on
EXPOSE 7000

# Set environment variable to use CPU
ENV CUDA_VISIBLE_DEVICES=""

# Command to run the application
CMD ["python", "api_tricoscopio.py"] 