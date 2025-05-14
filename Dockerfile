FROM python:3.9-slim

WORKDIR /app

# Install only necessary system dependencies for CPU usage
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python packages - ensure specific versions are used
RUN pip install --no-cache-dir numpy==1.24.3 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# Copy the application code first (without model) to leverage Docker cache
COPY api_tricoscopio.py nginx.conf ./

# Copy the model file separately (this is usually the largest file)
COPY bestv2.pt ./

# Verify model file exists and is valid
RUN python -c "import os; assert os.path.exists('bestv2.pt') and os.path.getsize('bestv2.pt') > 0, 'Model file missing or empty'"

# Copy any remaining files
COPY . .

# Expose the port the app runs on
EXPOSE 7000

# Set environment variable to use CPU
ENV CUDA_VISIBLE_DEVICES=""

# Command to run the application
CMD ["python", "api_tricoscopio.py"] 