FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
#
# The CPU build of torch goes in FIRST, from PyTorch's own index. requirements.txt
# declares torch because the inference engine fails closed without it, but the
# default Linux wheel drags in the entire CUDA stack - well over 2 GB of nvidia-*
# packages this image has no use for. Installing the CPU build here satisfies
# that requirement before pip reaches it.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend application source
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run with hot reloading enabled for development
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
