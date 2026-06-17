FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies required for audio processing (ffmpeg, libsndfile, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
# We use --no-cache-dir to keep the image size small
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the Gradio default port
EXPOSE 7860

# Command to run the application (Gradio apps need to listen on all interfaces 0.0.0.0)
ENV GRADIO_SERVER_NAME="0.0.0.0"
CMD ["python", "app.py"]
