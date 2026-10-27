FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Environment variables for Ollama
ENV LLM_PROVIDER=ollama \
    OLLAMA_MODEL=qwen2.5:1.5b \
    OLLAMA_BASE_URL=http://127.0.0.1:11434

# Expose port
EXPOSE 8000

# Start script: launch Ollama in background, then start server
CMD sh -c "ollama serve & sleep 3 && ollama pull qwen2.5:1.5b && python run.py"
