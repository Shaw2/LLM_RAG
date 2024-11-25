# Use Ollama base image
FROM ollama/ollama:latest

# Set environment variable for Ollama model path
ENV OLLAMA_MODEL_PATH=/models

# Set working directory to /app
WORKDIR /app

# Expose port 11434 for Ollama API
EXPOSE 11434

# Create volume to persist models
VOLUME ["/models"]

# Run Ollama server when the container starts
ENTRYPOINT ["ollama", "serve"]
