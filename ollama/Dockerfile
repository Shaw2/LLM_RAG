# Use Ollama base image
FROM ollama/ollama:latest

# Set environment variable for Ollama model path
ENV OLLAMA_MODEL_PATH=/models

# Set working directory to /app
WORKDIR /app

# Expose port 11434 for Ollama API
EXPOSE 11434

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Create volume to persist models
VOLUME ["/models"]

# Copy entrypoint script and set execute permissions
COPY ./entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Use entrypoint.sh as the entry point
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
