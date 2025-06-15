# Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files to the container
COPY . .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    python train_models.py

# Expose the ports for Flask and gRPC
EXPOSE 8000
EXPOSE 50051

# Run both Flask and gRPC server together
CMD ["sh", "-c", "python grpc_db_server.py & python app.py"]