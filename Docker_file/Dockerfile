# Use Python base image
FROM python:3.12-slim
WORKDIR /app
# Copy the requirements.txt file into the container
COPY requirements.txt .
# Copy the Python files
COPY iris_reg.py .
COPY app.py .

# Install Dependencies
RUN pip install flask
RUN pip install -r requirements.txt

# Expose flask port 8080
EXPOSE 8080

# Run the server
CMD ["python", "app.py"]
