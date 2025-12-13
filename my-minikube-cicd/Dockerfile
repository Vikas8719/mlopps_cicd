# Lightweight Python
FROM python:3.10-slim

WORKDIR /app

# Copy requirements for caching
COPY requirements.txt .

RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt


# Copy the rest of the app
COPY . .

# Env variables for cloud/minikube flexibility
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV DATA_PATH=src/SMSSpamCollection

EXPOSE 8080

# Run FastAPI app
CMD ["sh", "-c", "uvicorn src.app:app --host 0.0.0.0 --port ${PORT}"]
