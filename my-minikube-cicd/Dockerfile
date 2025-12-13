FROM python:3.10-slim

WORKDIR /app

# Copy requirements for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy src folder and model
COPY src/ ./src/
COPY model.pkl .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV MLFLOW_TRACKING_URI=http://mlflow-service:5000
ENV DATA_PATH=src/SMSSpamCollection

EXPOSE 8080

# Run FastAPI
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8080"]
