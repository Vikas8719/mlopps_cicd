# Use lightweight python
FROM python:3.10-slim

WORKDIR /app

# copy requirements first for caching
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV PORT=8080

EXPOSE 8080

# default command - run app (serve model/api)
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--workers", "1"]
