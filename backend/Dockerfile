FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ML source so backend can import evaluate.py / model.py
COPY ../ml /app/ml
COPY app/ /app/app/

# Expose FastAPI on port 8000
EXPOSE 8000

# Initialize SQLite DB tables on startup via an entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]
