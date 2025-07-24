# Base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy files
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt



# Copy rest of the code
COPY . .

# Create necessary folders
RUN mkdir -p data/raw data/preprocessed embeddings vectorstore

# Default command (optional)
#CMD ["streamlit", "run", "ui/app.py"]
