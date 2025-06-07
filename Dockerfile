# Use official Python image
FROM python:3.10-slim

# Avoid interactive prompts during installs
ENV DEBIAN_FRONTEND=noninteractive

# Create app directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
