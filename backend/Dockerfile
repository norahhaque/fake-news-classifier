# Use an official Python runtime as a base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data (stopwords)
RUN python -m nltk.downloader stopwords

# Add a non-root user for better security
RUN useradd -m appuser
USER appuser

# Copy your application code into the container
COPY . .

# Expose the port that FastAPI runs on
EXPOSE 8000

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
