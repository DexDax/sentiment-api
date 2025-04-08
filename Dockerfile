# Use an official Python runtime as a base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install the Python dependencies specified in the requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files into the container
COPY app.py .
# Ensure model is copied inside the container
# COPY BERT_Fine_Tuned_model /app/BERT_Fine_Tuned_model
# COPY bert_sentiment_new_dataset_final /app/bert_sentiment_new_dataset_final

# Expose port 8000 for the FastAPI application
EXPOSE 8000

# Start the FastAPI app using Uvicorn with debugging enabled
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-level", "debug"]
