from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Load model and tokenizer
model_name = "dex0dax/odoo_sentiment_analysis_bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Label mapping (adjust if needed based on your training)
id2label = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

# Define input data structure
class InputText(BaseModel):
    text: str

# API endpoint to predict sentiment
@app.post("/predict")
def predict_sentiment(input_data: InputText):
    # Log the received input
    logger.info(f"Received input text: {input_data.text}")
    
    # Tokenize input text
    inputs = tokenizer(input_data.text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # Get raw outputs
        probabilities = torch.softmax(logits, dim=1)  # Convert logits to probabilities
        predictions = torch.argmax(probabilities, dim=1).item()  # Get predicted class
    
    # Convert prediction to sentiment
    sentiment = id2label[predictions]
    
    # Convert probabilities to percentage and round to 2 decimal places
    probabilities = probabilities.squeeze().tolist()  # Convert to list
    probabilities = [round(prob * 100, 2) for prob in probabilities]  # Convert to percentage

    # Log the prediction and probabilities
    logger.info(f"Sentiment prediction: {sentiment}")
    logger.info(f"Probabilities: {probabilities}")

    return {
        "sentiment": sentiment,
        "probabilities": {
            "Negative": probabilities[0],
            "Neutral": probabilities[1],
            "Positive": probabilities[2]
        }
    }

# To run this app, use: uvicorn app:app --reload
