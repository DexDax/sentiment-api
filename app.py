from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Load sentiment analysis pipeline
model_name = "dex0dax/odoo_sentiment_analysis_bert"
sentiment_analysis = pipeline("sentiment-analysis", model=model_name)

# Define input data structure
class InputText(BaseModel):
    text: str

# API endpoint to predict sentiment
@app.post("/predict")
def predict_sentiment(input_data: InputText):
    # Log the received input
    logger.info(f"Received input text: {input_data.text}")
    
    # Get the prediction using the pipeline
    result = sentiment_analysis(input_data.text)
    
    # Log the prediction
    logger.info(f"Prediction result: {result}")

    # Extract sentiment and probabilities
    sentiment = result[0]['label']
    probability = result[0]['score'] * 100  # Convert to percentage
    
    # Log the sentiment and probability
    logger.info(f"Sentiment prediction: {sentiment}")
    logger.info(f"Probability: {probability:.2f}%")

    # Map sentiment label to your custom labels (if needed)
    sentiment_map = {
        "LABEL_0": "Negative",  # Adjust this according to your model's label mapping
        "LABEL_1": "Neutral",   # Adjust this according to your model's label mapping
        "LABEL_2": "Positive"   # Adjust this according to your model's label mapping
    }

    # Return the response
    return {
        "sentiment": sentiment_map.get(sentiment, sentiment),  # Custom label mapping if needed
        "probability": round(probability, 2)
    }
