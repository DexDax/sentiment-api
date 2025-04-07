from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the model path
MODEL_PATH = "./bert_sentiment_new_dataset_final"

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=3)
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

# Set model to evaluation mode
model.eval()

# Define API
app = FastAPI()

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(input_data: InputText):
    print(f"Received input: {input_data.text}")  # Test if this shows up in Docker logs
    
    # Tokenize input text
    inputs = tokenizer(input_data.text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # Get raw outputs
        probabilities = torch.softmax(logits, dim=1)  # Convert logits to probabilities
        predictions = torch.argmax(probabilities, dim=1).item()  # Get predicted class
    
    # Convert prediction to sentiment
    sentiment = "Positive" if predictions == 2 else "Neutral" if predictions == 1 else "Negative"
    
    # Convert probabilities to percentage and round to 2 decimal places
    probabilities = probabilities.squeeze().tolist()  # Convert to list
    probabilities = [round(prob * 100, 2) for prob in probabilities]  # Convert to percentage
    
    print(f"Sentiment: {sentiment}, Probabilities: {probabilities}")  # Log output
    
    return {
        "sentiment": sentiment,
        "probabilities": {
            "Negative": probabilities[0],
            "Neutral": probabilities[1],
            "Positive": probabilities[2]
        }
    }
