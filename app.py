from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load your model pipeline
model = joblib.load("C:/Users/hp/Desktop/MINI_PROJECT/Mini_project/model.joblib")

# Initialize FastAPI app
app = FastAPI()

# Define request schema
class ReviewInput(BaseModel):
    review: str

# Prediction endpoint
@app.post("/predict")
def predict_sentiment(data: ReviewInput):
    review = data.review
    prediction = model.predict([review])[0]
    probability = model.predict_proba([review]).max()

    return {
        "review": review,
        "sentiment": "positive" if prediction == 1 else "negative",
        "confidence": round(probability, 2)
    }
