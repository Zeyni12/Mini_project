from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn

# Load your model pipeline
model = joblib.load("C:/Users/hp/Desktop/MINI_PROJECT/Mini_project/model.joblib")

# Initialize FastAPI app
app = FastAPI()

# Define request schema
class ReviewInput(BaseModel):
    review: str

@app.post("/predict")
def predict_sentiment(data: ReviewInput):
    review = data.review
    prediction = model.predict([review])[0]  # 'positive' or 'negative'
    probability = model.predict_proba([review]).max()

    return {
        "review": review,
        "sentiment": prediction,  # already a string
        "confidence": round(probability, 2)
    }


# Run the app
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
