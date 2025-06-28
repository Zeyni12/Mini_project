import sys
import joblib

# Load the model
model = joblib.load("C:/Users/hp/Desktop/MINI_PROJECT/Mini_project/model.joblib")

# Get input from command line
if len(sys.argv) < 2:
    print(" Please provide a review as a command-line argument.")
    print(' Example: python predict.py "This movie was amazing!"')
    sys.exit()

review_text = " ".join(sys.argv[1:])

# Make prediction
probabilities = model.predict_proba([review_text])[0]
prediction = model.predict([review_text])[0]
confidence = max(probabilities)

# Print result
print(f"Review: {review_text}")
print(f"Sentiment: {prediction} ({confidence:.2f} confidence)")

