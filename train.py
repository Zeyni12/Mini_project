import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# 1. Load the dataset
df = pd.read_csv("data/imdb_small.csv")

# 2. Split data
X = df["review"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 3. Create pipeline: TF-IDF + Logistic Regression
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000, stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000))
])

# 4. Train
pipeline.fit(X_train, y_train)

# 5. Evaluate (optional)
score = pipeline.score(X_test, y_test)
print(f"✅ Model Accuracy on Test Set: {score:.2f}")

# 6. Save model
joblib.dump(pipeline, "models/model.joblib")
print("✅ Model saved to models/model.joblib")

