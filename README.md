# Mini Sentiment Analysis Project

This project is a simple sentiment classifier for text reviews (e.g., movie reviews), using a machine learning pipeline built with **scikit-learn**, **TF-IDF**, and **Logistic Regression**. It includes both a command-line interface and a web API using **FastAPI**.

---

# Installation

1. Clone the project or download the source files.
2. (Optional) Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # macOS/Linux

3. Install the required packages:

   ```bash 
   pip install -r requirements.txt

# Training the Model

```bash
python train.py

# Running Predictions 

1. Command-Line Interface (CLI)
You can test predictions from the terminal using:

```bash
python predict.py "This movie was amazing!"

Sample output:
 Review: This movie was amazing!
 Sentiment: positive (0.97 confidence)

2. Web API with FastAPI 
  Start the API server:

```bash
python app.py

then go to  ,
http://127.0.0.1:8000/docs

Submit a review using the /predict endpoint:

json 
{
  "review": "this movie was amazing"
}

Sample response:

json
{
  "review": "this movie was amazing",
  "sentiment": "positive",
  "confidence": 0.97
}

🧾 Project Structure

Mini_project/
│
notbook/
    ├── analysis.ipynb    # data analysis for the imdb.csv
    ├── traning_data.ipynb # Script to train and save the model
├── app.py             # FastAPI app for prediction
├── predict.py         # CLI script to test predictions
├── model.joblib       # Trained model file
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation

 Requirements
 #Python 3.8+

. scikit-learn

. joblib

. FastAPI

. uvicorn

. numpy

. pydantic

Author 

Zeyneb Mulat – Mini project for text classification