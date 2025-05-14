# main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import pickle
from model_architecture import NewsClassifierNN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


# Initialize the app
app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


# Data model for incoming JSON
class NewsArticle(BaseModel):
    content: str


# Load vectorizer, svd transformer, and model
try:
    with open("preprocessing/tfidf_vectorizer.pkl", "rb") as f:
        tfidf_vectorizer = pickle.load(f)

    with open("preprocessing/svd_transformer.pkl", "rb") as f:
        svd = pickle.load(f)

    model = NewsClassifierNN(input_dim=100)
    model.load_state_dict(torch.load("model/best_model.pth"))
    model.eval()

except Exception as e:
    raise HTTPException(status_code=500, detail=f"Startup failed: {str(e)}")


# Prediction Endpoint
@app.post("/predict")
def predict(article: NewsArticle):
    try:
        X = tfidf_vectorizer.transform([article.content]).toarray()
        X_reduced = svd.transform(X)

        with torch.no_grad():
            output = model(torch.tensor(X_reduced, dtype=torch.float32))
            prob = torch.sigmoid(output).item()

        return {
            "real_confidence": round(1 - prob, 3),
            "fake_confidence": round(prob, 3)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
