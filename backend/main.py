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

# =======================
# Initialize the app
# =======================

origins = ["*"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,   
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"]   
)

# =======================
# Data model for incoming JSON
# =======================
class NewsArticle(BaseModel):
    content: str

# =======================
# Load the vectorizer
# =======================
try:
    print("Loading TF-IDF Vectorizer...")
    with open("preprocessing/tfidf_vectorizer.pkl", "rb") as f:
        tfidf_vectorizer = pickle.load(f)
    print("TF-IDF Vectorizer loaded successfully.")
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Failed to load vectorizer: {str(e)}")

# =======================
# Load the SVD transformer
# =======================
try:
    print("Loading SVD Transformer...")
    with open("preprocessing/svd_transformer.pkl", "rb") as f:
        svd = pickle.load(f)
    print("SVD Transformer loaded successfully.")
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Failed to load SVD: {str(e)}")

# =======================
# Load the model
# =======================
try:
    print("Loading model...")
    model = NewsClassifierNN(input_dim=100)
    model.load_state_dict(torch.load("model/best_model.pth"))
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

# =======================
# Prediction Endpoint
# =======================
@app.post("/predict")
def predict(article: NewsArticle):
    try:
        # ===== Step 1: TF-IDF Transformation =====
        print("Performing TF-IDF transformation...")
        X = tfidf_vectorizer.transform([article.content])

        if X is None or X.shape[0] == 0:
            raise ValueError("TF-IDF transformation returned an empty matrix.")
        print(f"TF-IDF transformation complete. Shape: {X.shape}")
        
        # 🔎 **Debugging Logs**
        print("Vectorizer Vocabulary Size:", len(tfidf_vectorizer.vocabulary_))
        print("Vectorizer Feature Names Count:", len(tfidf_vectorizer.get_feature_names_out()))

        # ===== Step 2: Convert to Dense Array =====
        X = X.toarray()
        print(f"Dense array shape: {X.shape}")

        # ===== Step 3: SVD Transformation =====
        print("Performing SVD transformation...")
        print(f"SVD expects this many features: {svd.components_.shape[1]}")
        
        # 🔎 **Debugging Check**
        if X.shape[1] != svd.components_.shape[1]:
            print(f"❌ Size Mismatch: TF-IDF features: {X.shape[1]} vs. SVD expected: {svd.components_.shape[1]}")
            raise ValueError("TF-IDF vectorizer and SVD do not match in feature count.")
        
        X_reduced = svd.transform(X)
        print(f"SVD transformation complete. Shape: {X_reduced.shape}")

        # ===== Step 4: Model Prediction =====
        with torch.no_grad():
            print("Making prediction...")
            output = model(torch.tensor(X_reduced, dtype=torch.float32))
            prob = torch.sigmoid(output).item()
            print(f"Prediction complete. Real: {1 - prob}, Fake: {prob}")

        # ===== Step 5: Return the Response =====
        return {
            "real_confidence": round(1 - prob, 3),
            "fake_confidence": round(prob, 3)
        }

    except Exception as e:
        print("Prediction failed:", str(e))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
