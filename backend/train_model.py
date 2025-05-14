import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model_architecture import NewsClassifierNN
import os

# Load and label data
real_df = pd.read_csv("../data/True.csv")
fake_df = pd.read_csv("../data/Fake.csv")
real_df["label"] = 0
fake_df["label"] = 1

df = pd.concat([real_df, fake_df]).sample(frac=1, random_state=42).reset_index(drop=True)
articles = df["text"].values
labels = df["label"].values

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    articles, labels, test_size=0.2, random_state=42, stratify=labels
)

# Vectorization and SVD reduction
tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

svd = TruncatedSVD(n_components=100, random_state=42)
X_train_reduced = svd.fit_transform(X_train_tfidf)
X_test_reduced = svd.transform(X_test_tfidf)

# Saving transformers
os.makedirs("preprocessing", exist_ok=True)
with open("preprocessing/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf_vectorizer, f)
with open("preprocessing/svd_transformer.pkl", "wb") as f:
    pickle.dump(svd, f)

# PyTorch Dataset & DataLoader 
class NewsDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # Make it shape (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = NewsDataset(X_train_reduced, y_train)
test_dataset = NewsDataset(X_test_reduced, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# Model Setup
model = NewsClassifierNN(input_dim=100)
loss_function = nn.BCEWithLogitsLoss()   
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training Loop
EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f}")


# Evaluating on Test Set

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        preds = torch.sigmoid(outputs)
        all_preds.extend(preds.squeeze().tolist())
        all_labels.extend(batch_y.squeeze().tolist())

# Convert to binary predictions
binary_preds = [1 if p > 0.5 else 0 for p in all_preds]

print("Test Accuracy:", accuracy_score(all_labels, binary_preds))
print("Precision:", precision_score(all_labels, binary_preds))
print("Recall:", recall_score(all_labels, binary_preds))
print("F1 Score:", f1_score(all_labels, binary_preds))


# Save the trained model
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/best_model.pth")
print("Model saved to model/best_model.pth")
