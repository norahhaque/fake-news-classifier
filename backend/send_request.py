import requests
import json

# URL for your FastAPI app
url = "http://127.0.0.1:8000/predict"

# Read the article from the file
with open("article.txt", "r") as file:
    content = file.read()

# Make the POST request
response = requests.post(url, json={"content": content})

# Check for errors
if response.status_code == 200:
    print("✅ Prediction Successful!")
    print(json.dumps(response.json(), indent=4))
else:
    print("❌ Prediction Failed.")
    print("Response Code:", response.status_code)
    print("Details:", response.json())
