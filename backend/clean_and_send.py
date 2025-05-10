import requests
import json

def clean_article(content):
    """
    Escapes double quotes and backslashes for proper JSON formatting.
    """
    return content.replace('\\', '\\\\').replace('"', '\\"')


def send_request(article_content):
    """
    Sends a cleaned request to the FastAPI endpoint.
    """
    cleaned_content = clean_article(article_content)

    # Prepare the data in JSON format
    data = {"content": cleaned_content}
    
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=data)
        response.raise_for_status()  # Raise an exception for HTTP errors
        print("Response:", response.json())
    except requests.RequestException as e:
        print("Request failed:", e)


if __name__ == "__main__":
    article = """Paste your article text here, including any "quotes" or special characters."""
    send_request(article)
