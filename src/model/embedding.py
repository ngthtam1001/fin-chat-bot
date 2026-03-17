import requests

def embed_ollama(text):

    response = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "mxbai-embed-large",
            "input": text
        }
    )

    return response.json()["embeddings"][0]
