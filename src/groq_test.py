import requests
import os

api_key = "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"
url = "https://api.groq.com/openai/v1/models"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

response = requests.get(url, headers=headers)

print(response.json())