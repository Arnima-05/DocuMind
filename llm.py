import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

url = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

question = input("Ask me anything: ")

data = {
    "model": "openrouter/free",
    "messages": [
        {
            "role": "user",
            "content": question
        }
    ]
}

response = requests.post(
    url,
    headers=headers,
    json=data
)

result = response.json()

print(result["choices"][0]["message"]["content"])