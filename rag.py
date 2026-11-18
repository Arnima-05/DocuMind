import os
import chromadb
import requests

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


# -----------------------------
# 1. Load environment variables
# -----------------------------

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")


# -----------------------------
# 2. Load embedding model
# -----------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# 3. Connect to ChromaDB
# -----------------------------

client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_collection(
    name="documind"
)


# -----------------------------
# 4. Ask the user a question
# -----------------------------

question = input("Ask a question about the document: ")


# -----------------------------
# 5. Convert question to embedding
# -----------------------------

question_embedding = model.encode([question])


# -----------------------------
# 6. Retrieve relevant chunks
# -----------------------------

results = collection.query(
    query_embeddings=question_embedding.tolist(),
    n_results=3
)

documents = results["documents"][0]


# -----------------------------
# 7. Combine retrieved chunks
# -----------------------------

context = "\n\n".join(documents)


# -----------------------------
# 8. Create prompt for the LLM
# -----------------------------

prompt = f"""
You are a helpful document assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer cannot be found in the context,
say that the information is not available in the document.

Context:
{context}

Question:
{question}

Answer:
"""


# -----------------------------
# 9. Send context + question to OpenRouter
# -----------------------------

url = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

data = {
    "model": "openrouter/free",
    "messages": [
        {
            "role": "user",
            "content": prompt
        }
    ]
}


response = requests.post(
    url,
    headers=headers,
    json=data
)


# -----------------------------
# 10. Get the LLM answer
# -----------------------------

result = response.json()

answer = result["choices"][0]["message"]["content"]


# -----------------------------
# 11. Display the answer
# -----------------------------

print("\nAnswer:")
print(answer)