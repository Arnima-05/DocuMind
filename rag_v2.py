import os
import requests
import chromadb

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


# -----------------------------
# 1. Load API key
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
    name="documind_v2"
)


# -----------------------------
# 4. Ask user for a question
# -----------------------------

question = input("\nAsk a question about the document: ")


# -----------------------------
# 5. Create question embedding
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
metadatas = results["metadatas"][0]


# -----------------------------
# 7. Create context
# -----------------------------

context_parts = []

for document, metadata in zip(documents, metadatas):

    source = metadata["source"]
    page = metadata["page"]

    context_parts.append(
        f"Source: {source}, Page: {page}\n"
        f"{document}"
    )


context = "\n\n".join(context_parts)


# -----------------------------
# 8. Create LLM prompt
# -----------------------------

prompt = f"""
You are DocuMind, an AI document assistant.

Answer the user's question using ONLY the information
provided in the document context below.

Do not use outside knowledge.

If the answer cannot be found in the provided context,
say:

"The information is not available in the document."

Give a clear and concise answer.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""


# -----------------------------
# 9. Send request to OpenRouter
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
# 10. Handle response
# -----------------------------

result = response.json()

if "choices" not in result:

    print("\nSomething went wrong.")
    print(result)

else:

    answer = result["choices"][0]["message"]["content"]

    print("\n==============================")
    print("ANSWER")
    print("==============================\n")

    print(answer)

    print("\n==============================")
    print("SOURCES")
    print("==============================")

    seen_sources = set()

    for metadata in metadatas:

        source = metadata["source"]
        page = metadata["page"]

        source_text = f"{source} - Page {page}"

        if source_text not in seen_sources:

            print("📄", source_text)

            seen_sources.add(source_text)