import chromadb
from sentence_transformers import SentenceTransformer


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Connect to ChromaDB
client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_collection(
    name="documind_v2"
)


# Ask the user a question
question = input("Ask a question about the document: ")


# Convert question into an embedding
question_embedding = model.encode([question])


# Retrieve the most relevant chunks
results = collection.query(
    query_embeddings=question_embedding.tolist(),
    n_results=3
)


# Display results
print("\nRelevant information:\n")

for i in range(len(results["documents"][0])):

    document = results["documents"][0][i]
    metadata = results["metadatas"][0][i]

    print(f"--- Result {i + 1} ---")
    print(document)

    print(
        f"Source: {metadata['source']}, "
        f"Page: {metadata['page']}"
    )

    print()