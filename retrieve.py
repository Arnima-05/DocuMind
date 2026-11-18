import chromadb
from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to our existing ChromaDB
client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_collection(
    name="documind"
)

# Ask a question
question = input("Ask a question about the document: ")

# Convert question into an embedding
question_embedding = model.encode([question])

# Search ChromaDB
results = collection.query(
    query_embeddings=question_embedding.tolist(),
    n_results=3
)

# Display the retrieved chunks
print("\nRelevant chunks:\n")

for i, document in enumerate(results["documents"][0]):
    print(f"--- Result {i + 1} ---")
    print(document)
    print()