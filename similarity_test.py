from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [
    "Compute isolation separates CPU and memory between cloud users.",
    "Network isolation separates network traffic between cloud tenants.",
    "Storage isolation protects stored data from unauthorized access."
]

question = "How does cloud computing prevent one user's CPU from affecting another user?"

text_embeddings = model.encode(texts)
question_embedding = model.encode([question])

similarities = cosine_similarity(
    question_embedding,
    text_embeddings
)

for i, score in enumerate(similarities[0]):
    print(f"Text {i + 1} similarity: {score:.4f}")