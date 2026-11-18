from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

text = "Compute isolation separates CPU and memory between cloud users."

embedding = model.encode(text)

print("Embedding created!")
print("Number of values:", len(embedding))
print("First 10 values:")
print(embedding[:10])