import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader


# -----------------------------
# 1. Read the PDF
# -----------------------------

pdf_path = "data/sample.pdf"

reader = PdfReader(pdf_path)

full_text = ""

for page in reader.pages:
    text = page.extract_text()

    if text:
        full_text += text + "\n"


# -----------------------------
# 2. Create chunks
# -----------------------------

chunk_size = 500

chunks = []

for i in range(0, len(full_text), chunk_size):
    chunk = full_text[i:i + chunk_size]
    chunks.append(chunk)


print("Number of chunks:", len(chunks))


# -----------------------------
# 3. Load embedding model
# -----------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# 4. Create embeddings
# -----------------------------

embeddings = model.encode(chunks)

print("Embeddings created!")


# -----------------------------
# 5. Create ChromaDB
# -----------------------------

client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_or_create_collection(
    name="documind"
)


# -----------------------------
# 6. Store chunks + embeddings
# -----------------------------

collection.add(
    ids=[str(i) for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings.tolist()
)

print("Chunks stored in ChromaDB!")